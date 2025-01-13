import sys
import time
import joblib
import tensorflow as tf
import pandas as pd
from ultralytics import YOLO

YOLO_MODEL_PATH = 'src/cv/models/teddy_model/weights/best.pt'
LSTM_MODEL_PATH = 'src/ml/models/teddy/teddy_state_predictor.h5'
SCALER_PATH     = 'src/ml/models/teddy/scaler.pkl'

COLS = [
    "frame_idx",
    "teddy_state",
    "teddy_confidence",
    "teddy_x",
    "teddy_y",
    "teddy_w",
    "teddy_h",
    "teddy_dx",
    "teddy_dy",
    "closest_toy_confidence",
    "closest_toy_x",
    "closest_toy_y",
    "closest_toy_w",
    "closest_toy_h",
    "closest_toy_dx",
    "closest_toy_dy",
    "water_confidence",
    "water_x",
    "water_y",
    "water_w",
    "water_h",
]

priority_state = [
    {"STATE": "TEDDY",         "LABEL": "teddy",         "PRIORITY": 2},
    {"STATE": "TEDDY_LYING",   "LABEL": "teddy_lying",   "PRIORITY": 1},
    {"STATE": "TEDDY_PLAYING", "LABEL": "teddy_play",    "PRIORITY": 3},
    {"STATE": "TEDDY_HOWLING", "LABEL": "teddy_howling", "PRIORITY": 4},
]


def extract_detections(result, prev_frame, frame_idx):
    detection_data = {
        "frame_idx": frame_idx,
        "teddy_state": "none",
        "teddy_confidence": -1,
        "teddy_x": -1,
        "teddy_y": -1,
        "teddy_w": -1,
        "teddy_h": -1,
        "teddy_dx": 0,
        "teddy_dy": 0,
        "closest_toy_confidence": -1,
        "closest_toy_x": -1,
        "closest_toy_y": -1,
        "closest_toy_w": -1,
        "closest_toy_h": -1,
        "closest_toy_dx": 0,
        "closest_toy_dy": 0,
        "water_confidence": -1,
        "water_x": -1,
        "water_y": -1,
        "water_w": -1,
        "water_h": -1,
    }

    boxes = result.boxes
    names = result.names

    best_teddy_conf = 0.0
    best_toy_conf   = 0.0
    best_water_conf = 0.0
    best_teddy_label = "none"

    for box in boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = names[class_id].lower()

        if any(label.startswith(x) for x in ["teddy", "teddy_lying", "teddy_howling", "teddy_play"]):
            if conf > best_teddy_conf:
                best_teddy_conf = conf
                x, y, w, h = box.xywh[0].tolist()
                detection_data["teddy_confidence"] = conf
                detection_data["teddy_x"] = x
                detection_data["teddy_y"] = y
                detection_data["teddy_w"] = w
                detection_data["teddy_h"] = h
                best_teddy_label = label

        elif label == "toy":
            if conf > best_toy_conf:
                best_toy_conf = conf
                x, y, w, h = box.xywh[0].tolist()
                detection_data["closest_toy_confidence"] = conf
                detection_data["closest_toy_x"] = x
                detection_data["closest_toy_y"] = y
                detection_data["closest_toy_w"] = w
                detection_data["closest_toy_h"] = h

        elif label == "water":
            if conf > best_water_conf:
                best_water_conf = conf
                x, y, w, h = box.xywh[0].tolist()
                detection_data["water_confidence"] = conf
                detection_data["water_x"] = x
                detection_data["water_y"] = y
                detection_data["water_w"] = w
                detection_data["water_h"] = h

    if best_teddy_conf > 0.0:
        detection_data["teddy_state"] = best_teddy_label
    else:
        detection_data["teddy_state"] = "none"

    if prev_frame is not None:
        if detection_data["teddy_x"] != -1 and prev_frame["teddy_x"] != -1:
            detection_data["teddy_dx"] = detection_data["teddy_x"] - prev_frame["teddy_x"]
            detection_data["teddy_dy"] = detection_data["teddy_y"] - prev_frame["teddy_y"]

        if detection_data["closest_toy_x"] != -1 and prev_frame["closest_toy_x"] != -1:
            detection_data["closest_toy_dx"] = (detection_data["closest_toy_x"] 
                                                - prev_frame["closest_toy_x"])
            detection_data["closest_toy_dy"] = (detection_data["closest_toy_y"] 
                                                - prev_frame["closest_toy_y"])

    return detection_data


def run_lstm_inference(frames_buffer, model_lstm, scaler):
    df = pd.DataFrame(frames_buffer)

    feature_cols = [
        "teddy_x", "teddy_y", "teddy_w", "teddy_h", 
        "teddy_dx", "teddy_dy", 
        "closest_toy_x", "closest_toy_y", 
        "closest_toy_dx", "closest_toy_dy", 
        "water_x", "water_y"
    ]

    df[feature_cols] = scaler.transform(df[feature_cols])

    sequence = df[feature_cols].values.reshape(1, len(frames_buffer), len(feature_cols))
    pred_probs = model_lstm.predict(sequence, verbose=0)[0]

    state_probs = {
        priority_state[i]["STATE"]: float(pred_probs[i])
        for i in range(len(priority_state))
    }
    predicted_next_state = max(state_probs.items(), key=lambda x: x[1])[0]

    return predicted_next_state, state_probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]

    print(f"[INFO] Loading YOLO model from: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    times_file = open('end-to-end-times.txt', 'w')
    times_file.write("Frame, YOLO + LSTM inference time:\n")

    results_file = open('end-to-end-results.txt', 'w')
    results_file.write("Frame, LSTM predicted next state, State probabilities, Current State:\n")

    print(f"[INFO] Loading LSTM model from: {LSTM_MODEL_PATH}")
    model_lstm = tf.keras.models.load_model(
        LSTM_MODEL_PATH,
        compile=False,
        custom_objects={
            'LSTM': lambda **kwargs: tf.keras.layers.LSTM(
                kwargs.pop('units'),
                **{k: v for k, v in kwargs.items() if k != 'time_major'}
            )
        }
    )
    print(f"[INFO] Loading scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    sequence_length = 10 # This should match the sequence length the lstm model was trained on
    frames_buffer   = []
    prev_frame      = None
    consecutive_teddy_frames = 0
    prev_predicted_state = None
    state_probs = {}

    print(f"[INFO] Starting inference on: {video_path}")
    results_iterator = yolo_model.predict(
        source=video_path,
        conf=0.5,
        iou=0.5,
        imgsz=640,
        device='cpu',
        stream=True
    )

    frame_idx = 0
    low_inference_mode = False
    frame_counter = 0
    
    while True:
        start_time = time.perf_counter()

        frame_counter = (frame_counter + 1) % 3

        should_skip = (low_inference_mode and frame_counter != 0)
        
        if should_skip:
            frame_idx += 1
            predicted_state = prev_predicted_state
            # state_probs = {}

            end_time = time.perf_counter()
            total_inference_time_ms = (end_time - start_time) * 1000.0

            print(f"Frame {frame_idx} SKIPPED inference.")
            print(f"  (Maintaining previous state: {predicted_state})")
            print(f"  Total YOLO + LSTM inference time: {total_inference_time_ms:.2f} ms\n")

            times_file.write(f"{frame_idx}, {total_inference_time_ms:.2f} ms\n")
            results_file.write(f"{frame_idx}, {predicted_state}, {state_probs}, {frames_buffer[-1]['teddy_state']}\n")
            continue

        try:
            result = next(results_iterator)
        except StopIteration:
            break

        frame_idx += 1

        current_frame = extract_detections(result, prev_frame, frame_idx)
        frames_buffer.append(current_frame)
        prev_frame = current_frame

        predicted_state = None
        state_probs = {}
        
        if len(frames_buffer) >= sequence_length:
            last_n_frames = frames_buffer[-sequence_length:]
            predicted_state, state_probs = run_lstm_inference(last_n_frames, model_lstm, scaler)
            prev_predicted_state = predicted_state

            if prev_frame["teddy_state"] == "teddy_lying":
                consecutive_teddy_frames += 1
                if consecutive_teddy_frames >= 5:
                    low_inference_mode = True
            else:
                consecutive_teddy_frames = 0
                low_inference_mode = False

        end_time = time.perf_counter()
        total_inference_time_ms = (end_time - start_time) * 1000.0

        print(f"Frame {frame_idx} inference complete.")
        if predicted_state is not None:
            print(f"  LSTM predicted next state: {predicted_state}")
            print(f"  State probabilities: {state_probs}")
        print(f"  Total YOLO + LSTM inference time: {total_inference_time_ms:.2f} ms\n")

        times_file.write(f"{frame_idx}, {total_inference_time_ms:.2f} ms\n")
        results_file.write(f"{frame_idx}, {predicted_state}, {state_probs}, {frames_buffer[-1]['teddy_state']}\n")

    times_file.close()
    results_file.close()

    print("[INFO] Inference complete.")


if __name__ == "__main__":
    main()
