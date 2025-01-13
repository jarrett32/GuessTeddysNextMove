import sys
import time

from ultralytics import YOLO

model_path = 'src/cv/models/teddy_model/weights/best.pt'

def inference(video_path):
    model = YOLO(model_path)
    
    with open('output.txt', 'a') as f, open('output_times_cv.txt', 'a') as tf:
        tf.write("Frame,Inference_Time_ms,Num_Detections,Mean_Confidence\n")
        
        results_iterator = model.predict(video_path, conf=0.5, iou=0.5, imgsz=640,
                                      device='cpu', stream=True)
        frame_idx = 0
        
        while True:
            start_time = time.perf_counter()
            try:
                result = next(results_iterator)
                inference_time = (time.perf_counter() - start_time) * 1000
            except StopIteration:
                break
            
            boxes = result.boxes
            num_detections = len(boxes)
            mean_confidence = sum(float(box.conf[0]) for box in boxes) / num_detections if num_detections > 0 else 0
            
            tf.write(f"{frame_idx},{inference_time:.4f},{num_detections},{mean_confidence:.3f}\n")
            
            for box in boxes:
                x, y, w, h = box.xywh[0].tolist()
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                f.write(f"Frame {frame_idx}: {class_name} ({confidence:.2f}): "
                        f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}\n")
                
            frame_idx += 1

    return result

if __name__ == '__main__':
    print("Inference CV: " + model_path)
    if len(sys.argv) < 2:
        print("Usage: python inference.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    results = inference(video_path)
