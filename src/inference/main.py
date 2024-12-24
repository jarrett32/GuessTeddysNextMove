import sys

from ultralytics import YOLO


def inference(video_path):
    model = YOLO('runs/train/teddy_model/weights/best.pt')
    results = model.predict(video_path, conf=0.5, iou=0.5, imgsz=640, device='cpu')

    with open('output.txt', 'w') as f:
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            for box in boxes:
                x, y, w, h = box.xywh[0].tolist()
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])

                f.write(f"Frame {frame_idx}: {class_name} ({confidence:.2f}): "
                        f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}\n")

    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    inference(video_path)
