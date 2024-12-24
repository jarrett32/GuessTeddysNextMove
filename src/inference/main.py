from ultralytics import YOLO
import sys

def inference(video_path):
    model = YOLO('runs/train/teddy_model/weights/best.pt')
    results = model.predict(video_path, conf=0.5, iou=0.5, imgsz=640, device='cpu')
    
    # Open output file to write results
    with open('output.txt', 'w') as f:
        # Process each frame's results
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (in x, y, width, height format)
                x, y, w, h = box.xywh[0].tolist()
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Write to file: frame_number, class, confidence, x, y, width, height
                f.write(f"Frame {frame_idx}: {class_name} ({confidence:.2f}): x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}\n")
    
    return results

if __name__ == '__main__':
    # Use command line argument if provided, otherwise use default
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'data/test_vids/output1.mp4'
    inference(video_path)
