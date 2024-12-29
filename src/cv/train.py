import sys
import os
from ultralytics import YOLO

def train_model(data_path):
    model = YOLO('yolov8n.pt')

    _results = model.train(
        data=data_path,
        epochs=25,
        imgsz=640,
        batch=16,
        device='cpu',
        project='src/cv/models',
        name='teddy_model'
    )

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_data_yaml>")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f"Error: {sys.argv[1]} does not exist")
        sys.exit(1)

    data_path = sys.argv[1]

    print("Training CV Model: " + data_path)
    train_model(data_path)
