from ultralytics import YOLO


def train_model():
    model = YOLO('yolov8n.pt')

    _results = model.train(
        data='src/train/teddy/data.yaml',
        epochs=25,
        imgsz=640,
        batch=16,
        device='cpu',
        project='runs/train',
        name='teddy_model'
    )

if __name__ == '__main__':
    train_model()
