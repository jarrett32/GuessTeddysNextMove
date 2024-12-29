from ultralytics import YOLO

data_path = 'src/train/teddy/data.yaml'
def train_model():
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
    print("Training CV Model: " + data_path)
    train_model()
