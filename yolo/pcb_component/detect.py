from ultralytics import YOLO

model = YOLO("yolo11l.yaml")

results = model.train(
    data="data.yaml", 
    epochs=100, 
    imgsz=1024, 
    device=[0, 1], 
    pretrained=False
)
