import ultralytics
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")

results = model.train(
    data="coco.yaml", 
    epochs=100, 
    imgsz=640, 
    device=[0, 1], 
    pretrained=False
)
