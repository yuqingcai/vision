import ultralytics
from ultralytics import YOLO

model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")

results = model.train(
    data="coco-seg.yaml", 
    epochs=2, 
    imgsz=640, 
    device=[0, 1]
)

