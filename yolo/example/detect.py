from ultralytics import YOLO

# model = YOLO("../../ultralytics/runs/detect/train7/weights/best.pt")
model = YOLO("yolo11n.yaml")

results = model.train(
    data="coco.yaml", 
    epochs=1, 
    imgsz=640, 
    device=[0, 1], 
    pretrained=False
)
