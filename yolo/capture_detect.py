import cv2
from ultralytics import YOLO


# coco训练集产出的模型
# model = YOLO("detect/train7/weights/best.pt")
model = YOLO("detect/train17/weights/best.pt")

# cap = cv2.VideoCapture('./IMG_1117_720.MOV')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("cap open failed.")
    exit()
    
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("cap read failed.")
        break
    
    results = model(frame)
    for result in results:
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        
        for i in range(len(names)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            label = f"{names[i]} {confs[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


