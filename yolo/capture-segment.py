import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("segment/train8/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("cap open failed.")
    exit()
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("cap read failed.")
        break
    
    results = model(frame)
    for result in results:
        xyxy = result.boxes.xyxy
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        confs = result.boxes.conf

        # 画分割mask
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # shape: (num, h, w)
            for mask in masks:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
                color = np.array([0.0, 0.0, 255.0], dtype=np.float32)
                colored_mask = np.zeros_like(frame, dtype=np.float32)
                for i in range(3):
                    colored_mask[..., i] = mask * color[i]
                frame = cv2.addWeighted(frame, 1.0, 
                                        colored_mask.astype(np.uint8), 0.5, 0)

        for i in range(len(names)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            label = f"{names[i]} {confs[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


