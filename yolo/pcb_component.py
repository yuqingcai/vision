import cv2
import numpy as np
import os

image_dir = "/Volumes/VolumeEXT/Project/NNDL/datasets/components_data_uncropped/train/images/"
label_dir = "/Volumes/VolumeEXT/Project/NNDL/datasets/components_data_uncropped/train/labels/"
image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))[30:50]]
label_files = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))[30:50]]

for idx, (img_path, label_path) in enumerate(zip(image_files, label_files)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image load failed: {img_path}")
        continue
    h, w = img.shape[:2]
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_center, y_center, bw, bh = map(float, parts)
                x_center, y_center, bw, bh = x_center * w, y_center * h, bw * w, bh * h
                x1 = int(x_center - bw / 2)
                y1 = int(y_center - bh / 2)
                x2 = int(x_center + bw / 2)
                y2 = int(y_center + bh / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.imshow(f'Image_{idx}', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
