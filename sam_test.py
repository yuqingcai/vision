import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# 1. 加载模型
# sam_checkpoint = "../model/sam_vit_b_01ec64.pth"
# model_type = "vit_b"

# sam_checkpoint = "../model/sam_vit_l_0b3195.pth"
# model_type = "vit_l"

sam_checkpoint = "../model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 2. 准备图像
image_path = "../dataset/coco/test2017/000000000057.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. 创建预测器
predictor = SamPredictor(sam)
predictor.set_image(image)

# 4. 设置提示点（point prompt）
# 这里假设你想分割图像中间的物体，你可以自定义点坐标
input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
input_label = np.array([1])  # 1表示 foreground（前景）

# 5. 执行分割
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# 6. 显示结果
for i, mask in enumerate(masks):
    plt.figure()
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.title(f"Mask {i+1} (score: {scores[i]:.3f})")
    plt.axis("off")
plt.show()
