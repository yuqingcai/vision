import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import json
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from model import MaskRCNN
from dataset import create_dataset


# tf.config.set_visible_devices([], 'GPU')

coco_root = "../../dataset/coco2017/"
train_img_dir = os.path.join(coco_root, "train2017")
ann_file = os.path.join(coco_root, "annotations/instances_train2017.json")


if __name__ == "__main__":
    ds = create_dataset(
        ann_file=ann_file,
        img_dir=train_img_dir,
        batch_size=4
    )
    
    for i, data in enumerate(ds):
        print(f'data {i}:')
        print(f'file_path:')
        for image_path in data['file_path']:
            print(f'{image_path.numpy().decode()}')

        print(f'category_ids:')
        for category_id in data['category_ids']:
            print(f'{category_id.numpy().tolist()}')

        print(f'bboxes:')
        for bbox in data['bboxes']:
            print(f'{bbox.numpy().tolist()}')


        print(f'segmentations:')
        for segmentation in data['segmentations']:
            print(f'{segmentation.numpy().tolist()}')
        
        print(f'')

    # # 选择一张图像 ID（比如第一张）
    # img_id = image_ids[1]
    # # 获取图像信息
    # img_info = coco.loadImgs(img_id)[0]
    # print("图像信息：", img_info)
    # # 获取该图像的所有标注 ID
    # ann_ids = coco.getAnnIds(imgIds=img_id)
    # # 加载所有标注
    # anns = coco.loadAnns(ann_ids)
    # # 遍历打印标注信息
    # for i, ann in enumerate(anns):
    #     print(f'ann{i}:')
    #     print("类别ID:", ann['category_id'])
    #     print("边框: ", ann['bbox'])  # 格式为 [x, y, width, height]
    #     print("分割: ", ann.get('segmentation'))  # 可能是 polygon 或 RLE
    #     print("是否是 crowd:", ann.get('iscrowd', 0))

    # # 假设 anns 是 COCO 的 annotation 列表，img_info 是图片信息
    # height, width = img_info['height'], img_info['width']
    # binary_mask = np.zeros((height, width), dtype=np.uint8)

    # for ann in anns:
    #     segm = ann['segmentation']
    #     if isinstance(segm, list):  # polygon
    #         rles = maskUtils.frPyObjects(segm, height, width)
    #         rle = maskUtils.merge(rles)
    #     elif isinstance(segm['counts'], list):  # uncompressed RLE
    #         rle = maskUtils.frPyObjects(segm, height, width)
    #     else:  # compressed RLE
    #         rle = segm
    #     m = maskUtils.decode(rle)
    #     binary_mask = np.logical_or(binary_mask, m)

    # plt.imshow(binary_mask, cmap='gray')
    # plt.axis('off')
    # plt.title('Binary Mask')
    # plt.show()
    
    
    # img_0, scale_0 = image_tensor('000000000049.jpg')
    # model = MaskRCNN()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    # model.summary()

    # # output: [logits, bbox_deltas, proposals]
    # output = model(img_0)
    # proposals = output[2]
    # print(proposals.shape)

    # img_np = img_0[0].numpy()  # 去掉 batch 维，转为 numpy
    # plt.imshow(img_np)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img_np)

    # ax = plt.gca()
    # for box in proposals.numpy():
    #     x1, y1, x2, y2 = box
    #     rect = patches.Rectangle(
    #         (x1, y1), x2 - x1, y2 - y1,
    #         linewidth=1, edgecolor='r', facecolor='none', alpha=0.4
    #     )
    #     ax.add_patch(rect)

    # plt.title(f"Proposals ({proposals.shape[0]} boxes)")
    # plt.axis('off')
    # plt.show()