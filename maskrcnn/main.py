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

# tf.config.set_visible_devices([], 'GPU')
   
def resize_image(img, min_size=800, max_size=1333):
    h, w = img.shape[:2]
    scale = min(min_size / min(h, w), max_size / max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = tf.image.resize(img, (new_h, new_w), method='bilinear')
    return img_resized, scale


def padding_image(img, target_height=1400, target_width=1400):
    h, w = img.shape[:2]
    pad_h = target_height - h
    pad_w = target_width - w
    if pad_h < 0 or pad_w < 0:
        print(f'w: {w}, h: {h}')
        raise ValueError("Image is larger than target size.")
    img_padded = tf.image.pad_to_bounding_box(
        img, 0, 0, target_height, target_width)
    return img_padded


def expand_batch_axis(img):
    return tf.expand_dims(img, axis=0)


def image_tensor(file_name):
    path = f'../../dataset/coco2017/train2017/{file_name}'
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img_resized, scale = resize_image(img, min_size=800, max_size=1333)
    return img_resized, scale


if __name__ == "__main__":

    ann_path = '../../dataset/coco2017/annotations/instances_train2017.json'
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()

    batch = []
    max_width = 0
    max_height = 0

    n = len(image_ids)
    indices = np.random.choice(n, size=10, replace=False)

    for i in indices:
        img_id = image_ids[i]
        
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        origin_height = img_info['height']
        origin_width = img_info['width']
        img, scale = image_tensor(file_name)
        target_height = int(round(origin_height * scale))
        target_width = int(round(origin_width * scale))

        batch.append({
            'image': img, 
            'scale': scale, 
            'image_id': img_id, 
            'file_name': file_name, 
            'origin_height': origin_height, 
            'origin_width': origin_width,
            'target_height': target_height,
            'target_width': target_width
            })
        
        if target_height > max_height:
            max_height = target_height

        if target_width > max_width:
            max_width = target_width

    images = []
    for item in batch:
        img = item['image']
        img = padding_image(
            img, 
            target_height=max_height, 
            target_width=max_width
            )
        img = expand_batch_axis(img)
        item['image'] = img
        images.append(img)
    
    images = tf.concat(images, axis=0)
    print(f'images shape: {images.shape}')

    for item in batch:
        img = item['image']
        img_np = img[0].numpy()
        plt.figure()
        plt.title(f"Image ID: {item['image_id']}")
        plt.imshow(img_np)

    plt.show()



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