import tensorflow as tf
import json
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from model import MaskRCNN

# tf.config.set_visible_devices([], 'GPU')
   
def resize_image(img, min_size=800, max_size=1333):
    h, w = img.shape[:2]
    scale = min(min_size / min(h, w), max_size / max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = tf.image.resize(img, (new_h, new_w), method='bilinear')
    return img_resized, scale


def build_coco_ann_index(ann_path):
    f = open(ann_path, "r")
    coco_anns = json.load(f)
    f.close()

    filename_to_id = {img["file_name"]: img["id"] for img in coco_anns["images"]}
    id_to_anns = defaultdict(list)
    for ann in coco_anns["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)
    return filename_to_id, id_to_anns


def expand_batch_axis(img):
    return tf.expand_dims(img, axis=0)


def image_tensor(file_name):
    path = f'../../dataset/coco2017/train2017/{file_name}'
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img_resized, scale = resize_image(img, min_size=800, max_size=1333)
    img_expand = expand_batch_axis(img_resized)
    return img_expand, scale


if __name__ == "__main__":
    # filename_to_id, id_to_anns = build_coco_ann_index(
    #     '../../dataset/coco/annotations/instances_train2017.json')

    img_0, scale_0 = image_tensor('000000000049.jpg')

    model = MaskRCNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    model.summary()

    # output: [logits, bbox_deltas, proposals]
    output = model(img_0)
    proposals = output[2]
    print(proposals.shape)

    img_np = img_0[0].numpy()  # 去掉 batch 维，转为 numpy
    plt.imshow(img_np)
    plt.figure(figsize=(12, 12))
    plt.imshow(img_np)

    ax = plt.gca()
    for box in proposals.numpy():
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor='r', facecolor='none', alpha=0.4
        )
        ax.add_patch(rect)

    plt.title(f"Proposals ({proposals.shape[0]} boxes)")
    plt.axis('off')
    plt.show()