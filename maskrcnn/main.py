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
import itertools
import random
from tensorflow.keras import mixed_precision

os.environ["GPU_ENABLE"] = "FALSE"

if os.environ.get("GPU_ENABLE", "FALSE") == "FALSE":
    tf.config.set_visible_devices([], 'GPU')


if os.environ.get("GPU_ENABLE", "FALSE") == "TRUE":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# mixed_precision.set_global_policy('mixed_float16')

coco_root = '../../dataset/coco2017/'
train_img_dir = os.path.join(coco_root, 'train2017')
ann_file = os.path.join(coco_root, 'annotations/instances_train2017.json')

if __name__ == '__main__':
    batch_size = 4
    ds_train = create_dataset(
        ann_file=ann_file,
        img_dir=train_img_dir,
        batch_size=batch_size,
        shuffle=False
    )

    model = MaskRCNN(
        input_shape=(None, None, 3),
        batch_size=batch_size,
        backbone_type='resnet101'
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    )
    model.summary()

    epochs = 10
    t_0 = time.time()
    for epoch in range(epochs):
        for step, batch in enumerate(ds_train):
            t_1 = time.time()
            loss = model.train_step(batch)
            d0 = time.time() - t_0
            d1 = time.time() - t_1
            print(f'epoch {epoch}, step {step}, '
                  f'class_loss: {loss["class_loss"]:.4f}, '
                  f'bbox_loss: {loss["bbox_loss"]:.4f}, '
                  f'mask_loss: {loss["mask_loss"]:.4f}, '
                  f'total_loss: {loss["total_loss"]:.4f}, '
                  f'setp_duration: {d1:.2f}s, '
                  f'total_duration: {d0:.2f}s'
            )
        model.reset_metrics()
