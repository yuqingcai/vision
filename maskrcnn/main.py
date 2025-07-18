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

mixed_precision.set_global_policy('mixed_float16')

coco_root = '../../dataset/coco2017/'
train_img_dir = os.path.join(coco_root, 'train2017')
ann_file = os.path.join(coco_root, 'annotations/instances_train2017.json')

if __name__ == '__main__':
    
    batch_size = 2
    ds_train = create_dataset(
        ann_file=ann_file,
        img_dir=train_img_dir,
        batch_size=batch_size,
        shuffle=False,
        min_size=800,   # 800
        max_size=1333   # 1333
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
    for epoch in range(epochs):
        t_0 = time.time()
        for step, batch in enumerate(ds_train):
            t_1 = time.time()
            loss = model.train_step(
                batch['image'],
                batch['size'],
                batch['bbox'],
                batch['mask'],
                batch['label'],
            )
            d0 = time.time() - t_0
            d1 = time.time() - t_1
            print(f'epoch {epoch}, step {step}, '
                  f'l_objectness: {loss["loss_objectness"]:.4f}, '
                  f'l_rpn_box_reg: {loss["loss_rpn_box_reg"]:.4f}, '
                  f'l_class: {loss["loss_class"]:.4f}, '
                  f'l_box_reg: {loss["loss_box_reg"]:.4f}, '
                  f'l_mask: {loss["loss_mask"]:.4f}, '
                  f'l_total: {loss["loss_total"]:.4f}, '
                  f'setp_t: {d1:.2f}s, '
                  f'total_t: {d0:.2f}s '
                  f'av_t: {d0/(step+1):.2f}s '
            )
        
        model.reset_metrics()
