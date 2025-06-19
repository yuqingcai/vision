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

# tf.config.set_visible_devices([], 'GPU')

coco_root = '../../dataset/coco2017/'
train_img_dir = os.path.join(coco_root, 'train2017')
ann_file = os.path.join(coco_root, 'annotations/instances_train2017.json')

if __name__ == '__main__':
    batch_size = 1
    ds_train = create_dataset(
        ann_file=ann_file,
        img_dir=train_img_dir,
        batch_size=batch_size,
        shuffle=False
    )

    model = MaskRCNN(
        input_shape=(None, None, 3),
        batch_size=batch_size,
        backbone_type='resnet50'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    )

    model.summary()

    # Train the model
    epochs = 10
    for epoch in range(epochs):
        for step, batch in enumerate(ds_train):
            loss = model.train_step(batch)
            if step % 100 == 0:
                # print(f'Epoch {epoch}, Step {step}, Loss: {loss.numpy():.4f}')
                print(f'Epoch {epoch}, Step {step}, Loss: {loss}')
        model.reset_metrics()
