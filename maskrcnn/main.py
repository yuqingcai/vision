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
from validate import validate
from tensorflow.python.client import device_lib

os.environ["GPU_ENABLE"] = "TRUE"

if os.environ.get("GPU_ENABLE", "FALSE") == "FALSE":
    tf.config.set_visible_devices([], 'GPU')

mixed_precision.set_global_policy('mixed_float16')

coco_root = '../../dataset/coco2017/'
ann_file_train = os.path.join(coco_root, 'annotations/instances_train2017.json')
img_dir_train = os.path.join(coco_root, 'train2017')
ann_file_validate = os.path.join(coco_root, 'annotations/instances_val2017.json')
img_dir_validate = os.path.join(coco_root, 'val2017')


def save_model(name):
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save(f'saved_model/{name}')


def log_loss(epoch, step, loss, d0, d1):
    print(
        f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} '
            f'epoch {epoch}, step {step}, '
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


if __name__ == '__main__':
    
    for x in device_lib.list_local_devices():
        if x.device_type == 'GPU':
            print(f"{x.name}: {x.physical_device_desc}")

    
    strategy = tf.distribute.MirroredStrategy(
        devices=[''
            '/device:GPU:0', 
            '/device:GPU:1'
        ]
    )
    
    batch_size = 4
    min_size = 800
    max_size = 1000
    accumulation_steps = 8

    ds_train, train_len = create_dataset(
        ann_file=ann_file_train, 
        img_dir=img_dir_train, 
        batch_size=batch_size, 
        min_size=min_size, 
        max_size=max_size
    )
    ds_train = ds_train.shuffle(buffer_size=1000)
    ds_train = strategy.experimental_distribute_dataset(ds_train)
    
    ds_validate, validate_len = create_dataset(
        ann_file=ann_file_validate, 
        img_dir=img_dir_validate, 
        batch_size=batch_size, 
        min_size=min_size, 
        max_size=max_size
    )
    
    with strategy.scope():
        num_gpus = strategy.num_replicas_in_sync                        

        model = MaskRCNN(
            input_shape=(None, None, 3),
            backbone_type='resnet101'
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=2e-2),
        )

        dummy_images = tf.zeros(
            [batch_size//num_gpus, 800, 800, 3], 
            dtype=tf.float32
        )
        dummy_sizes = tf.zeros(
            [batch_size//num_gpus, 2], 
            dtype=tf.int32
        )
        model(dummy_images, dummy_sizes, training=True)

        model.summary()
        
        steps_per_epoch = train_len // batch_size //accumulation_steps
        print(f'train steps per epoch: {steps_per_epoch}')
        
        epochs = 10
        for epoch in range(epochs):
            t_0 = time.time()
            iterator = iter(ds_train)

            for step in range(steps_per_epoch):
                t_1 = time.time()

                loss = strategy.run(model.train_step, args=(accumulation_steps, iterator))
                
                d0 = time.time() - t_0
                d1 = time.time() - t_1

                log_loss(epoch, step, loss, d0, d1)

            model.reset_metrics()
            del iterator
            
#            validate(
#                epoch, 
#                model, 
#                ann_file_validate, 
#                ds_validate, 
#                validate_len, 
#                batch_size
#            )
#            save_model(f'mask_rcnn_{batch_size}_{accumulation_steps}_{steps_per_epoch}_{epoch}.keras')
