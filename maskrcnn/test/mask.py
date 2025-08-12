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
from dataset import create_dataset


coco_root = '../../../dataset/coco2017/'
train_img_dir = os.path.join(coco_root, 'train2017')
ann_file = os.path.join(coco_root, 'annotations/instances_train2017.json')

if __name__ == '__main__':
    
    batch_size = 1
    ds_train = create_dataset(
        ann_file=ann_file,
        img_dir=train_img_dir,
        batch_size=batch_size,
        shuffle=False,
        min_size=800,   # 800
        max_size=1333   # 1333
    )
    
    batch_0 = next(iter(ds_train))

    for image, masks, file_path in zip(
        batch_0['image'], 
        batch_0['mask'], 
        batch_0['file_path']):

        file_name = os.path.basename(file_path.numpy().decode('utf-8'))
        fig1 = plt.figure()
        plt.imshow(image.numpy())
        plt.axis('off')
        plt.title(file_name)
        
        for mask in masks:
            fig2 = plt.figure()
            plt.title(file_name)
            plt.imshow(mask.numpy(), alpha=0.5, cmap='gray')
            plt.axis('off')
        
    plt.show()
