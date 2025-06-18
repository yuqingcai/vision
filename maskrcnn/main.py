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


def show_image(sample, i):
    img = sample['image'][i]
    size = sample['size'][i]
    bboxes = sample['bboxes'][i]
    category_ids = sample['category_ids'][i]
    masks = sample['masks'][i]
    padded_size = tf.shape(img)

    print(f'padded_size: {padded_size[0]}x{padded_size[1]}')

    plt.figure()
    plt.imshow(img.numpy())
    plt.axis('off')
    plt.gca().set_aspect('auto')

    # size border
    rect = patches.Rectangle((0, 0), size[1], size[0], 
                             linewidth=1, 
                             edgecolor='green',
                             facecolor='none',
                             alpha=1.0
    )
    plt.gca().add_patch(rect)
    
    for bbox, category_id, mask in \
        zip(bboxes, category_ids, masks):

        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, 
                                 y2 - y1, linewidth=1,
                                 edgecolor='r', 
                                 facecolor='none', 
                                 alpha=0.4
        )
        plt.gca().add_patch(rect)

        cid = int(category_id.numpy()) if \
            hasattr(category_id, 'numpy') else int(category_id)
        plt.gca().text(x1, y1 - 2, str(cid), 
                        color='yellow', fontsize=6, 
                        va='bottom', ha='left', 
                        bbox=dict(facecolor='black', alpha=0.5, pad=0))
        
        mask = mask.numpy() if hasattr(mask, 'numpy') else mask
        highlight = np.zeros((*mask.shape, 4), dtype=np.float32)
        highlight[mask == 1] = [1, 0, 0, 0.5]
        plt.imshow(highlight)
        

if __name__ == '__main__':
    batch_size = 4
    ds_train = create_dataset(
        ann_file=ann_file,
        img_dir=train_img_dir,
        batch_size=batch_size
    )

    # time_0 = time.time()
    # for i, sample in enumerate(ds_train):
    #     print(f"Batch {i}:")
    # time_1 = time.time()
    # print(f"Time taken to iterate through dataset: {time_1 - time_0:.2f} seconds")
    
    # n = random.randint(0, 1000)
    # sample = next(itertools.islice(ds_train, n-1, n))

    sample = next(iter(ds_train))
    for i in range(batch_size):
        show_image(sample, i)
    plt.show()
