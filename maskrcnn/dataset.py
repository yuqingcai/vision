import tensorflow as tf
from pycocotools.coco import COCO
import os
import numpy as np
import functools


def load_image_info(coco, img_ids, img_dir):
    data = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_path = os.path.join(img_dir, img_info['file_name'])
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        bboxes = []
        # classes = []
        for ann in anns:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x+w, y+h])
            else:
                tf.print(f'Warning: No bbox for annotation {ann["id"]} in image {img_id}')
                bboxes.append([])
                # classes.append(ann['category_id'])

        data.append({
            'file_path': file_path,
            'bboxes' : bboxes
        })
    return data


def generator(entries):
    for entry in entries:
        yield {
            'file_path': entry['file_path'],
            'bboxes': tf.ragged.constant(
                entry['bboxes'], 
                dtype=tf.float32)
        }


def preprocess(batch):
    tf.print(batch)
    # for item in batch:
    #     print(f'oops!!!! {item}')
    # for item in batch:
    #     img = tf.io.read_file(item['file_path'])
    #     img = tf.image.decode_jpeg(img, channels=3)
    #     img = tf.image.convert_image_dtype(img, tf.float32)

    return 'oops1', 'oops2'  # Placeholder for actual preprocessing logic


def create_dataset(ann_file, img_dir, batch_size=4):
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    entries = load_image_info(coco, img_ids, img_dir)

    ds = tf.data.Dataset.from_generator(
        functools.partial(generator, entries), 
        output_signature={   
            'file_path': tf.TensorSpec(
                shape=(), 
                dtype=tf.string),
            'bboxes': tf.RaggedTensorSpec(
                shape=(None, 4), dtype=tf.float32),
        })
    
    ds = ds.batch(batch_size)    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
