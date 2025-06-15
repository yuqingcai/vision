import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import numpy as np
import functools
import json


def load_image_info(coco, img_ids, img_dir):
    data = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_path = os.path.join(img_dir, img_info['file_name'])
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        width = img_info['width']
        height = img_info['height']

        # Ensure the file exists
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist.")
            continue

        if not annotations:
            # If no annotations, skip this image
            print(f"Warning: No annotations found for {file_path}.")
            continue
        
        # Collect all bounding boxes and class_ids
        bboxes = []
        category_ids = []
        segmentations = []
        for annotation in annotations:
            if annotation.get('iscrowd', 0) == 1:
                # Skip images with crowd annotations
                print(f"Warning: Skipping crowd annotation in {file_path}.")
                continue
            
            if 'bbox' in annotation and 'category_id' \
                and 'segmentation' in annotation:
                
                # category_id handling
                category_id = annotation['category_id']
                category_ids.append(category_id)

                # bboxes handling
                x, y, w, h = annotation['bbox']
                bboxes.append([x, y, x+w, y+h])

                # segmentation handling
                # Check if segmentation is a polygon or RLE, then make compress
                segmentation = annotation['segmentation']
                if isinstance(segmentation, list):
                    # polygon
                    rles = maskUtils.frPyObjects(segmentation, height, width)
                    rle = maskUtils.merge(rles)
                elif isinstance(segmentation['counts'], list):
                    # uncompressed
                    rle = maskUtils.frPyObjects(segmentation, height, width)
                    rle = maskUtils.merge(rle)
                else:
                    # compressed
                    rle = segmentation

                if isinstance(rle, list):
                    rle = rle[0]
                # Ensure counts is a string
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('ascii')
                rle_str = json.dumps(rle)
                segmentations.append(rle_str)

        # If no bounding boxes or category_ids, skip this image
        if not bboxes or not category_ids or not segmentations:
            print(f"Warning: No bboxes or class_ids or segmentations found for {file_path}.")
            continue
        
        if len(bboxes) != len(category_ids) or \
            len(bboxes) != len(segmentations):
            print(f"Warning: Mismatch in number of bboxes and category_ids, segmentations for {file_path}.")
            continue
        
        data.append({
            'file_path': file_path,
            'bboxes' : bboxes,
            'category_ids': category_ids,
            'segmentations' : segmentations, 
        })
    return data


def generator(entries):
    for entry in entries:
        yield {
            'file_path': tf.constant(entry['file_path'], dtype=tf.string),
            'bboxes': tf.ragged.constant(entry['bboxes'], dtype=tf.float32),
            'category_ids': tf.constant(entry['category_ids'], dtype=tf.int32),
            'segmentations': tf.constant(entry['segmentations'], dtype=tf.string),
        }


def preprocess(batch):
    # tf.print(batch)
    # for item in batch:
    #     img = tf.io.read_file(item['file_path'])
    #     img = tf.image.decode_jpeg(img, channels=3)
    #     img = tf.image.convert_image_dtype(img, tf.float32)
    return batch


def create_dataset(ann_file, img_dir, batch_size=4):
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    entries = load_image_info(coco, img_ids, img_dir)

    ds = tf.data.Dataset.from_generator(
        functools.partial(generator, entries), 
        # 
        # file_path 是一个标量，shape=()
        # bboxes 是一个第1维长度不固定，第2维长度固定等于4的tensor,
        # 必须使用 RaggedTensorSpec 进行申明，shape=(None, 4)
        # category_ids 是长度不固定的一维Tensor，shape=(None, )
        # segmentations 是长度不固定的一维Tensor，shape=(None, )
        # 
        # 对于shape的申明，None 表示长度不固定
        # 
        # category_ids 和 segmentations，虽然他们的长度不固定，但
        # 这里只是使用 tf.TensorSpec 来申明，因为他们是一维Tensor, 
        # 在 batch 时 TensorFlow 会自动将其转换为 RaggedTensor。
        #
        # 总结，启用batch时，对于长度不固定的一维Tensor，可以使用
        # tf.TensorSpec 对其进行申明，TensorFlow 会自动将其转换成 
        # RaggedTensor，对于多维 Tesnsor，如果某一维的长度不固定，
        # 则必须使用 RaggedTensorSpec 对其申明，指出哪个维度是不等
        # 长的。
        # 
        output_signature={
            'file_path': tf.TensorSpec(shape=(), dtype=tf.string),
            'bboxes': tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32),
            'category_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'segmentations': tf.TensorSpec(shape=(None,), dtype=tf.string),
        })
    
    ds = ds.ragged_batch(batch_size)    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
