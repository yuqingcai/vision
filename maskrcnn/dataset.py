import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import functools
import numpy as np


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
            # print(f"Warning: No annotations found for {file_path}.")
            continue
        
        # Collect all bounding boxes and class_ids
        bboxes = []
        category_ids = []
        segmentations = []
        for annotation in annotations:
            if annotation.get('iscrowd', 0) == 1:
                # Skip images with crowd annotations
                # print(f"Warning: Skipping crowd annotation in {file_path}.")
                continue
            
            if 'bbox' in annotation and 'category_id' \
                and 'segmentation' in annotation \
                and isinstance(annotation['segmentation'], list):
                
                # category_id handling
                category_ids.append(annotation['category_id'])

                # bboxes handling
                x, y, w, h = annotation['bbox']
                bboxes.append([x, y, x+w, y+h])

                # segmentation handling
                segmentations.append(annotation['segmentation'])
        
        # If no bounding boxes or category_ids, skip this image
        if not bboxes or not category_ids or not segmentations:
            print(f"Warning: Annotation details not found for {file_path}.")
            continue
        
        if len(bboxes) != len(category_ids) or \
            len(bboxes) != len(segmentations):
            print(f"Warning: Annotation details mismatch in {file_path}.")
            continue
        
        data.append({
            'file_path' : file_path, 
            'size' :  (height, width), 
            'bboxes' : bboxes,
            'category_ids' : category_ids,
            'segmentations' : segmentations, 
        })

    return data


def scale_to(size, min_size=800, max_size=1333):
    h = tf.cast(size[0], tf.float32)
    w = tf.cast(size[1], tf.float32)
    scale = tf.minimum(min_size / tf.minimum(h, w), \
                max_size / tf.maximum(h, w))
    return scale


def resize_to(size, scale):
    h = tf.cast(size[0], tf.float32)
    w = tf.cast(size[1], tf.float32)
    new_h = tf.cast(tf.round(h * tf.cast(scale, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(w * tf.cast(scale, tf.float32)), tf.int32)
    return tf.stack([new_h, new_w])


def resize_image(image, scale):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    new_hw = resize_to(tf.stack([h, w]), scale)
    resized = tf.image.resize(image, new_hw, method='bilinear')
    return resized


def padding_image(image, height=1400, width=1400):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    
    pad_h = tf.subtract(tf.cast(height, tf.int32), h)
    pad_w = tf.subtract(tf.cast(width, tf.int32), w)

    tf.debugging.assert_non_negative(
        [pad_h, pad_w],
        message="Image is larger than target size. Check h and w."
    )

    padded = tf.image.pad_to_bounding_box(image, 0, 0, height, width)
    return padded


def max_size_in_batch(sizes):
    max_height = tf.reduce_max(sizes[:, 0])
    max_width = tf.reduce_max(sizes[:, 1])
    return max_height, max_width


def load_image(file_path, scale, padding_hw):
    img_raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized = resize_image(image, scale)
    padded = padding_image(resized, height=padding_hw[0], 
                           width=padding_hw[1])
    return padded


def resize_bboxes(bboxes, scale):
    return bboxes * tf.cast(scale, tf.float32)


def resize_segmentations(segmentations, scale):
    return segmentations * tf.cast(scale, tf.float32)


def generator(entries):
    for entry in entries:
        yield {
            'file_path': tf.constant(
                entry['file_path'], 
                dtype=tf.string
                ),
            'size': tf.constant(
                entry['size'], 
                dtype=tf.int32
                ),
            'bboxes': tf.ragged.constant(
                entry['bboxes'], 
                dtype=tf.float32
                ),
            'category_ids': tf.constant(
                entry['category_ids'], 
                dtype=tf.int32
                ),
            'segmentations': tf.ragged.constant(
                entry['segmentations'], 
                dtype=tf.float32
                ),
        }


def polygons_to_mask(segmentations, height, width):
    rles = maskUtils.frPyObjects(segmentations, height, width)
    merged_rle = maskUtils.merge(rles)
    mask = maskUtils.decode(merged_rle)
    if len(mask.shape) == 3:
        mask = np.max(mask, axis=2)
    return mask


def preprocess(batch):
    min_size = 800
    max_size = 1333
    
    scales = tf.map_fn(
        lambda size: scale_to(size, min_size, max_size),
        elems=batch['size'],
        fn_output_signature=tf.TensorSpec(
            shape=(), 
            dtype=tf.float32
        )
    )
    
    resizes = tf.map_fn(
        lambda args: resize_to(args[0], args[1]),
        (batch['size'], scales),
        fn_output_signature=tf.TensorSpec(
            shape=(2,), 
            dtype=tf.int32
        )
    )
    
    padding_hw = max_size_in_batch(resizes)

    images = tf.map_fn(
        lambda args: load_image(args[0], args[1], padding_hw), 
        (batch['file_path'], scales),
        fn_output_signature=tf.TensorSpec(
            shape=(None, None, 3), 
            dtype=tf.float32
        )
    )

    bboxes_resized = tf.map_fn(
        lambda args: resize_bboxes(args[0], args[1]), 
        (batch['bboxes'], scales),
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, 4), 
            dtype=tf.float32
        )
    )

    segmentations_resized = tf.map_fn(
        lambda args: resize_segmentations(args[0], args[1]),
        (batch['segmentations'], scales),
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, None, None), 
            dtype=tf.float32
        )
    )

    batch['image'] = images
    batch['origin_size'] = batch['size']
    batch['size'] = resizes
    batch['bboxes'] = bboxes_resized
    batch['segmentations'] = segmentations_resized
    
    return batch


def create_dataset(ann_file, img_dir, batch_size=4):
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    entries = load_image_info(coco, img_ids, img_dir)

    ds = tf.data.Dataset.from_generator(
        functools.partial(generator, entries), 

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
        output_signature={
            'file_path': tf.TensorSpec(
                shape=(), 
                dtype=tf.string
                ),

            'size': tf.TensorSpec(
                shape=(2,), 
                dtype=tf.int32
                ), 

            'bboxes': tf.RaggedTensorSpec(
                shape=(None, 4), 
                dtype=tf.float32
                ),

            'category_ids': tf.TensorSpec(
                shape=(None,), 
                dtype=tf.int32),

            # 一个 segmentation 是一个2维 List，由若干个 polygon 构成，
            # segmentation = [ [], [], []...]，
            # pycocotools.mask.merge 用于把多个 polygon 合并成一个。
            # 每张图片的 segmentations就是 N 个 List，每个 List 的长
            # 度不固定，shape=(None, None, None)
            'segmentations': tf.RaggedTensorSpec(
                shape=(None, None, None), 
                dtype=tf.float32
                ),
        })
    
    ds = ds.ragged_batch(batch_size)    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
