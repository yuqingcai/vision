import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import functools
import numpy as np
import json
import cv2
import time


def load_image_info(coco, img_ids, img_dir):
    print('loading image info from annotations...')
    time_0 = time.time()

    entries = []
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
        rles = []
        for annotation in annotations:
            if annotation.get('iscrowd', 0) == 1:
                # Skip images with crowd annotations
                # print(f"Warning: Skipping crowd annotation in {file_path}.")
                continue
            
            if 'bbox' in annotation and 'category_id' in annotation \
                and 'segmentation' in annotation:
                
                # category_id handling
                category_ids.append(annotation['category_id'])
                
                # bboxes handling
                x, y, w, h = annotation['bbox']
                bboxes.append([x, y, x+w, y+h])
                # segmentation handling
                rle = rle_from_seg(annotation['segmentation'], 
                                   height, width)
                rles.append(rle)
        
        # If no bounding boxes or category_ids, skip this image
        if not bboxes or not category_ids or not rles:
            print(f"Warning: Annotation details not found for {file_path}.")
            continue
        
        if len(bboxes) != len(category_ids) or len(bboxes) != len(rles):
            print(f"Warning: Annotation details mismatch in {file_path}.")
            continue
        
        entries.append({
            'file_path' : file_path, 
            'size' :  (height, width), 
            'bboxes' : bboxes,
            'category_ids' : category_ids,
            'rles' : rles, 
        })

    time_1 = time.time()
    print(f"Done (t={time_1 - time_0:.2f}s)")
    return entries


def seg_is_rle(seg):
    if not isinstance(seg, dict):
        return False
    if 'counts' not in seg or 'size' not in seg:
        return False
    if not isinstance(seg['size'], (list, tuple)) or len(seg['size']) != 2:
        return False
    counts = seg['counts']
    if isinstance(counts, (list, str, bytes)):
        return True
    return False


def seg_is_polygon(seg):
    if isinstance(seg, list):
        return all(isinstance(p, (list, tuple)) for p in seg)
    return False


def rle_from_seg(seg, height, width):

    if seg_is_polygon(seg):
        # Polygon format
        rle = maskUtils.frPyObjects(seg, height, width)
        rle = maskUtils.merge(rle)
    elif seg_is_rle(seg):
        if isinstance(seg['counts'], list):
            # uncompressed
            rle = maskUtils.frPyObjects(seg, height, width)
            rle = maskUtils.merge(rle)
        else:
            # compressed
            rle = seg
    else:
        raise ValueError("Invalid segmentation format. "
                         "Expected list of polygons or RLE format.")
    
    # if maskUtils.merge() function return a list, 
    # list[0] is the target rle 
    if isinstance(rle, list):
        rle = rle[0]

    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    
    rle_str = json.dumps(rle)

    return rle_str


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


def padding_image(image, size=(1400, 1400)):
    origin_height = tf.shape(image)[0]
    origin_width = tf.shape(image)[1]
    
    target_height = tf.cast(size[0], tf.int32)
    target_width = tf.cast(size[1], tf.int32)

    pad_height = tf.subtract(target_height, origin_height)
    pad_width = tf.subtract(target_width, origin_width)
    tf.debugging.assert_non_negative(
        [pad_height, pad_width],
        message="Image is larger than target size. Check h and w."
    )

    padded = tf.image.pad_to_bounding_box(
        image, 0, 0, target_height, target_width)
    return padded


def max_size_in_batch(sizes):
    max_height = tf.reduce_max(sizes[:, 0])
    max_width = tf.reduce_max(sizes[:, 1])
    return max_height, max_width


def load_image(file_path, scale, padding_size):
    img_raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized = resize_image(image, scale)
    padded = padding_image(resized, padding_size)
    return padded


def resize_bboxes(bboxes, scale):
    return bboxes * tf.cast(scale, tf.float32)


def resize_segmentations(segmentations, scale):
    return segmentations * tf.cast(scale, tf.float32)


# rles contain a list RLE object of a picture
def create_masks(rles, scale):

    def rle_to_mask(rle, scale):
        rle = json.loads(rle.numpy().decode('utf-8'))
        mask = maskUtils.decode(rle).astype(np.uint8)
        new_height = int(mask.shape[0] * scale)
        new_width = int(mask.shape[1] * scale)

        # cv2 size is (WxH), so we need to swap width and height
        new_size = (new_width, new_height)
        resized = cv2.resize(mask, 
                            new_size, 
                            interpolation=cv2.INTER_NEAREST)
        return resized.astype(np.uint8)

    def rle_to_mask_dummy(rle, scale):
        return tf.constant([[1, 0],[0, 1]], dtype=np.uint8)

    def rle_to_mask_wrap(rle, scale):
        mask = tf.py_function(
            rle_to_mask,
            inp=[rle, scale],
            Tout=tf.uint8
        )
        return tf.RaggedTensor.from_tensor(mask)
    

    masks = tf.map_fn(
        lambda rle: rle_to_mask_wrap(rle, scale),
        rles,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, None), 
            dtype=tf.uint8)
        )
    
    return masks


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
            'rles': tf.constant(
                entry['rles'], 
                dtype=tf.string
            ),
        }


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
    
    padding_size = max_size_in_batch(resizes)
    
    images = tf.map_fn(
        lambda args: load_image(args[0], args[1], padding_size), 
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
    
    masks = tf.map_fn(
        lambda args: create_masks(args[0], args[1]),
        (batch['rles'], scales),
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, None, None), 
            dtype=tf.uint8
        )
    )

    batch['image'] = images
    batch['size'] = resizes
    batch['bboxes'] = bboxes_resized
    batch['masks'] = masks
    # batch['origin_size'] = batch['size']
    
    return batch


def create_dataset(ann_file, img_dir, batch_size=4, shuffle=False):
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    entries = load_image_info(coco, img_ids, img_dir)
    
    ds = tf.data.Dataset.from_generator(
        functools.partial(generator, entries), 
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
                dtype=tf.int32
            ),

            'rles': tf.TensorSpec(
                shape=(None,), 
                dtype=tf.string
            ),
        })
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(entries))
    
    print(f"Dataset size: {len(entries)/batch_size}")

    ds = ds.ragged_batch(batch_size)    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
