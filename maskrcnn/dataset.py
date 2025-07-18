import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from functools import partial
import os
import functools
import numpy as np
import json
import cv2
import time


def load_image_info(coco, img_ids, img_dir, id_to_index):
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
        labels = []
        rles = []
        for annotation in annotations:
            if annotation.get('iscrowd', 0) == 1:
                # Skip images with crowd annotations
                # print(f"Warning: Skipping crowd annotation in {file_path}.")
                continue
            
            if 'bbox' in annotation and 'category_id' in annotation \
                and 'segmentation' in annotation:
                
                # category_id handling
                class_index = id_to_index[annotation['category_id']]
                labels.append(class_index)
                
                # bboxes handling
                # a bbox: [x1, y1, x2, y2]
                x, y, w, h = annotation['bbox']
                bboxes.append([x, y, x+w, y+h])
                
                # segmentation handling
                rle = rle_from_seg(
                    annotation['segmentation'], 
                    height, 
                    width
                )
                rles.append(rle)
        
        # If no bounding boxes or category_ids, skip this image
        if not bboxes or not labels or not rles:
            print(f"Warning: Annotation details not found for {file_path}.")
            continue
        
        if len(bboxes) != len(labels) or len(bboxes) != len(rles):
            print(f"Warning: Annotation details mismatch in {file_path}.")
            continue
        
        entries.append({
            'file_path' : file_path, 
            'size' :  (height, width), 
            'bbox' : bboxes,
            'label' : labels,
            'rle' : rles, 
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
    scale = tf.minimum(
        min_size / tf.minimum(h, w),
        max_size / tf.maximum(h, w))
    return scale


def resize_to(size, scale):
    h = tf.cast(size[0], tf.float32)
    w = tf.cast(size[1], tf.float32)
    new_h = tf.cast(tf.round(h * tf.cast(scale, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(w * tf.cast(scale, tf.float32)), tf.int32)
    return tf.stack([new_h, new_w])


def resize_image(image, scale, output_dtype=None):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    new_hw = resize_to(tf.stack([h, w]), scale)
    resized = tf.image.resize(image, new_hw, method='bilinear')
    if output_dtype is not None:
        resized = tf.cast(resized, output_dtype)
    return resized


def padding_image(image, size=(1400, 1400), output_dtype=None):
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
    
    if output_dtype is not None:
        padded = tf.cast(padded, output_dtype)

    return padded


def max_size_in_batch(sizes):
    max_height = tf.reduce_max(sizes[:, 0])
    max_width = tf.reduce_max(sizes[:, 1])
    return max_height, max_width


def load_image(file_path, scale, padding_size):
    img_raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    # mixed_precision, using tf.float16
    image = tf.image.convert_image_dtype(image, tf.float16)
    resized = resize_image(image, scale, tf.float16)
    padded = padding_image(resized, padding_size, tf.float16)
    
    return padded


def resize_bboxes(bboxes, scale):
    return bboxes * tf.cast(scale, tf.float32)


def resize_segmentations(segmentations, scale):
    return segmentations * tf.cast(scale, tf.float32)


def create_masks(rles, scale):
    """rles contain a list RLE object of a picture,
    the data return is a list of masks,
    each mask is a 2D array with shape (H, W),
    where H and W are the height and width of the resized image.
    The mask is resized to the same scale as the image.
    The mask is a binary mask, where 1 means the object is present,
    and 0 means the object is not present.
    """
    def rle_to_mask(rle, scale):
        rle = json.loads(rle.numpy().decode('utf-8'))
        mask = maskUtils.decode(rle).astype(np.uint8)

        new_height = int(round(mask.shape[0] * float(scale.numpy())))
        new_width = int(round(mask.shape[1] * float(scale.numpy())))
        
        # cv2 size is (Width x Height)
        new_size = (new_width, new_height)

        resized = cv2.resize(
            mask, 
            new_size, 
            interpolation=cv2.INTER_NEAREST
        )
        return resized.astype(np.uint8)

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
            'bbox': tf.ragged.constant(
                entry['bbox'], 
                dtype=tf.float32
            ),
            'label': tf.constant(
                entry['label'], 
                dtype=tf.int32
            ),
            'rle': tf.constant(
                entry['rle'], 
                dtype=tf.string
            ),
        }


def preprocess(batch, min_size, max_size):
    
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
    tf.print('padding_size', padding_size)
    
    # mixed_precision, using tf.float16
    images = tf.map_fn(
        lambda args: load_image(args[0], args[1], padding_size), 
        (batch['file_path'], scales),
        fn_output_signature=tf.TensorSpec(
            shape=(None, None, 3), 
            dtype=tf.float16
        )
    )

    bboxes_resized = tf.map_fn(
        lambda args: resize_bboxes(args[0], args[1]), 
        (batch['bbox'], scales),
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, 4), 
            dtype=tf.float32
        )
    )

    masks = tf.map_fn(
        lambda args: create_masks(args[0], args[1]),
        (batch['rle'], scales),
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, None, None), 
            dtype=tf.uint8
        )
    )
    
    batch['image'] = images
    batch['size'] = resizes
    batch['bbox'] = bboxes_resized
    batch['mask'] = masks

    return batch


def coco_category_id_index(coco):
    categories = coco.dataset['categories']
    cat_ids = [cat['id'] for cat in categories]
    id_to_index = {
        cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)
    }

    # background class
    id_to_index[0] = 0
    
    return id_to_index

def create_dataset(ann_file, img_dir, batch_size=4, shuffle=False, 
                   min_size=800, max_size=1333):
    coco = COCO(ann_file)

    id_index = coco_category_id_index(coco)
    
    img_ids = coco.getImgIds()
    entries = load_image_info(coco, img_ids, img_dir, id_index)
    
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

            'bbox': tf.RaggedTensorSpec(
                shape=(None, 4), 
                dtype=tf.float32
            ),

            'label': tf.TensorSpec(
                shape=(None,), 
                dtype=tf.int32
            ),

            'rle': tf.TensorSpec(
                shape=(None,), 
                dtype=tf.string
            ),
        })
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(entries))
    
    print(f"Dataset size: {len(entries)/batch_size}")

    ds = ds.ragged_batch(batch_size)
    
    ds = ds.map(
        partial(preprocess, min_size=min_size, max_size=max_size), 
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
