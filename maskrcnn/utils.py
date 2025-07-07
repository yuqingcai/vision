import tensorflow as tf

def compute_iou(boxes1, boxes2):
    """
    boxes1: [P, 4], boxes2: [G, 4]
    """
    boxes1 = tf.expand_dims(boxes1, 1)  # [P,1,4]
    boxes2 = tf.expand_dims(boxes2, 0)  # [1,G,4]

    inter_x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * \
        tf.maximum(inter_y2 - inter_y1, 0)
    
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])
    
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / tf.maximum(union_area, 1e-8)

    # shape [P, G]
    return iou


def bilinear_interpolate(image, coords):
    """
    image: [H, W, C], float32
    coords: [N, 2], float32 (y, x)
    returns: [N, C]
    """
    H = tf.cast(tf.shape(image)[0], tf.float32)
    W = tf.cast(tf.shape(image)[1], tf.float32)
    N = tf.shape(coords)[0]

    # tf.image.crop_and_resize 需要 boxes: [N, 4] (y1, x1, y2, x2), 范围[0,1]
    # 这里每个 box 是一个点，y1==y2, x1==x2
    y = coords[:, 0] / (H - 1)
    x = coords[:, 1] / (W - 1)
    boxes = tf.stack([y, x, y, x], axis=1)      # [N, 4]
    box_indices = tf.zeros([N], dtype=tf.int32) # 假设 image 没有 batch 维

    image = tf.expand_dims(image, axis=0)  # [1, H, W, C]
    vals = tf.image.crop_and_resize(
        image, boxes, box_indices, crop_size=[1, 1], method='bilinear'
    )  # [N, 1, 1, C]
    vals = tf.squeeze(vals, axis=[1, 2])  # [N, C]
    return vals


def image_sample_and_resize(image, box, output_size, sampling_ratio):
    """
    image: [H, W, C], float32
    box: [4] - x1, y1, x2, y2
    Returns: [ output_size[0], output_size[1], C]
    """
    x1, y1, x2, y2 = tf.unstack(box)
    h_out, w_out = output_size

    roi_h = tf.maximum(y2 - y1, 1e-6)
    roi_w = tf.maximum(x2 - x1, 1e-6)

    bin_h = roi_h / tf.cast(h_out, tf.float32)
    bin_w = roi_w / tf.cast(w_out, tf.float32)

    # bin center
    grid_y = tf.linspace(0.0, tf.cast(h_out, tf.float32) - 1.0, h_out)
    grid_x = tf.linspace(0.0, tf.cast(w_out, tf.float32) - 1.0, w_out)
    grid_y, grid_x = tf.meshgrid(grid_y, grid_x, indexing='ij')
    grid_y = y1 + (grid_y + 0.5) * bin_h
    grid_x = x1 + (grid_x + 0.5) * bin_w

    # sample offsets
    sampling_offsets = tf.linspace(0.0, 1.0, sampling_ratio + 1)[:-1] + \
        0.5 / sampling_ratio
    sampling_offsets /= tf.cast(sampling_ratio, tf.float32)
    offset_y, offset_x = tf.meshgrid(
        sampling_offsets, sampling_offsets, indexing='ij'
    )
    offset_y = tf.reshape(offset_y, [-1])
    offset_x = tf.reshape(offset_x, [-1])

    # sample coordinates
    grid_y = tf.expand_dims(grid_y, axis=-1) + offset_y * bin_h  # [H, W, S*S]
    grid_x = tf.expand_dims(grid_x, axis=-1) + offset_x * bin_w  # [H, W, S*S]

    coords = tf.stack([grid_y, grid_x], axis=-1)  # [H, W, S*S, 2]
    coords = tf.reshape(coords, [-1, 2])  # [H*W*S*S, 2]

    sampled_vals = bilinear_interpolate(image, coords)  # [H*W*S*S, C]
    sampled_vals = tf.reshape(
        sampled_vals, 
        [h_out, w_out, sampling_ratio * sampling_ratio, -1]
    )  # [H, W, S*S, C]
    avg_vals = tf.reduce_mean(sampled_vals, axis=2)  # [H, W, C]
    return avg_vals
