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

import tensorflow as tf

def bilinear_interpolate_batch(images, coords, batch_indices):
    """
    images: [N, H, W, C]
    coords: [M, S, 2] (y, x)  # S=采样点数
    batch_indices: [M]，每个box对应的image索引
    返回 [M, S, C]
    """
    N = tf.shape(images)[0]
    H = tf.cast(tf.shape(images)[1], tf.float32)
    W = tf.cast(tf.shape(images)[2], tf.float32)

    y = coords[..., 0]  # [M, S]
    x = coords[..., 1]  # [M, S]

    y0 = tf.floor(y)
    x0 = tf.floor(x)
    y1 = y0 + 1
    x1 = x0 + 1

    y0_clip = tf.clip_by_value(y0, 0, H - 1)
    y1_clip = tf.clip_by_value(y1, 0, H - 1)
    x0_clip = tf.clip_by_value(x0, 0, W - 1)
    x1_clip = tf.clip_by_value(x1, 0, W - 1)

    y0_int = tf.cast(y0_clip, tf.int32)
    y1_int = tf.cast(y1_clip, tf.int32)
    x0_int = tf.cast(x0_clip, tf.int32)
    x1_int = tf.cast(x1_clip, tf.int32)

    # batch_indices: [M] -> [M, S]
    b = tf.expand_dims(batch_indices, axis=-1)  # [M, 1]
    b = tf.tile(b, [1, tf.shape(coords)[1]])    # [M, S]

    def gather(b, y, x):
        # [M, S, 3] for gather_nd
        idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(images, idx)  # [M, S, C]

    Ia = gather(b, y0_int, x0_int)
    Ib = gather(b, y1_int, x0_int)
    Ic = gather(b, y0_int, x1_int)
    Id = gather(b, y1_int, x1_int)

    y0f = tf.cast(y0_clip, tf.float32)
    y1f = tf.cast(y1_clip, tf.float32)
    x0f = tf.cast(x0_clip, tf.float32)
    x1f = tf.cast(x1_clip, tf.float32)

    wa = tf.expand_dims((y1f - y) * (x1f - x), -1)
    wb = tf.expand_dims((y - y0f) * (x1f - x), -1)
    wc = tf.expand_dims((y1f - y) * (x - x0f), -1)
    wd = tf.expand_dims((y - y0f) * (x - x0f), -1)

    vals = wa * Ia + wb * Ib + wc * Ic + wd * Id  # [M, S, C]
    return vals

def image_sample_and_resize(images, boxes, box_indices, output_size, sampling_ratio):
    """
    images: [N, H, W, C] (float32)
    boxes: [M, 4] (x1, y1, x2, y2, float32, 坐标为像素下标)
    box_indices: [M] (每个 box 属于第几张 image)
    output_size: [h_out, w_out]
    sampling_ratio: int，每个输出位置采样采样点的 sqrt 个数
    返回: [M, h_out, w_out, C]
    """
    h_out, w_out = output_size
    M = tf.shape(boxes)[0]
    C = tf.shape(images)[-1]

    x1, y1, x2, y2 = tf.unstack(boxes, axis=1)
    roi_h = tf.maximum(y2 - y1, 1e-6)
    roi_w = tf.maximum(x2 - x1, 1e-6)

    bin_h = roi_h / tf.cast(h_out, tf.float32)
    bin_w = roi_w / tf.cast(w_out, tf.float32)

    grid_y = tf.linspace(0.0, tf.cast(h_out, tf.float32) - 1.0, h_out)  # [h_out]
    grid_x = tf.linspace(0.0, tf.cast(w_out, tf.float32) - 1.0, w_out)  # [w_out]
    grid_y, grid_x = tf.meshgrid(grid_y, grid_x, indexing='ij')  # [h_out, w_out]

    # [M, h_out, w_out]
    grid_y = y1[:, None, None] + (grid_y[None, :, :] + 0.5) * bin_h[:, None, None]
    grid_x = x1[:, None, None] + (grid_x[None, :, :] + 0.5) * bin_w[:, None, None]

    sampling_offsets = tf.linspace(0.0, 1.0, sampling_ratio + 1)[:-1] + 0.5 / sampling_ratio
    sampling_offsets /= tf.cast(sampling_ratio, tf.float32)
    offset_y, offset_x = tf.meshgrid(sampling_offsets, sampling_offsets, indexing='ij')
    offset_y = tf.reshape(offset_y, [-1])  # [S]
    offset_x = tf.reshape(offset_x, [-1])  # [S]
    S = sampling_ratio * sampling_ratio

    # [M, h_out, w_out, S]
    grid_y = tf.expand_dims(grid_y, axis=-1) + offset_y * bin_h[:, None, None, None]
    grid_x = tf.expand_dims(grid_x, axis=-1) + offset_x * bin_w[:, None, None, None]

    # [M, h_out, w_out, S, 2]
    coords = tf.stack([grid_y, grid_x], axis=-1)
    coords = tf.reshape(coords, [M, -1, 2])  # [M, h_out*w_out*S, 2]

    # 双线性插值采样
    sampled_vals = bilinear_interpolate_batch(images, coords, box_indices)  # [M, h_out*w_out*S, C]
    sampled_vals = tf.reshape(sampled_vals, [M, h_out, w_out, S, C])       # [M, h_out, w_out, S, C]
    avg_vals = tf.reduce_mean(sampled_vals, axis=3)                        # [M, h_out, w_out, C]

    return avg_vals