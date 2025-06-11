import tensorflow as tf

def sample_and_assign_targets(
    proposals,
    gt_boxes,
    gt_classes,
    gt_masks,
    mask_size=28,
    positive_iou_thresh=0.5,
    negative_iou_thresh=0.5,
    num_samples=128,
    positive_fraction=0.25,
):
    """
    proposals: (N, 4)
    gt_boxes: (M, 4)
    gt_classes: (M,)
    gt_masks: (M, H, W)
    mask_size: int, mask target size
    positive_iou_thresh: float, IoU threshold for positive samples
    negative_iou_thresh: float, IoU threshold for negative samples
    num_samples: int, total number of sampled RoIs
    positive_fraction: float, fraction of positives in sampled RoIs

    Returns:
        roi_boxes: (num_samples, 4)
        roi_labels: (num_samples,)
        roi_bbox_targets: (num_samples, 4)
        roi_mask_targets: (num_samples, mask_size, mask_size)
    """
    # 1. 计算 proposals 和 gt_boxes 的 IoU
    ious = compute_iou(proposals, gt_boxes)  # (N, M)
    max_iou = tf.reduce_max(ious, axis=1)
    max_idx = tf.argmax(ious, axis=1)

    # 2. 正负样本分配
    positive_mask = max_iou >= positive_iou_thresh
    negative_mask = (max_iou < negative_iou_thresh)

    positive_indices = tf.where(positive_mask)[:, 0]
    negative_indices = tf.where(negative_mask)[:, 0]

    # 3. 正负样本采样
    num_pos = int(num_samples * positive_fraction)
    num_pos = min(num_pos, tf.size(positive_indices))
    num_neg = num_samples - num_pos
    num_neg = min(num_neg, tf.size(negative_indices))

    perm_pos = tf.random.shuffle(positive_indices)[:num_pos]
    perm_neg = tf.random.shuffle(negative_indices)[:num_neg]

    keep_indices = tf.concat([perm_pos, perm_neg], axis=0)
    roi_boxes = tf.gather(proposals, keep_indices)
    matched_gt_idx = tf.gather(max_idx, keep_indices)

    # 4. 分配标签
    roi_labels = tf.gather(gt_classes, matched_gt_idx)
    # 负样本标签设为0（背景）
    roi_labels = tf.where(
        tf.range(tf.shape(keep_indices)[0]) < num_pos,
        roi_labels,
        tf.zeros_like(roi_labels)
    )

    # 5. 边框回归目标
    matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
    roi_bbox_targets = encode_boxes(roi_boxes, matched_gt_boxes)

    # 6. mask target（只对正样本）
    matched_gt_masks = tf.gather(gt_masks, matched_gt_idx)  # (num_samples, H, W)
    roi_mask_targets = []
    for i in range(tf.shape(roi_boxes)[0]):
        if i < num_pos:
            # proposal和gt_box的坐标
            y1, x1, y2, x2 = tf.unstack(tf.cast(roi_boxes[i], tf.int32))
            mask = matched_gt_masks[i][y1:y2, x1:x2]
            mask = tf.image.resize(mask[..., tf.newaxis], (mask_size, mask_size), method='bilinear')
            mask = tf.squeeze(mask, axis=-1)
            # 二值化
            mask = tf.cast(mask >= 0.5, tf.float32)
            roi_mask_targets.append(mask)
        else:
            roi_mask_targets.append(tf.zeros((mask_size, mask_size), dtype=tf.float32))
    roi_mask_targets = tf.stack(roi_mask_targets, axis=0)

    return roi_boxes, roi_labels, roi_bbox_targets, roi_mask_targets

def compute_iou(boxes1, boxes2):
    """
    boxes1: (N, 4), boxes2: (M, 4)
    返回 (N, M) 的IoU矩阵
    """
    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)
    boxes1 = tf.expand_dims(boxes1, 1)  # (N, 1, 4)
    boxes2 = tf.expand_dims(boxes2, 0)  # (1, M, 4)
    y1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    x1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    y2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    x2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    intersection = tf.maximum(y2 - y1, 0) * tf.maximum(x2 - x1, 0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-8)

def encode_boxes(proposals, gt_boxes):
    """
    proposals, gt_boxes: (N, 4)
    返回 (N, 4) 的回归目标
    """
    proposals = tf.cast(proposals, tf.float32)
    gt_boxes = tf.cast(gt_boxes, tf.float32)
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    dx = (gx - px) / (pw + 1e-8)
    dy = (gy - py) / (ph + 1e-8)
    dw = tf.math.log(gw / (pw + 1e-8))
    dh = tf.math.log(gh / (ph + 1e-8))

    return tf.stack([dx, dy, dw, dh], axis=1)