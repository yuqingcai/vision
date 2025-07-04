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