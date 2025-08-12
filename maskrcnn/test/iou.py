import tensorflow as tf
from utils import compute_iou

# 两个 box 完全重合
boxes1 = tf.constant([[0, 0, 2, 2]], dtype=tf.float32)  # [P=1, 4]
boxes2 = tf.constant([[0, 0, 2, 2]], dtype=tf.float32)  # [G=1, 4]
iou = compute_iou(boxes1, boxes2)
print("IOU (完全重合):", iou.numpy())  # 1.0

# 两个 box 部分重叠
boxes1 = tf.constant([[0, 0, 2, 2]], dtype=tf.float32)
boxes2 = tf.constant([[1, 1, 3, 3]], dtype=tf.float32)
iou = compute_iou(boxes1, boxes2)
print("IOU (部分重叠):", iou.numpy())  # 0.14285715

# 两个 box 不重叠
boxes1 = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
boxes2 = tf.constant([[2, 2, 3, 3]], dtype=tf.float32)
iou = compute_iou(boxes1, boxes2)
print("IOU (不重叠):", iou.numpy())  # 0.0

# 多个 box 测试
boxes1 = tf.constant([[0, 0, 2, 2], [1, 1, 3, 3], [4, 4, 6, 6]], dtype=tf.float32)
boxes2 = tf.constant([[0, 0, 2, 2], [2, 2, 3, 3], [4, 4, 6, 6]], dtype=tf.float32)
iou = compute_iou(boxes1, boxes2)
print("IOU (多个box):\n", iou.numpy())