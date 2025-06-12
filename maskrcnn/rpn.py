import tensorflow as tf
from tensorflow.keras import layers, Model

"""
生成 anchors
anchors（锚框）：是按照特定规则（位置、尺度、比例）在特征图上密集生成的一组“参考框”，
它们和图片内容无关，只和特征图的空间位置、ratios、scales 有关。anchors 数量通常非
常多（几十万）。
proposals（候选框）：是通过 RPN Head 对 anchors 进行前景/背景分类和边框回归后，
经过解码、筛选（如 NMS、top-N）得到的一组“高质量候选区域”。proposals 数量远小于
anchors（通常几百到几千），用于后续 ROI Head 的
精细分类和回归。
"""
class AnchorGenerator(layers.Layer):
    def __init__(self, ratios=[0.5, 1, 2], scales=[8, 16, 32], **kwargs):
        super().__init__(**kwargs)
        self.ratios = tf.constant(ratios, dtype=tf.float32)
        self.scales = tf.constant(scales, dtype=tf.float32)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, strides, base_sizes, feature_maps):
        anchors_all = []
        for feature_map, stride, base_size in \
            zip(feature_maps, strides, base_sizes):

            shape = tf.shape(feature_map)
            H = tf.cast(shape[1], tf.int32)
            W = tf.cast(shape[2], tf.int32)

            # 生成所有 ratio/scale 组合
            base_size = tf.cast(base_size, tf.float32) 
            ratios = tf.reshape(self.ratios, [-1, 1])
            scales = tf.reshape(self.scales, [1, -1])
            ws = tf.sqrt(base_size * base_size * scales * scales / ratios)
            hs = ws * ratios
            ws = tf.reshape(ws, [-1])
            hs = tf.reshape(hs, [-1])

            # [A, 4]
            x1 = -ws / 2
            y1 = -hs / 2
            x2 = ws / 2
            y2 = hs / 2
            base_anchors = tf.stack([x1, y1, x2, y2], axis=1)

            # 平移到所有位置
            # anchor 的数量只与特征图的空间尺寸（高 H，宽 W）和每个位置的 
            # base_anchors 数量有关，与特征图的深度（通道数）无关。
            # 每层 anchor 数量的计算公式为：
            # base_anchor_num_per_dot = len(ratios) * len(scales)
            # anchor_num_per_layer = H × W × base_anchor_num_per_dot
            shift_x = (tf.range(W, dtype=tf.float32) + 0.5) * stride
            shift_y = (tf.range(H, dtype=tf.float32) + 0.5) * stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
            shifts = tf.stack([
                tf.reshape(shift_x, [-1]),
                tf.reshape(shift_y, [-1]),
                tf.reshape(shift_x, [-1]),
                tf.reshape(shift_y, [-1])
            ], axis=1)  # [K, 4]
            A = tf.shape(base_anchors)[0]
            K = tf.shape(shifts)[0]
            anchors = tf.reshape(base_anchors, [1, A, 4]) + \
                tf.reshape(shifts, [K, 1, 4])
            anchors = tf.reshape(anchors, [K * A, 4])
            anchors_all.append(anchors)
        # 合并所有层的 anchors

        return tf.concat(anchors_all, axis=0)
    

class RPNHead(layers.Layer):
    def __init__(self, num_anchors=9, feature_size=256, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.Conv2D(feature_size, 3, padding="same", 
                                  activation="relu")
        
        # 前景/背景分数 (batch, H, W, num_anchors)
        self.cls_logits = layers.Conv2D(num_anchors, 1)

        # 边框回归 (batch, H, W, num_anchors*4)
        self.bbox_deltas = layers.Conv2D(num_anchors * 4, 1)

    
    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    # 输入特征图列表，输出 logits 和 bbox_deltas
    # shape
    # logits: (batch, HW*A, 1)
    # bbox_deltas: (batch, HW*A, 4)
    # 
    def call(self, feature_maps):
        logits_all = []
        bbox_deltas_all = []

        for feature_map in feature_maps:
            x = self.conv(feature_map)
            logits = self.cls_logits(x)
            bbox_deltas = self.bbox_deltas(x)

            logits = tf.reshape(logits, [tf.shape(logits)[0], -1, 1])
            bbox_deltas = tf.reshape(
                bbox_deltas, [tf.shape(bbox_deltas)[0], -1, 4])
            
            logits_all.append(logits)
            bbox_deltas_all.append(bbox_deltas)

        logits_all = tf.concat(logits_all, axis=1)
        bbox_deltas_all = tf.concat(bbox_deltas_all, axis=1)
        
        # 去掉 batch 维度，现在这个模型只能处理单张图片。
        # shape
        # logits_all: (batch, N, 1)
        # bbox_deltas_all: (batch, N, 4)
        logits_all = tf.squeeze(logits_all, axis=0)
        bbox_deltas_all = tf.squeeze(bbox_deltas_all, axis=0)
        return logits_all, bbox_deltas_all


class ProposalGenerator(layers.Layer):
    def __init__(self, pre_nms_topk=6000, post_nms_topk=1000, 
                 nms_thresh=0.7, min_size=16, **kwargs):
        super().__init__(**kwargs)
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_size = min_size

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True
        
    def call(self, image, anchors, bbox_deltas, logits):

        # 1. 解码 proposals
        wa = anchors[:, 2] - anchors[:, 0]
        ha = anchors[:, 3] - anchors[:, 1]
        xa = anchors[:, 0] + 0.5 * wa
        ya = anchors[:, 1] + 0.5 * ha

        dx = bbox_deltas[:, 0]
        dy = bbox_deltas[:, 1]
        dw = bbox_deltas[:, 2]
        dh = bbox_deltas[:, 3]

        x = dx * wa + xa
        y = dy * ha + ya
        w = tf.exp(dw) * wa
        h = tf.exp(dh) * ha

        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        x2 = x + 0.5 * w
        y2 = y + 0.5 * h
        proposals = tf.stack([x1, y1, x2, y2], axis=1)

        # 2. 裁剪到图片边界
        shape = tf.shape(image)
        height = tf.cast(shape[1], tf.float32)
        width = tf.cast(shape[2], tf.float32)
        
        proposals = tf.stack([
            tf.clip_by_value(proposals[:, 0], 0, width - 1),
            tf.clip_by_value(proposals[:, 1], 0, height - 1),
            tf.clip_by_value(proposals[:, 2], 0, width - 1),
            tf.clip_by_value(proposals[:, 3], 0, height - 1)
        ], axis=1)

        # 3. 去除过小的框
        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        valid = tf.where((ws >= self.min_size) & (hs >= self.min_size))
        proposals = tf.gather(proposals, valid[:, 0])
        scores = tf.gather(tf.sigmoid(logits[:, 0]), valid[:, 0])

        # 4. 取 top-k
        topk = tf.math.top_k(
            scores, 
            k=tf.minimum(self.pre_nms_topk, tf.shape(scores)[0])
            )
        proposals = tf.gather(proposals, topk.indices)
        scores = tf.gather(scores, topk.indices)

        # 5. NMS
        keep = tf.image.non_max_suppression(
            proposals, 
            scores,
            max_output_size=self.post_nms_topk,
            iou_threshold=self.nms_thresh
        )
        proposals = tf.gather(proposals, keep)

        return proposals
