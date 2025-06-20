import tensorflow as tf
from tensorflow.keras import layers

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, 
             feature_maps, 
             strides, 
             base_sizes, 
             ratios, 
             scales, 
             origin_sizes):
        
        ratios = tf.constant(ratios, dtype=tf.float32)
        scales = tf.constant(scales, dtype=tf.float32)

        tf.print('origin_sizes shape:', tf.shape(origin_sizes))
        
        if len(feature_maps) != len(strides) or \
           len(feature_maps) != len(base_sizes):
            raise ValueError("feature_maps, strides, and base_sizes must have the same length.")
        
        anchors_all = []
        for feature_map_batch, stride, base_size in \
            zip(feature_maps, strides, base_sizes):

            anchors_batch = tf.map_fn(
                lambda args: self.generate(args[0], 
                                           args[1],
                                           ratios,
                                           scales,
                                           stride,
                                           base_size
                ),
                elems=(feature_map_batch, origin_sizes),
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=(None, 4), 
                    dtype=tf.float32
                ),
                parallel_iterations=32
            )

            anchors_all.append(anchors_batch)

        for anchors_batch in anchors_all:
            tf.print('anchors shape:', tf.shape(anchors_batch))

        return tf.constant([0.0, 0.0], dtype=tf.float32)
    
    def generate(self, 
                 feature_map, 
                 origin_size, 
                 ratios, 
                 scales, 
                 stride, 
                 base_size):
        
        shape = tf.shape(feature_map)
        height = tf.cast(shape[1], tf.int32)
        width = tf.cast(shape[2], tf.int32)

        base_size = tf.cast(base_size, tf.float32)
        ratios = tf.reshape(ratios, [-1, 1]) # [3, 1]
        scales = tf.reshape(scales, [1, -1]) # [1, 3]

        area = ((base_size * scales) ** 2)
        ws = tf.sqrt(area)/ratios
        hs = ws * ratios

        ws = tf.reshape(ws, [-1])
        hs = tf.reshape(hs, [-1])

        x1 = -ws / 2
        y1 = -hs / 2
        x2 = ws / 2
        y2 = hs / 2
        base_anchors = tf.stack([x1, y1, x2, y2], axis=1)
        tf.print('base_anchors shape:', base_anchors)
        
        dummy = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            ]
        return tf.ragged.constant(dummy, dtype=tf.float32)

class RPNHead(layers.Layer):
    def __init__(self, num_anchors=9, feature_size=256, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.Conv2D(feature_size, 3, padding="same", 
                                  activation="relu")
        
        # 前景/背景分数 (batch, H, W, num_anchors)
        self.cls_logits = layers.Conv2D(num_anchors, 1)

        # 边框回归 (batch, H, W, num_anchors*4)
        self.bbox_deltas = layers.Conv2D(num_anchors * 4, 1)

    
    # def build(self, input_shape):
    #     super().build(input_shape)
    #     self.built = True

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

    # def build(self, input_shape):
    #     super().build(input_shape)
    #     self.built = True
        
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
