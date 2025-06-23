import tensorflow as tf
from tensorflow.keras import layers


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

        # tf.print('origin_sizes shape:', tf.shape(origin_sizes))
        
        if len(feature_maps) != len(strides) or \
           len(feature_maps) != len(base_sizes):
            raise ValueError("feature_maps, strides, and base_sizes must have the same length.")
        
        anchors = []
        for feature_map, stride, base_size in \
            zip(feature_maps, strides, base_sizes):
            
            anchors_feature_map = tf.map_fn(
                lambda args: self.generate(args[0], 
                                           args[1],
                                           ratios,
                                           scales,
                                           stride,
                                           base_size,
                ),
                elems=(feature_map, origin_sizes),
                fn_output_signature=tf.TensorSpec(
                    shape=(None, 4), 
                    dtype=tf.float32
                ),
                parallel_iterations=32
            )
            
            anchors.append(anchors_feature_map)
        
        anchors = tf.concat(anchors, axis=1)

        return anchors
    
    def generate(self, 
                 feature_map, 
                 origin_size, 
                 ratios, 
                 scales, 
                 stride, 
                 base_size
                 ):
        
        # generate base anchors
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
        # base_anchors shape is [A, 4]
        base_anchors = tf.stack([x1, y1, x2, y2], axis=1)

        # shift base anchors to all locations on the feature map
        # shifts shape is [K, 4]
        height = tf.cast(tf.shape(feature_map)[0], tf.int32)
        width = tf.cast(tf.shape(feature_map)[1], tf.int32)
        shift_x = (tf.range(width, dtype=tf.float32) + 0.5) * stride
        shift_y = (tf.range(height, dtype=tf.float32) + 0.5) * stride
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        shifts = tf.stack([
            tf.reshape(shift_x, [-1]),
            tf.reshape(shift_y, [-1]),
            tf.reshape(shift_x, [-1]),
            tf.reshape(shift_y, [-1])
        ], axis=1)
        # broadcast add, anchors shape is [K, A, 4]
        A = tf.shape(base_anchors)[0]
        K = tf.shape(shifts)[0]
        anchors = tf.reshape(base_anchors, [1, A, 4]) + \
            tf.reshape(shifts, [K, 1, 4])
        
        # reshape anchors shape to [K*A, 4]
        anchors = tf.reshape(anchors, [K * A, 4])

        # clip anchors to the image size
        origin_height = tf.cast(origin_size[0], tf.float32)
        origin_width = tf.cast(origin_size[1], tf.float32)
        anchors = tf.stack([
            tf.clip_by_value(anchors[:, 0], 0, origin_width - 1),
            tf.clip_by_value(anchors[:, 1], 0, origin_height - 1),
            tf.clip_by_value(anchors[:, 2], 0, origin_width - 1),
            tf.clip_by_value(anchors[:, 3], 0, origin_height - 1)
        ], axis=1)
        
        return anchors


class RPNHead(layers.Layer):
    def __init__(self, anchors_per_location, feature_size, **kwargs):
        super().__init__(**kwargs)

        tf.print('anchors_per_location:', anchors_per_location)

        self.conv = layers.Conv2D(feature_size, 
                                  3, 
                                  padding="same", 
                                  activation="relu")
        self.class_logits = layers.Conv2D(anchors_per_location, 1)
        self.bbox_deltas = layers.Conv2D(anchors_per_location * 4, 1)
    

    def call(self, feature_maps):
        class_logits_all = []
        bbox_deltas_all = []

        for i, feature_map in enumerate(feature_maps):
            x = self.conv(feature_map)
            class_logits = self.class_logits(x)
            bbox_deltas = self.bbox_deltas(x)

            class_logits = tf.reshape(
                class_logits, 
                [tf.shape(class_logits)[0], -1, 1])
            
            bbox_deltas = tf.reshape(
                bbox_deltas, 
                [tf.shape(bbox_deltas)[0], -1, 4])
            
            class_logits_all.append(class_logits)
            bbox_deltas_all.append(bbox_deltas)

        class_logits_all = tf.concat(class_logits_all, axis=1)
        bbox_deltas_all = tf.concat(bbox_deltas_all, axis=1)

        return class_logits_all, bbox_deltas_all


class ProposalGenerator(layers.Layer):
    def __init__(self, pre_nms_topk=6000, post_nms_topk=1000, 
                 nms_thresh=0.7, min_size=16, **kwargs):
        super().__init__(**kwargs)
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        
    def call(self, images, origin_sizes, anchors, bbox_deltas, logits):

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
        shape = tf.shape(images)
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
