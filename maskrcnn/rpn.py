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
             image_sizes):
        
        ratios = tf.constant(ratios, dtype=tf.float32)
        scales = tf.constant(scales, dtype=tf.float32)

        # tf.print('origin_sizes shape:', tf.shape(origin_sizes))
        
        if len(feature_maps) != len(strides) or \
           len(feature_maps) != len(base_sizes):
            raise ValueError("feature_maps, strides, and base_sizes must have the same length.")
        
        anchors = []
        batch_indices = []
        for feature_map, stride, base_size in \
            zip(feature_maps, strides, base_sizes):
            
            anchors_feature_map = tf.map_fn(
                lambda args: self.generate(
                    args[0], args[1], ratios, scales, stride, base_size,
                ),
                elems=(feature_map, image_sizes),
                fn_output_signature=tf.TensorSpec(
                    shape=(None, 4), 
                    dtype=tf.float32
                ),
                parallel_iterations=32
            )
            anchors.append(anchors_feature_map)
            batch_size = tf.shape(anchors_feature_map)[0]
            num_anchors = tf.shape(anchors_feature_map)[1]
            batch_idx = tf.repeat(tf.range(batch_size), num_anchors)
            batch_indices.append(batch_idx)
        
        # flatten anchors: [B, N, 4] -> [B*N, 4]
        # concat batch indices: [B*N]
        anchors_flatten = tf.concat(
            [ tf.reshape(a, [-1, 4]) for a in anchors ], axis=0
        )
        batch_indices = tf.concat(batch_indices, axis=0)
        
        tf.print('anchors_flatten shape:', tf.shape(anchors_flatten))
        tf.print('batch_indices shape:', tf.shape(batch_indices))

        return anchors_flatten, batch_indices
    
    def generate(self, 
                 feature_map, 
                 image_size, 
                 ratios, 
                 scales, 
                 stride, 
                 base_size):
        
        # generate base anchors
        base_size = tf.cast(base_size, tf.float32)
        ratios = tf.reshape(ratios, [-1, 1]) # [3, 1]
        scales = tf.reshape(scales, [1, -1]) # [1, 3]

        area = ((base_size * scales) ** 2)
        w_s = tf.sqrt(area)/ratios
        h_s = w_s * ratios
        w_s = tf.reshape(w_s, [-1])
        h_s = tf.reshape(h_s, [-1])
        x_1 = -w_s / 2
        y_1 = -h_s / 2
        x_2 = w_s / 2
        y_2 = h_s / 2
        # base_anchors shape is [A, 4]
        base_anchors = tf.stack([x_1, y_1, x_2, y_2], axis=1)

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
        height = tf.cast(image_size[0], tf.float32)
        width = tf.cast(image_size[1], tf.float32)
        anchors = tf.stack([
            tf.clip_by_value(anchors[:, 0], 0, width - 1),
            tf.clip_by_value(anchors[:, 1], 0, height - 1),
            tf.clip_by_value(anchors[:, 2], 0, width - 1),
            tf.clip_by_value(anchors[:, 3], 0, height - 1)
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

        for feature_map in feature_maps:
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
        
    def call(self, image_sizes, anchors, class_logits, bbox_deltas):
        proposals = tf.map_fn(
            lambda args: self.generate(
                args[0], args[1], args[2], args[3],
            ),
            elems=(image_sizes, anchors, class_logits, bbox_deltas),
            fn_output_signature=tf.RaggedTensorSpec(
                shape=(None, 4), 
                dtype=tf.float32
                ),
            parallel_iterations=32
        )
        return proposals

    def generate(self, image_size, anchors, class_logits, bbox_deltas):
        # decode bbox
        # a_x, a_y are the center coordinates of the anchors
        # a_w, a_h are the width and height of the anchors
        a_w = anchors[:, 2] - anchors[:, 0]
        a_h = anchors[:, 3] - anchors[:, 1]
        a_x = anchors[:, 0] + 0.5 * a_w
        a_y = anchors[:, 1] + 0.5 * a_h

        # t_x, t_y are the offsets to the center coordinates
        # t_w, t_h are the log-scaled width and height
        t_x = bbox_deltas[:, 0]
        t_y = bbox_deltas[:, 1]
        t_w = bbox_deltas[:, 2]
        t_h = bbox_deltas[:, 3]

        # p_x, p_y are the predicted center coordinates
        # p_w, p_h are the predicted width and height
        # tf.clip_by_value is used to limit the range of t_w and t_h
        # to prevent too large or too small boxes
        p_x = a_x + t_x * a_w
        p_y = a_y + t_y * a_h
        p_w = a_w * tf.exp(tf.clip_by_value(t_w, -10.0, 10.0))
        p_h = a_h * tf.exp(tf.clip_by_value(t_h, -10.0, 10.0))

        # x_1, y_1, x_2, y_2 are the coordinates of the proposals
        x_1 = p_x - 0.5 * p_w
        y_1 = p_y - 0.5 * p_h
        x_2 = p_x + 0.5 * p_w
        y_2 = p_y + 0.5 * p_h
        proposals = tf.stack([x_1, y_1, x_2, y_2], axis=1)

        # clip proposals to the image size
        heights = tf.cast(image_size[0], tf.float32)
        widths  = tf.cast(image_size[1], tf.float32)
        proposals = tf.stack([
            tf.clip_by_value(proposals[:, 0], 0, widths - 1),
            tf.clip_by_value(proposals[:, 1], 0, heights - 1),
            tf.clip_by_value(proposals[:, 2], 0, widths - 1),
            tf.clip_by_value(proposals[:, 3], 0, heights - 1)
        ], axis=1)

        # remove small boxes
        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        valid = tf.where((ws >= self.min_size) & (hs >= self.min_size))
        proposals = tf.gather(proposals, valid[:, 0])
        fg_scores = tf.gather(tf.sigmoid(class_logits[:, 0]), valid[:, 0])
        
        # get tok-k pre nms proposals
        top_k = tf.math.top_k(
            fg_scores, 
            k=tf.minimum(self.pre_nms_topk, tf.shape(fg_scores)[0])
        )
        proposals = tf.gather(proposals, top_k.indices)
        fg_scores = tf.gather(fg_scores, top_k.indices)

        # apply non-maximum suppression (NMS)
        keep = tf.image.non_max_suppression(
            proposals, 
            fg_scores,
            max_output_size=self.post_nms_topk,
            iou_threshold=self.nms_thresh
        )
        proposals = tf.gather(proposals, keep)
        
        return tf.RaggedTensor.from_tensor(proposals)

