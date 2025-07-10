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

        if len(feature_maps) != len(strides) or \
           len(feature_maps) != len(base_sizes):
            raise ValueError("feature_maps, strides, and base_sizes must have the same length.")
        
        anchors = []
        for feature_map, stride, base_size in \
            zip(feature_maps, strides, base_sizes):
            
            anchors_fm = tf.map_fn(
                lambda args: self.anchors_per_feature_map(
                    args[0], args[1], ratios, scales, stride, 
                    base_size,
                ),
                elems=(feature_map, image_sizes),
                fn_output_signature=tf.TensorSpec(
                    shape=(None, 4), 
                    dtype=tf.float32
                ),
                parallel_iterations=32
            )
            # anchors_fm shape is [B, Ni, 4]
            anchors.append(anchors_fm)
        
        # anchors: list of [B, Ni, 4] -> [B, N, 4]
        anchors = tf.concat(anchors, axis=1)
        
        # tf.print('AnchorGenerator',
        #         'anchors :', tf.shape(anchors))
        return anchors
    
    def anchors_per_feature_map(
            self, 
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

        self.conv = layers.Conv2D(
            feature_size, 3, padding="same", activation="relu"
        )
        self.objectness_logits = layers.Conv2D(
            anchors_per_location, 1, activation=None
        )
        self.bbox_deltas = layers.Conv2D(
            anchors_per_location * 4, 1, activation=None
        )
    
    def call(self, feature_maps):
        objectness_logits = []
        bbox_deltas = []

        for feature_map in feature_maps:
            x = self.conv(feature_map)
            objectness_fm = self.objectness_logits(x)
            bbox_deltas_fm = self.bbox_deltas(x)

            # reshape objectness_fm from 
            # [B, H, W, anchors_per_location] to [B, N, 1]
            objectness_fm = tf.reshape(
                objectness_fm, 
                [ tf.shape(objectness_fm)[0], -1, 1]
            )
            
            # reshape bbox_deltas_fm from 
            # [B, H, W, anchors_per_location * 4] to [B, N, 4]
            bbox_deltas_fm = tf.reshape(
                bbox_deltas_fm, 
                [ tf.shape(bbox_deltas_fm)[0], -1, 4]
            )

            objectness_logits.append(objectness_fm)
            bbox_deltas.append(bbox_deltas_fm)

        objectness_logits = tf.concat(objectness_logits, axis=1)
        bbox_deltas = tf.concat(bbox_deltas, axis=1)
        
        # tf.print(
        #     'RPNHead',
        #     'objectness_logits :', tf.shape(objectness_logits),
        #     'bbox_deltas :', tf.shape(bbox_deltas)
        # )
        
        return objectness_logits, bbox_deltas


class ProposalGenerator(layers.Layer):
    def __init__(self, pre_nms_topk=6000, post_nms_topk=1000, 
                 nms_thresh=0.7, min_size=16, **kwargs):
        super().__init__(**kwargs)
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_size = min_size
    
    def call(self, anchors, image_sizes, objectness_logits, bbox_deltas):
        
        # select top-k proposals per image
        def topk_per_image(anchors, image_size, objectness_logits, bbox_deltas):
            
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
            height = tf.cast(image_size[0], tf.float32)
            width  = tf.cast(image_size[1], tf.float32)
            proposals = tf.stack([
                tf.clip_by_value(proposals[:, 0], 0, width - 1),
                tf.clip_by_value(proposals[:, 1], 0, height - 1),
                tf.clip_by_value(proposals[:, 2], 0, width - 1),
                tf.clip_by_value(proposals[:, 3], 0, height - 1)
            ], axis=1)

            # remove small boxes
            ws = proposals[:, 2] - proposals[:, 0]
            hs = proposals[:, 3] - proposals[:, 1]
            valid = tf.where((ws >= self.min_size) & (hs >= self.min_size))

            # update proposals, objectness_logits and bbox_deltas
            proposals = tf.gather(proposals, valid[:, 0])
            objectness_logits = tf.gather(objectness_logits, valid[:, 0])
            bbox_deltas = tf.gather(bbox_deltas, valid[:, 0])

            fg_scores = tf.sigmoid(objectness_logits[:, 0])
            # get tok-k pre nms proposals
            top_k = tf.math.top_k(
                fg_scores, 
                k=tf.minimum(self.pre_nms_topk, tf.shape(fg_scores)[0])
            )
            
            # update proposals, objectness_logits, bbox_deltas and fg_scores
            proposals = tf.gather(proposals, top_k.indices)
            objectness_logits = tf.gather(objectness_logits, top_k.indices)
            bbox_deltas = tf.gather(bbox_deltas, top_k.indices)
            fg_scores = tf.gather(fg_scores, top_k.indices)

            # apply non-maximum suppression (NMS)
            keep = tf.image.non_max_suppression(
                proposals, 
                fg_scores,
                max_output_size=self.post_nms_topk,
                iou_threshold=self.nms_thresh
            )

            # update proposals, objectness_logits, and bbox_deltas
            proposals = tf.gather(proposals, keep)
            objectness_logits = tf.gather(objectness_logits, keep)
            bbox_deltas = tf.gather(bbox_deltas, keep)
            
            # pad proposals, objectness_logits and bbox_deltas to post_nms_topk
            n = tf.shape(proposals)[0]
            pad = self.post_nms_topk - n

            if pad > 0:
                proposals = tf.pad(
                    proposals, 
                    [[0, pad], [0, 0]], 
                    constant_values=0.0
                )
                objectness_logits = tf.pad(
                    objectness_logits, 
                    [[0, pad], [0, 0]], 
                    constant_values=1e-8
                )
                bbox_deltas = tf.pad(
                    bbox_deltas, 
                    [[0, pad], [0, 0]], 
                    constant_values=0.0
                )

            return proposals, objectness_logits, bbox_deltas, \
                tf.range(self.post_nms_topk) < n
        
        results = tf.map_fn(
            lambda args: topk_per_image(args[0], args[1], args[2], args[3]),
            elems=(anchors, image_sizes, objectness_logits, bbox_deltas),
            fn_output_signature=(
                tf.TensorSpec(shape=(self.post_nms_topk, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(self.post_nms_topk, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.post_nms_topk, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(self.post_nms_topk, ), dtype=tf.bool),
            )
        )

        # proposals, objectness_logits, bbox_deltas, valid_mask = results
        # tf.print(
        #     'ProposalGenerator',
        #     'proposals :', tf.shape(proposals),
        #     'objectness_logits :', tf.shape(objectness_logits),
        #     'bbox_deltas :', tf.shape(bbox_deltas),
        #     'valid_mask :', tf.shape(valid_mask)
        # )

        return results
    