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

        # tf.print('anchors_flatten shape:', tf.shape(anchors_flatten))
        # tf.print('batch_indices shape:', tf.shape(batch_indices))

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
            objectness_feature_map = self.objectness_logits(x)
            bbox_deltas_feature_map = self.bbox_deltas(x)

            # reshape objectness_feature_map from 
            # [B, H, W, anchors_per_location] to [B, N, 1]
            objectness_feature_map = tf.reshape(
                objectness_feature_map, 
                [ tf.shape(objectness_feature_map)[0], -1, 1]
            )
            
            # reshape bbox_deltas_feature_map from 
            # [B, H, W, anchors_per_location * 4] to [B, N, 4]
            bbox_deltas_feature_map = tf.reshape(
                bbox_deltas_feature_map, 
                [ tf.shape(bbox_deltas_feature_map)[0], -1, 4]
            )

            objectness_logits.append(objectness_feature_map)
            bbox_deltas.append(bbox_deltas_feature_map)

        # flatten objectness_logits and bbox_deltas
        # objectness_logits shape [B, N, 1] -> [B*N, 1]
        # bbox_deltas shape [B, N, 4] -> [B*N, 4]
        objectness_logits = tf.concat(
            [ tf.reshape(a, [-1, 1]) for a in objectness_logits ], axis=0
        )
        bbox_deltas = tf.concat(
            [ tf.reshape(a, [-1, 4]) for a in bbox_deltas ], axis=0
        )
        
        return objectness_logits, bbox_deltas


class ProposalGenerator(layers.Layer):
    def __init__(self, pre_nms_topk=6000, post_nms_topk=1000, 
                 nms_thresh=0.7, min_size=16, **kwargs):
        super().__init__(**kwargs)
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_size = min_size
    
    def call(self, anchors, batch_indices, image_sizes,   
             objectness_logits, bbox_deltas):
            
        image_sizes = tf.gather(image_sizes, batch_indices)
        batch_size = tf.reduce_max(batch_indices) + 1

        # select top-k proposals per image
        def topk_proposals_per_image(index):
            mask = tf.equal(batch_indices, index)
            selected = tf.where(mask)[:, 0]
            anchors_selected = tf.gather(anchors, selected)
            objectness_logits_selected = tf.gather(objectness_logits, selected)
            bbox_deltas_selected = tf.gather(bbox_deltas, selected)
            image_sizes_selected = tf.gather(image_sizes, selected)

            # decode bbox
            # a_x, a_y are the center coordinates of the anchors
            # a_w, a_h are the width and height of the anchors
            a_w = anchors_selected[:, 2] - anchors_selected[:, 0]
            a_h = anchors_selected[:, 3] - anchors_selected[:, 1]
            a_x = anchors_selected[:, 0] + 0.5 * a_w
            a_y = anchors_selected[:, 1] + 0.5 * a_h

            # t_x, t_y are the offsets to the center coordinates
            # t_w, t_h are the log-scaled width and height
            t_x = bbox_deltas_selected[:, 0]
            t_y = bbox_deltas_selected[:, 1]
            t_w = bbox_deltas_selected[:, 2]
            t_h = bbox_deltas_selected[:, 3]

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
            heights = tf.cast(image_sizes_selected[:, 0], tf.float32)
            widths  = tf.cast(image_sizes_selected[:, 1], tf.float32)
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

            # update proposals, objectness_logits and bbox_deltas
            proposals = tf.gather(proposals, valid[:, 0])
            objectness_logits_selected = tf.gather(
                objectness_logits_selected, valid[:, 0])
            bbox_deltas_selected = tf.gather(bbox_deltas_selected, valid[:, 0])

            fg_scores = tf.sigmoid(objectness_logits_selected[:, 0])
            # get tok-k pre nms proposals
            top_k = tf.math.top_k(
                fg_scores, 
                k=tf.minimum(self.pre_nms_topk, tf.shape(fg_scores)[0])
            )
            
            # update proposals, objectness_logits, bbox_deltas and fg_scores
            proposals = tf.gather(proposals, top_k.indices)
            objectness_logits_selected = tf.gather(
                objectness_logits_selected, top_k.indices
            )
            bbox_deltas_selected = tf.gather(
                bbox_deltas_selected, top_k.indices
            )
            fg_scores = tf.gather(fg_scores, top_k.indices)

            # apply non-maximum suppression (NMS)
            keep = tf.image.non_max_suppression(
                proposals, 
                fg_scores,
                max_output_size=self.post_nms_topk,
                iou_threshold=self.nms_thresh
            )

            # update proposals, objectness_logits, and bbox_deltas
            # create proposals_batch_indices
            proposals = tf.gather(proposals, keep)
            proposal_bis = tf.fill([tf.shape(proposals)[0]], index)
            objectness_logits_selected = tf.gather(
                objectness_logits_selected, keep
            )
            bbox_deltas_selected = tf.gather(bbox_deltas_selected, keep)
            
            tf.print(
                'proposals shape:',
                tf.shape(proposals),
                ', proposal_bis shape:',
                tf.shape(proposal_bis),
                ', objectness_logits shape:',
                tf.shape(objectness_logits_selected),
                ', bbox_deltas shape:',
                tf.shape(bbox_deltas_selected)
            )
            
            return tf.RaggedTensor.from_tensor(proposals), \
                proposal_bis,\
                tf.RaggedTensor.from_tensor(objectness_logits_selected), \
                tf.RaggedTensor.from_tensor(bbox_deltas_selected)
        
        results = tf.map_fn(
            topk_proposals_per_image,
            tf.range(batch_size),
            fn_output_signature=(
                tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.RaggedTensorSpec(shape=(None,), dtype=tf.int32),
                tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32)
            )
        )

        # proposals shape: [B, (None), 4]
        # batch_indices shape: [B, (None)]
        # objectness_logits shape: [B, (None), 1]
        # bbox_deltas shape: [B, (None), 4]
        # where (None) is the number of proposals for each image
        # in the batch, which can vary
        proposals, batch_indices, objectness_logits, bbox_deltas = results

        # flatten proposals, batch_indices, objectness_logits, and bbox_deltas
        # to make them compatible with the rest of the model
        # 
        # proposals shape: [B, (None), 4] -> [B*(None), 4]
        # batch_indices shape: [B, (None)] -> [B*(None)]
        # objectness_logits shape: [B, (None), 1] -> [B*(None), 1]
        # bbox_deltas shape: [B, (None), 4] -> [B*(None), 4]
        # 
        proposals = proposals.merge_dims(0, 1)
        if isinstance(proposals, tf.RaggedTensor):
            proposals = proposals.to_tensor()

        batch_indices = batch_indices.merge_dims(0, 1)
        if isinstance(batch_indices, tf.RaggedTensor):
            batch_indices = batch_indices.to_tensor()

        objectness_logits = objectness_logits.merge_dims(0, 1)
        if isinstance(objectness_logits, tf.RaggedTensor):
            objectness_logits = objectness_logits.to_tensor()

        bbox_deltas = bbox_deltas.merge_dims(0, 1)
        if isinstance(bbox_deltas, tf.RaggedTensor):
            bbox_deltas = bbox_deltas.to_tensor()
        
        return proposals, batch_indices, objectness_logits, bbox_deltas
    