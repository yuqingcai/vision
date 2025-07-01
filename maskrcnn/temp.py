
    # def batch_features(self, feature_maps, rois):
    #     if isinstance(rois, tf.RaggedTensor):
    #         rois = rois.to_tensor()
    #     """
    #     feature_maps is a list of feature maps, each feature map 
    #     is a [H, W, C] tensor.
    #     rois is a [N, 4] tensor, where each row is [x1, y1, x2, y2]
    #     calculate roi_level, it is used to select the feature map.
    #     roi_level is determined by the size of the roi, the range 
    #     is [2, 5] corresponding P2, P3, P4 and P5 feature maps.
    #     small rois are assigned to P2, large rois are assigned to P5.
    #     roi_level shape is [N, 1], where N is the number of rois.
    #     """
    #     x1, y1, x2, y2 = tf.split(rois, 4, axis=1)
    #     roi_h = y2 - y1
    #     roi_w = x2 - x1
    #     roi_area = roi_h * roi_w
    #     roi_levels = tf.math.log(tf.sqrt(roi_area) / 224.0) / \
    #         tf.math.log(2.0) + 4.0
    #     roi_levels = tf.squeeze(roi_levels, axis=1)
    #     roi_levels = tf.clip_by_value(
    #         tf.cast(tf.round(roi_levels), tf.int32), 2, 5
    #     )
        
    #     features = tf.map_fn(
    #         lambda args: self.roi_features(args[0], args[1], feature_maps),
    #         elems=(roi_levels, rois),
    #         fn_output_signature=tf.TensorSpec(
    #             shape=[ 
    #                 self.output_size, 
    #                 self.output_size,
    #                 self.feature_size
    #             ],
    #             dtype=tf.float32,
    #         ),
    #         parallel_iterations=32
    #     )

    #     return tf.RaggedTensor.from_tensor(features)
    
    # def roi_features(self, roi_level, roi, feature_maps):
    #     """
    #     select the feature map according to roi_level, 
    #     roi_level starts from 2, and feature_maps starts 
    #     from 0
    #     """
    #     feature_map, stride = tf.switch_case(
    #         roi_level - 2,
    #         branch_fns=[
    #             lambda: (feature_maps[0], self.feature_strides[0]),
    #             lambda: (feature_maps[1], self.feature_strides[1]),
    #             lambda: (feature_maps[2], self.feature_strides[2]),
    #             lambda: (feature_maps[3], self.feature_strides[3]),
    #         ]
    #     )
        
    #     # scale to feature_map size
    #     scale = 1.0 / tf.cast(stride, tf.float32)
    #     box = roi * scale

    #     # normalized to [0, 1], order is [y1, x1, y2, x2]
    #     fm_height = tf.cast(tf.shape(feature_map)[0], tf.float32)
    #     fm_width = tf.cast(tf.shape(feature_map)[1], tf.float32)
    #     x1 = box[0] / fm_width
    #     y1 = box[1] / fm_height
    #     x2 = box[2] / fm_width
    #     y2 = box[3] / fm_height
        
    #     normalized_box = tf.stack([y1, x1, y2, x2], axis=0)

    #     # crop_and_resize, feature_map should be [1, H, W, C]
    #     # feature_map shape: [H, W, C]
    #     # normalized_box shape: [4], each row is [y1, x1, y2, x2]
    #     # box_indices shape: [1], each value is 0
    #     # features shape: [output_size, output_size, C]
    #     feature_map = tf.expand_dims(feature_map, axis=0)  # [1, H, W, C]
    #     normalized_box = tf.expand_dims(normalized_box, axis=0)  # [1, 4]
    #     box_indices = tf.zeros([1], dtype=tf.int32)  # [1]
    #     features = tf.image.crop_and_resize(
    #         feature_map, 
    #         normalized_box, 
    #         box_indices, 
    #         crop_size=[self.output_size, self.output_size],
    #         method="bilinear"
    #     )
    #     features = tf.squeeze(features, axis=0)
        
    #     # features = self.roi_align_single(
    #     #     feature_map, normalized_box, self.output_size, 
    #     #     self.sampling_ratio)
        
    #     return features
    
    # def roi_align_single(self, feature_map, box, output_size, sampling_ratio):
    #     """
    #     feature_map: [H, W, C]
    #     box: [y1, x1, y2, x2], normalized to [0, 1] relative to feature_map
    #     output_size: int, pooled output H and W
    #     sampling_ratio: int, number of samples per bin side 
    #                     (total sampling_ratio^2 samples per bin)
    #     return: [output_size, output_size, C]
    #     """
    #     H = tf.cast(tf.shape(feature_map)[0], tf.float32)
    #     W = tf.cast(tf.shape(feature_map)[1], tf.float32)
    #     C = tf.shape(feature_map)[2]

    #     y1, x1, y2, x2 = tf.unstack(box)

    #     roi_h = (y2 - y1) * H
    #     roi_w = (x2 - x1) * W

    #     bin_h = roi_h / output_size
    #     bin_w = roi_w / output_size

    #     outputs = []
    #     for ph in range(output_size):
    #         for pw in range(output_size):
    #             # bin's top-left in normalized coordinates
    #             bin_y1 = y1 * H + ph * bin_h
    #             bin_x1 = x1 * W + pw * bin_w

    #             samples = []
    #             for iy in range(sampling_ratio):
    #                 for ix in range(sampling_ratio):
    #                     sample_y = bin_y1 + (iy + 0.5) * bin_h / sampling_ratio
    #                     sample_x = bin_x1 + (ix + 0.5) * bin_w / sampling_ratio
    #                     sample_y = tf.clip_by_value(sample_y, 0, H - 1)
    #                     sample_x = tf.clip_by_value(sample_x, 0, W - 1)
    #                     val = self.bilinear_interpolate(
    #                         feature_map, 
    #                         sample_y, sample_x)
    #                     samples.append(val)
    #             samples = tf.stack(samples, axis=0)  # [sampling_ratio^2, C]
    #             pooled_val = tf.reduce_mean(samples, axis=0)  # [C]
    #             outputs.append(pooled_val)

    #     outputs = tf.stack(outputs, axis=0)
    #     return tf.reshape(outputs, [output_size, output_size, C])

    # def bilinear_interpolate(self, feature_map, y, x):
    #     """
    #     feature_map: [H, W, C]
    #     y, x: float, single position
    #     return: [C]
    #     """
    #     H = tf.shape(feature_map)[0]
    #     W = tf.shape(feature_map)[1]

    #     y0 = tf.cast(tf.floor(y), tf.int32)
    #     x0 = tf.cast(tf.floor(x), tf.int32)
    #     y1 = y0 + 1
    #     x1 = x0 + 1

    #     y0_clipped = tf.clip_by_value(y0, 0, H - 1)
    #     y1_clipped = tf.clip_by_value(y1, 0, H - 1)
    #     x0_clipped = tf.clip_by_value(x0, 0, W - 1)
    #     x1_clipped = tf.clip_by_value(x1, 0, W - 1)

    #     Ia = feature_map[y0_clipped, x0_clipped]
    #     Ib = feature_map[y1_clipped, x0_clipped]
    #     Ic = feature_map[y0_clipped, x1_clipped]
    #     Id = feature_map[y1_clipped, x1_clipped]

    #     wa = (tf.cast(x1, tf.float32) - x) * (tf.cast(y1, tf.float32) - y)
    #     wb = (tf.cast(x1, tf.float32) - x) * (y - tf.cast(y0, tf.float32))
    #     wc = (x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y)
    #     wd = (x - tf.cast(x0, tf.float32)) * (y - tf.cast(y0, tf.float32))

    #     return wa * Ia + wb * Ib + wc * Ic + wd * Id




class ProposalGenerator(layers.Layer):
    def __init__(self, pre_nms_topk=6000, post_nms_topk=1000, 
                 nms_thresh=0.7, min_size=16, **kwargs):
        super().__init__(**kwargs)
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_size = min_size
    
    def call(self, anchors, batch_indices, image_sizes,   
             class_logits, bbox_deltas):
            
        image_sizes = tf.gather(image_sizes, batch_indices)
        batch_size = tf.reduce_max(batch_indices) + 1

        def topk_per_image(index):
            mask = tf.equal(batch_indices, index)
            selected = tf.where(mask)[:, 0]
            anchors_selected = tf.gather(anchors, selected)
            class_logits_selected = tf.gather(class_logits, selected)
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

            # update proposals, class_logits and bbox_deltas
            proposals = tf.gather(proposals, valid[:, 0])
            class_logits_selected = tf.gather(class_logits_selected, valid[:, 0])
            bbox_deltas_selected = tf.gather(bbox_deltas_selected, valid[:, 0])

            fg_scores = tf.sigmoid(class_logits_selected[:, 0])
            # get tok-k pre nms proposals
            top_k = tf.math.top_k(
                fg_scores, 
                k=tf.minimum(self.pre_nms_topk, tf.shape(fg_scores)[0])
            )
            
            # update proposals, class_logits, bbox_deltas and fg_scores
            proposals = tf.gather(proposals, top_k.indices)
            class_logits_selected = tf.gather(
                class_logits_selected, top_k.indices
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

            # update proposals, class_logits, and bbox_deltas
            # create proposals_batch_indices
            proposals = tf.gather(proposals, keep)
            class_logits_selected = tf.gather(class_logits_selected, keep)
            bbox_deltas_selected = tf.gather(bbox_deltas_selected, keep)
            proposals_batch_indices = tf.fill([tf.shape(proposals)[0]], index)

            return proposals, proposals_batch_indices, \
                class_logits_selected, bbox_deltas_selected
        
        results = tf.map_fn(
            topk_per_image,
            tf.range(batch_size),
            fn_output_signature=(
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
            )
        )

        # proposals shape: [B, (None), 4]
        # batch_indices shape: [B, (None)]
        proposals, batch_indices, class_logits, bbox_deltas = results

        # flatten proposals and batch_indices
        # proposals shape: [B, (None), 4] -> [B*(None), 4]
        # batch_indices shape: [B, (None)] -> [B*(None)]
        proposals = tf.reshape(proposals, [-1, 4])
        batch_indices = tf.reshape(batch_indices, [-1])
        class_logits = tf.reshape(class_logits, [-1, 1])
        bbox_deltas = tf.reshape(bbox_deltas, [-1, 4])
        
        tf.print('proposals shape:', tf.shape(proposals))
        tf.print('batch_indices shape:', tf.shape(batch_indices))
        tf.print('class_logits shape:', tf.shape(class_logits))
        tf.print('bbox_deltas shape:', tf.shape(bbox_deltas))

        return proposals, batch_indices, class_logits, bbox_deltas



       shape = tf.shape(features)
        feature_dim = shape[1] * shape[2] * shape[3]
        # flatten features to shape [ N, feature_dim ]
        x = tf.reshape(features, [ -1,  feature_dim ])
        x = self.fc1(x)
        x = self.fc2(x)
        bbox_deltas = self.bbox_pred(x)
        
        tf.print('bbox_deltas shape:', tf.shape(bbox_deltas))
        return bbox_deltas
