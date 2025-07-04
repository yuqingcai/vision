
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

images shape: [2 230 298 3] 
p2: [2 58 75 256] 
p3: [2 29 38 256] 
p4: [2 15 19 256] 
p5: [2 8 10 256]

anchors: [39150 4] stride: 4 base_size: 32
anchors: [9918 4] stride: 8 base_size: 64
anchors: [2565 4] stride: 16 base_size: 128
anchors: [720 4] stride: 32 base_size: 256

230 / 4 = 57.5 -> 58,       298 / 4 = 74.5 -> 75        58 * 75 * 9 = 39150
230 / 8 = 28.75 -> 29,      298 / 8 = 37.25 -> 38
230 / 16 = 14.375 -> 15,    298 / 16 = 18.625 -> 19
230 / 32 = 7.1875 -> 8,     298 / 32 = 9.3125 -> 10


class ROIBBoxHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.bbox_pred = layers.Dense(self.num_classes * 4, activation=None)

    def call(self, features, batch_indices):
        """
            features shape: [N, sample_size, sample_size, feature_size]
        """
        batch_size = tf.reduce_max(batch_indices) + 1

        def bbox_deltas_per_image(index):
            mask = tf.equal(batch_indices, index)
            features_per_image = tf.boolean_mask(features, mask)
            features_indices = tf.boolean_mask(
                tf.range(tf.shape(features)[0]), mask
            )

            # flatten features to shape [ M, feature_dim ]
            shape = tf.shape(features_per_image)
            feature_dim = shape[1] * shape[2] * shape[3]
            x = tf.reshape(features_per_image, [ -1,  feature_dim ])
            x = self.fc1(x)
            x = self.fc2(x)
            bbox_deltas = self.bbox_pred(x)

            # bbox_deltas shape: [M, num_classes * 4]
            # features_indices shape: [M]
            return tf.RaggedTensor.from_tensor(bbox_deltas), \
                features_indices

        results = tf.map_fn(
            bbox_deltas_per_image,
            tf.range(batch_size),
            fn_output_signature=(
                tf.RaggedTensorSpec(
                    shape=(None, self.num_classes*4), 
                    dtype=tf.float32
                ),
                tf.RaggedTensorSpec(
                    shape=(None,), 
                    dtype=tf.int32
                )
            )
        )
        
        bbox_deltas, indices = results

        # sort all indices to maintain the order of features
        indices = indices.merge_dims(0, 1)
        if isinstance(indices, tf.RaggedTensor):
            indices = indices.to_tensor()
        indices = tf.argsort(indices, axis=0)
        
        # bbox_deltas shape: [ M, num_classes * 4 ]
        bbox_deltas = bbox_deltas.merge_dims(0, 1)
        if isinstance(bbox_deltas, tf.RaggedTensor):
            bbox_deltas = bbox_deltas.to_tensor()
        bbox_deltas = tf.gather(bbox_deltas, indices)


        # tf.print(
        #     'ROIBBoxHead',
        #     'bbox_deltas:', tf.shape(bbox_deltas)
        # )

        return bbox_deltas