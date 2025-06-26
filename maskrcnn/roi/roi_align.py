import tensorflow as tf
from tensorflow.keras import layers


class ROIAlign(layers.Layer):
    def __init__(self, output_size, sampling_ratio, 
                 feature_strides, feature_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.feature_strides = feature_strides
        self.feature_size = feature_size

    def call(self, feature_maps, rois):
        features = tf.map_fn(
            lambda args: self.batch_features(args[0], args[1]),
            elems=(feature_maps, rois),
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[
                    None, 
                    self.output_size,
                    self.output_size,
                    self.feature_size
                ],
                dtype=tf.float32,
                ragged_rank=1
            ),
            parallel_iterations=32
        )
        return features
    
    def batch_features(self, feature_maps, rois):
        if isinstance(rois, tf.RaggedTensor):
            rois = rois.to_tensor()
        
        """
        feature_maps is a list of feature maps, each feature map 
        is a [H, W, C] tensor.
        rois is a [N, 4] tensor, where each row is [x1, y1, x2, y2]
        calculate roi_level, it is used to select the feature map.
        roi_level is determined by the size of the roi, the range 
        is [2, 5] corresponding P2, P3, P4 and P5 feature maps.
        small rois are assigned to P2, large rois are assigned to P5.
        roi_level shape is [N, 1], where N is the number of rois.
        """
        x1, y1, x2, y2 = tf.split(rois, 4, axis=1)
        roi_h = y2 - y1
        roi_w = x2 - x1
        roi_area = roi_h * roi_w
        roi_levels = tf.math.log(tf.sqrt(roi_area) / 224.0) / \
            tf.math.log(2.0) + 4.0
        roi_levels = tf.squeeze(roi_levels, axis=1)
        roi_levels = tf.clip_by_value(
            tf.cast(tf.round(roi_levels), tf.int32), 2, 5
        )
        
        all_features = []
        all_indices = []

        # select the feature map according to roi_level
        for i, stride in enumerate(self.feature_strides):
            # i + 2 because roi_level starts from 2
            # and feature_maps starts from 0
            level = i + 2
            mask = tf.equal(roi_levels, level)
            rois_in_level = tf.boolean_mask(rois, mask)
            indices_in_level = tf.boolean_mask(
                tf.range(tf.shape(rois)[0]), mask
            )

            def rois_features(rois, indices, feature_map, stride, output_size):
                scale = 1.0 / stride
                boxes = rois * scale
                # normalized to [0, 1], order is [y1, x1, y2, x2]
                fm_height = tf.cast(tf.shape(feature_map)[0], tf.float32)
                fm_width = tf.cast(tf.shape(feature_map)[1], tf.float32)
                x1 = boxes[:, 0] / fm_width
                y1 = boxes[:, 1] / fm_height
                x2 = boxes[:, 2] / fm_width
                y2 = boxes[:, 3] / fm_height
                normalized_boxes = tf.stack([y1, x1, y2, x2], axis=1)
                box_indices = tf.zeros([tf.shape(rois)[0]], dtype=tf.int32)

                # features shape: [M, output_size, output_size, C]
                feature_map_expand = tf.expand_dims(feature_map, axis=0)
                features = tf.image.crop_and_resize(
                    feature_map_expand, # [1, H, W, C]
                    normalized_boxes,   # [M, 4], each row is [y1, x1, y2, x2]
                    box_indices,        # [M], each value is 0
                    crop_size=[output_size, output_size],
                    method="bilinear"
                )
                return features, indices
            
            def no_features(output_size, feature_size):
                return (
                    tf.zeros(
                        [ 0, output_size, output_size, feature_size ], 
                        dtype=tf.float32
                    ),
                    tf.zeros([0], dtype=tf.int32)
                )
            
            features, indices = tf.cond(
                tf.shape(rois_in_level)[0] > 0,
                lambda: rois_features(
                    rois_in_level, 
                    indices_in_level, 
                    feature_maps[i], 
                    stride, 
                    self.output_size
                ),
                lambda: no_features(
                    self.output_size, 
                    self.feature_size
                )
            )
            
            all_features.append(features)
            all_indices.append(indices)

        all_features = tf.concat(all_features, axis=0)
        all_indices = tf.concat(all_indices, axis=0)
        sorted_indices = tf.argsort(all_indices)
        sorted_features = tf.gather(all_features, sorted_indices)
        
        return tf.RaggedTensor.from_tensor(sorted_features)
