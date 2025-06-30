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

    def call(self, feature_maps, rois, batch_indices):

        """
            feature_maps is a list of feature map, each feature map is 
            a [B, H, W, C] tensor.
            rois is a [N, 4] tensor, where each row is [x1, y1, x2, y2]
            calculate roi_level, it is used to select the feature map.
            roi_level is determined by the size of the roi, the range 
            is [2, 5] corresponding P2, P3, P4 and P5 feature maps.
            small rois are assigned to P2, large rois are assigned to P5.
            roi_level shape is [N, 1], where N is the number of rois.
        """

        tf.print('rois shape:', tf.shape(rois),
                'batch_indices shape:', tf.shape(batch_indices))
                
        x1, y1, x2, y2 = tf.split(rois, 4, axis=1)
        roi_h = y2 - y1
        roi_w = x2 - x1
        roi_area = roi_h * roi_w
        levels = tf.math.log(tf.sqrt(roi_area) / 224.0) / \
            tf.math.log(2.0) + 4.0
        levels = tf.squeeze(levels, axis=1)
        levels = tf.clip_by_value(tf.cast(tf.round(levels), tf.int32), 2, 5)
        
        all_features = []
        all_indices = []
        all_feature_batch_indices = []

        for i, stride in enumerate(self.feature_strides):
            # i + 2 because level starts from 2 and 
            # feature_maps index starts from 0
            level = i + 2
            mask = tf.equal(levels, level)
            rois_selected = tf.boolean_mask(rois, mask)
            roi_indices_selected = tf.boolean_mask(
                tf.range(tf.shape(rois)[0]), mask
            )
            batch_indices_selected = tf.boolean_mask(batch_indices, mask)

            def with_features(rois, roi_indices, batch_indices, feature_maps, 
                              stride, output_size):
                
                # select the feature map for the current level then
                # normalized roi box to [0, 1], order is [y1, x1, y2, x2]
                # feature_maps shape: [B, H, W, C]
                feature_maps = tf.gather(feature_maps, batch_indices)
                scale = 1.0 / stride
                bboxes = rois * scale
                fm_heights = tf.cast(tf.shape(feature_maps)[1], dtype=tf.float32)
                fm_widths = tf.cast(tf.shape(feature_maps)[2], dtype=tf.float32)
                x1 = bboxes[:, 0] / fm_widths
                y1 = bboxes[:, 1] / fm_heights
                x2 = bboxes[:, 2] / fm_widths
                y2 = bboxes[:, 3] / fm_heights
                normalized_boxes = tf.stack([y1, x1, y2, x2], axis=1)
                bbox_indices = tf.range(tf.shape(rois)[0])

                # features shape: [M, output_size, output_size, C]
                features = tf.image.crop_and_resize(
                    feature_maps,       # [M, H, W, C]
                    normalized_boxes,   # [M, 4], each row is [y1, x1, y2, x2]
                    bbox_indices,       # [M]
                    crop_size=[output_size, output_size],
                    method="bilinear"
                )

                return features, roi_indices
                
            def no_features(rois_in_level, output_size, feature_size):
                tf.print('rois_in_level', rois_in_level, ' no_features')
                return (
                    tf.zeros(
                        [ 0, output_size, output_size, feature_size ], 
                        dtype=tf.float32
                    ),
                    tf.zeros([0], dtype=tf.int32)
                )

            features, indices = tf.cond(
                tf.shape(rois_selected[0]) > 0,
                lambda: with_features(
                    rois_selected, 
                    roi_indices_selected, 
                    batch_indices_selected,
                    feature_maps[i], 
                    stride, 
                    self.output_size
                ),
                lambda: no_features(
                    rois_selected,
                    self.output_size, 
                    self.feature_size
                )
            )

            all_features.append(features)
            all_indices.append(indices)
            all_feature_batch_indices.append(batch_indices_selected)

        # sort all indices to maintain the order of rois
        roi_indices = tf.argsort(tf.concat(all_indices, axis=0), axis=0)

        roi_features = tf.gather(tf.concat(all_features, axis=0), roi_indices)
        roi_feature_batch_indices = tf.gather(
            tf.concat(all_feature_batch_indices, axis=0), 
            roi_indices
        )
        
        tf.debugging.assert_equal(
            tf.shape(roi_features)[0], tf.shape(rois)[0],
            message="roi features batch size does not match rois batch size"
        )

        # roi_features shape: [N, sample_size, sample_size, feature_size]
        # roi_feature_batch_indices shape: [N]        
        tf.print(
            'rois shape:', tf.shape(rois),
            'roi_features shape:', 
            tf.shape(roi_features),
            ', roi_feature_batch_indices shape:', 
            tf.shape(roi_feature_batch_indices))
        return roi_features, roi_feature_batch_indices
    