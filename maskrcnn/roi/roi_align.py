import tensorflow as tf
from tensorflow.keras import layers
from utils import sync_flatten_batch


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
            batch_indices_in_level =  tf.boolean_mask(
                batch_indices, mask
            )
            
            tf.print('roi_levels shape:', tf.shape(roi_levels))
            tf.print('rois_in_level shape:', tf.shape(rois_in_level))
            tf.print('indices_in_level shape:', tf.shape(indices_in_level))
            tf.print('batch_indices_in_level shape:', tf.shape(batch_indices_in_level))
            
            
            # features, indices = tf.cond(
            #     tf.shape(rois_in_level)[0] > 0,
            #     lambda: rois_features(
            #         rois_in_level, 
            #         indices_in_level, 
            #         batch_indices_in_level,
            #         feature_maps[i], 
            #         stride, 
            #         self.output_size
            #     ),
            #     lambda: no_features(
            #         self.output_size, 
            #         self.feature_size
            #     )
            # )
            
            # all_features.append(features)
            # all_indices.append(indices)

        
        features = None
        return features, batch_indices
    