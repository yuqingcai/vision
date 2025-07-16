import tensorflow as tf
from tensorflow.keras import layers, mixed_precision

class ROIAlign(layers.Layer):

    def __init__(self, 
                 output_size, 
                 sampling_ratio, 
                 feature_strides, 
                 feature_size, 
                 **kwargs):
        
        super().__init__(
            dtype=mixed_precision.Policy('float32'), 
            **kwargs
        )
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.feature_strides = feature_strides
        self.feature_size = feature_size

    def call(self, feature_maps, rois, valid_mask, roi_size_pred):
        """feature_maps is a list of feature map, each feature map is 
        a [B, H, W, C] tensor.
        rois is a [B, N, 4] tensor, where each row is [x1, y1, x2, y2]
        calculate roi_level, it is used to select the feature map.
        roi_level is determined by the size of the roi, the range 
        is [2, 5] corresponding P2, P3, P4 and P5 feature maps.
        small rois are assigned to P2, large rois are assigned to P5.
        roi_level shape is [N, 1], where N is the number of rois.
        """
        # tf.print('ROIAlign rois:', tf.shape(rois))

        features = tf.map_fn(
            lambda args: self.roi_align_per_image(
                args[0], args[1], args[2]
            ),
            elems=(feature_maps, rois, valid_mask),
            fn_output_signature=tf.TensorSpec(
                shape=(
                    roi_size_pred, 
                    self.output_size, 
                    self.output_size, 
                    self.feature_size
                ), 
                # mixed_precision, using tf.float16
                dtype=tf.float16
            )
        )
        
        # tf.print('ROIAlign features:', tf.shape(features))
        return features
    
    def roi_align_per_image(self, feature_maps, rois, valid_mask):
        # tf.print(
        #     'roi_align_per_image valid_mask:', 
        #     tf.reduce_sum(tf.cast(valid_mask, tf.int32))
        # )
        
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

        for i, stride in enumerate(self.feature_strides):
            # i + 2 because level starts from 2 and 
            # feature_maps index starts from 0
            level = i + 2
            mask = tf.equal(levels, level)
            rois_in_level = tf.boolean_mask(rois, mask)
            roi_indices = tf.boolean_mask(
                tf.range(tf.shape(rois)[0]), mask
            )
            
            features = self.rois_features(
                rois_in_level, 
                feature_maps[i], 
                stride,
                level=level
            )

            all_indices.append(roi_indices)
            all_features.append(features)
        
        indices = tf.argsort(tf.concat(all_indices, axis=0), axis=0)
        features = tf.gather(tf.concat(all_features, axis=0), indices)

        tf.debugging.assert_equal(
            tf.shape(features)[0], tf.shape(rois)[0],
            message="roi features batch size does not match rois batch size"
        )
        
        return features
    
    def rois_features(self, rois, feature_map, stride, level):
        # normalized roi box to [0, 1], order is [y1, x1, y2, x2]
        # feature_maps shape: [H, W, C]
        # 
        # rois shape: [N, 4], if N is 0, tf.image.crop_and_resize will 
        # return empty tensor

        scale = 1.0 / stride
        bboxes = rois * scale

        fm_height = tf.cast(tf.shape(feature_map)[0], dtype=tf.float32)
        fm_width = tf.cast(tf.shape(feature_map)[1], dtype=tf.float32)
        
        # normalize bboxes to [0, 1]
        # bboxes shape: [N, 4]
        x1 = bboxes[:, 0] / fm_width
        y1 = bboxes[:, 1] / fm_height
        x2 = bboxes[:, 2] / fm_width
        y2 = bboxes[:, 3] / fm_height
        
        # stack to [N, 4] tensor
        # where each row is [y1, x1, y2, x2]
        normalized_boxes = tf.stack([ y1, x1, y2, x2 ], axis=1)
        bbox_indices = tf.zeros(
            [ tf.shape(rois)[0] ], dtype=tf.int32
        )
        # feature_map shape: [H, W, C] -> [1, H, W, C]
        feature_map = tf.expand_dims(feature_map, axis=0)

        # features shape: [N, output_size, output_size, C]
        # if normalized_boxes is empty then crop_and_resize 
        # will return empty tensor
        features = tf.image.crop_and_resize(
            feature_map, 
            normalized_boxes, 
            bbox_indices, 
            [ self.output_size, self.output_size ],
            method='bilinear',
        )
        # mixed_precision, using tf.float16
        features = tf.cast(features, tf.float16) 
        
        # tf.print('rois_features:', tf.shape(features), 
        #          'in feature_map', level, 
        #          'rois:', tf.shape(rois)
        # )
        # features shape: [N, output_size, output_size, C]
        return features
    
