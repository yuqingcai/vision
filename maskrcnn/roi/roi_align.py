import tensorflow as tf
from tensorflow.keras import layers


class ROIAlign(layers.Layer):
    def __init__(self, output_size, sampling_ratio, 
                 feature_map_strides, feature_map_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.feature_map_strides = feature_map_strides
        self.feature_map_size = feature_map_size

    def call(self, feature_maps, rois):
        tf.print('feature_maps shape:', [tf.shape(fm) for fm in feature_maps])
        tf.print('rois shape:', tf.shape(rois))

        features = tf.map_fn(
            lambda args: self.features(args[0], args[1]),
            elems=(feature_maps, rois),
            fn_output_signature=tf.RaggedTensorSpec(
                shape=(None, 
                       self.feature_map_size, 
                       self.output_size, 
                       self.output_size),
                dtype=tf.float32,
                ragged_rank=1
            ),
            parallel_iterations=32
        )
        tf.print('features shape:', tf.shape(features))
        return features
    

    def features(self, feature_maps, rois):
        if isinstance(rois, tf.RaggedTensor):
            rois = rois.to_tensor()

        # calculate roi_level, it is used to select 
        # the feature map.
        # roi_level is determined by the size of the roi, 
        # the range is [2, 5] corresponding P2, P3, P4, 
        # and P5 feature maps
        x1, y1, x2, y2 = tf.split(rois, 4, axis=1)
        roi_h = y2 - y1
        roi_w = x2 - x1
        roi_area = roi_h * roi_w
        roi_level = tf.math.log(tf.sqrt(roi_area) / 224.0) / \
            tf.math.log(2.0) + 4.0
        roi_level = tf.squeeze(roi_level, axis=1)
        roi_level = tf.clip_by_value(
            tf.cast(tf.round(roi_level), tf.int32), 2, 5
        )
        
        # select the feature map according to roi_level
        for i, stride in enumerate(self.feature_map_strides):
            # i + 2 because roi_level starts from 2
            # and feature_maps starts from 0
            level = i + 2
            mask = tf.equal(roi_level, level)
            rois_level = tf.boolean_mask(rois, mask)

            if tf.shape(rois_level)[0] == 0:
                continue
            
            # scale to feature_map size
            scale = 1.0 / stride
            bboxes = rois_level * scale

            # normalized to [0, 1], order is [y1, x1, y2, x2]
            # fm_height = tf.cast(tf.shape(feature_maps[i])[0], tf.float32)
            # fm_width = tf.cast(tf.shape(feature_maps[i])[1], tf.float32)
            # x1 = bboxes[:, 0] / fm_width
            # y1 = bboxes[:, 1] / fm_height
            # x2 = bboxes[:, 2] / fm_width
            # y2 = bboxes[:, 3] / fm_height
            # normalized_boxes = tf.stack([y1, x1, y2, x2], axis=1)  # [M, 4]

            # box_indices = tf.zeros([tf.shape(rois_level)[0]], dtype=tf.int32)

            # # crop_and_resize, feature_map should be [1, H, W, C]
            # feature_map = tf.expand_dims(feature_maps[i], axis=0)
            # pooled = tf.image.crop_and_resize(
            #     feature_map,                   # [1, H, W, C]
            #     normalized_boxes,              # [M, 4], [y1, x1, y2, x2]
            #     box_indices,                   # [M]
            #     crop_size=[self.output_size, self.output_size],
            #     method="bilinear"
            # )
            # tf.print('+pooled shape:', tf.shape(pooled))
            # pooled = tf.squeeze(pooled, axis=0)
            # tf.print('-pooled shape:', tf.shape(pooled))

        n = tf.shape(rois)[0]
        return tf.RaggedTensor.from_tensor(
            tf.fill(
                [n, self.feature_map_size, self.output_size, self.output_size],
                3.0,
            ),
        )
    
