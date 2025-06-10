import tensorflow as tf
from tensorflow.keras import layers, Model

class RoIAlign(layers.Layer):
    def __init__(self, output_size=7, sampling_ratio=2, **kwargs):
        """
        output_size: 输出特征的空间尺寸（如7，输出为7x7）
        sampling_ratio: 每个bin采样点数
        """
        super().__init__(**kwargs)
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
    
    def call(self, feature_maps, rois, strides=[4, 8, 16, 32]):
        """
        feature_maps: [P2, P3, P4, P5], 每层shape为(batch, H, W, C)
        rois: (num_rois, 5), 每行 [batch_idx, x1, y1, x2, y2]，坐标为原图尺度
        strides: FPN每层相对原图的步长
        """
        # 计算proposal分配到哪个FPN层
        x1, y1, x2, y2 = tf.split(rois[:, 1:], 4, axis=1)
        roi_h = y2 - y1
        roi_w = x2 - x1
        roi_level = tf.math.log(tf.sqrt(roi_h * roi_w) / 224.0) / tf.math.log(2.0)
        roi_level = tf.clip_by_value(tf.cast(tf.round(4 + roi_level), tf.int32), 2, 5)

        pooled_features = []
        for i, stride in enumerate(strides):
            level = i + 2
            idx_in_level = tf.where(tf.equal(roi_level[:, 0], level))
            rois_level = tf.gather(rois, idx_in_level[:, 0])

            if tf.shape(rois_level)[0] == 0:
                continue

            # 将RoI从原图尺度映射到feature map尺度
            scale = 1.0 / stride
            boxes = rois_level[:, 1:] * scale
            box_indices = tf.cast(rois_level[:, 0], tf.int32)

            # tf.image.crop_and_resize: [batch, H, W, C], boxes: [num_rois, 4]
            features = tf.image.crop_and_resize(
                feature_maps[i], boxes,
                box_indices,
                crop_size=[self.output_size, self.output_size],
                method="bilinear"
            )
            pooled_features.append((idx_in_level[:, 0], features))

        # 合并所有FPN层的输出，并恢复proposal顺序
        indices = tf.concat([x[0] for x in pooled_features], 0)
        proposal_features = tf.concat([x[1] for x in pooled_features], 0)
        # 恢复到输入rois的顺序
        order = tf.argsort(indices)
        proposal_features = tf.gather(proposal_features, order)
        return proposal_features  # shape: (N, output_size, output_size, C)