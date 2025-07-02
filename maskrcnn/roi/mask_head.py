import tensorflow as tf
from tensorflow.keras import layers


class ROIMaskHead(layers.Layer):
    def __init__(self, 
                 num_classes, 
                 conv_dim, 
                 num_convs, 
                 roi_output_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.conv_layers = [
            layers.Conv2D(conv_dim, 3, padding='same', activation='relu') \
                for _ in range(num_convs)
        ]
        self.upsample = layers.Conv2DTranspose(
            conv_dim, 2, strides=2, padding='valid', activation='relu'
        )

        self.mask_pred = tf.keras.layers.Conv2D(
            num_classes, 1, activation=None 
        )

        self.resolution = roi_output_size * 2

    def call(self, features, batch_indices):
        """
            features shape: [N, sample_size, sample_size, feature_size]
        """

        batch_size = tf.reduce_max(batch_indices) + 1

        def masks_per_image(index):
            mask = tf.equal(batch_indices, index)
            features_indices = tf.boolean_mask(
                tf.range(tf.shape(features)[0]), mask
            )

            # x shape [ M, H, W, C ]
            x = tf.boolean_mask(features, mask)
            for conv in self.conv_layers:
                x = conv(x)
            x = self.upsample(x)
            masks = self.mask_pred(x)

            # masks shape: [M, resolution, resolution, num_classes]
            # features_indices shape: [M]
            return tf.RaggedTensor.from_tensor(masks), \
                features_indices

        results = tf.map_fn(
            masks_per_image,
            tf.range(batch_size),
            fn_output_signature=(
                tf.RaggedTensorSpec(
                    shape=(None, self.resolution, self.resolution, self.num_classes), 
                    dtype=tf.float32,
                    ragged_rank=1
                ),
                tf.RaggedTensorSpec(
                    shape=(None,), 
                    dtype=tf.int32
                )
            )
        )

        masks, indices = results

        # sort all indices to maintain the order of features
        indices = indices.merge_dims(0, 1)
        if isinstance(indices, tf.RaggedTensor):
            indices = indices.to_tensor()
        indices = tf.argsort(indices, axis=0)

        # masks shape: [M, resolution, resolution, num_classes]
        masks = masks.merge_dims(0, 1)
        if isinstance(masks, tf.RaggedTensor):
            masks = masks.to_tensor()
        masks = tf.gather(masks, indices)

        # tf.print(
        #     'ROIMaskHead', 
        #     'masks:', tf.shape(masks),
        # )

        return masks