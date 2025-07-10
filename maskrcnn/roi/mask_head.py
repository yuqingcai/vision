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

    def call(self, features, valid_mask, features_size_pred):
        """features shape: [B, N, S, S, F]
        valid_mask shape: [B, N]
        where B is batch size, N is number of ROIs,
        S is the size of the ROI (e.g., 14 for 14x14),
        and F is the feature dimension (e.g., 256).
        features_size_pred is the number of predicted class 
        logits per image.
        """

        def masks_per_image(features, valid_mask):
            features_valid = tf.boolean_mask(features, valid_mask)
            valid_indices = tf.boolean_mask(
                tf.range(tf.shape(features)[0]), 
                valid_mask
            )

            pad_indices = tf.boolean_mask(
                tf.range(tf.shape(features)[0]), 
                tf.logical_not(valid_mask)
            )

            # x shape [ M, H, W, C ]
            x = features_valid
            for conv in self.conv_layers:
                x = conv(x)
            x = self.upsample(x)
            masks = self.mask_pred(x)

            masks = tf.pad(
                masks, 
                [[0, tf.shape(pad_indices)[0]], [0, 0], [0, 0], [0, 0]], 
                constant_values=0.0
            )

            indices = tf.concat([valid_indices, pad_indices], axis=0)
            indices_sorted = tf.argsort(indices, axis=0)
            masks = tf.gather(masks, indices_sorted)

            # masks shape: [ M, resolution, resolution, num_classes ]
            return masks


        results = tf.map_fn(
            lambda args: masks_per_image(
                args[0], 
                args[1]
            ),
            elems=(
                features, 
                valid_mask
            ),
            fn_output_signature=(
                tf.TensorSpec(
                    shape=(
                        features_size_pred, 
                        self.resolution, 
                        self.resolution, 
                        self.num_classes), 
                    dtype=tf.float32,
                )
            )
        )

        # tf.print(
        #     'ROIMaskHead', 
        #     'masks:', tf.shape(results),
        # )
        
        # results shape: [
        #   B, 
        #   features_size_pred, 
        #   resolution, 
        #   resolution, 
        #   num_classes
        # ]
        return results