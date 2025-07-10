import tensorflow as tf
from tensorflow.keras import layers

class ROIClassifierHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.class_logits_pred = layers.Dense(num_classes, activation=None)
    
    def call(self, features, valid_mask, features_size_pred):
        """features shape: [B, N, S, S, F]
        valid_mask shape: [B, N]
        where B is batch size, N is number of ROIs,
        S is the size of the ROI (e.g., 7 for 7x7),
        and F is the feature dimension (e.g., 256).
        features_size_pred is the number of predicted class 
        logits per image.
        """

        def class_logits_per_image(features, valid_mask):
            features_valid = tf.boolean_mask(features, valid_mask)
            valid_indices = tf.boolean_mask(
                tf.range(tf.shape(features)[0]), 
                valid_mask
            )

            pad_indices = tf.boolean_mask(
                tf.range(tf.shape(features)[0]), 
                tf.logical_not(valid_mask)
            )

            # flatten features to shape [ M, feature_dim ]
            shape = tf.shape(features_valid)
            feature_dim = shape[1] * shape[2] * shape[3]
            x = tf.reshape(features_valid, [ -1,  feature_dim ])
            x = self.fc1(x)
            x = self.fc2(x)
            class_logits = self.class_logits_pred(x)

            class_logits = tf.pad(
                class_logits, 
                [[0, tf.shape(pad_indices)[0]], [0, 0]], 
                constant_values=0.0
            )

            indices = tf.concat([valid_indices, pad_indices], axis=0)
            indices_sorted = tf.argsort(indices, axis=0)
            class_logits = tf.gather(class_logits, indices_sorted)

            # class_logits shape: [ M, num_classes ]
            return class_logits

        results = tf.map_fn(
            lambda args: class_logits_per_image(
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
                        self.num_classes
                    ), 
                    dtype=tf.float32
                )
            )
        )

        # tf.print('class_logits:', tf.shape(results))

        # results shape: [B, features_size_pred, num_classes]
        return results
    