import tensorflow as tf
from tensorflow.keras import layers

class ROIBBoxHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.bbox_pred = layers.Dense(
            self.num_classes * 4, activation=None
        )

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

        tf.print(
            'bbox_deltas shape:', tf.shape(bbox_deltas)
        )

        return bbox_deltas
    
    