import tensorflow as tf
from tensorflow.keras import layers

class ROIBBoxHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.bbox_pred = layers.Dense(num_classes * 4)

    def call(self, features):
        """
            features shape: [N, sample_size, sample_size, feature_size]
        """
        shape = tf.shape(features)
        feature_dim = shape[1] * shape[2] * shape[3]
        # flatten features to shape [ N, feature_dim ]
        x = tf.reshape(features, [ -1,  feature_dim ])
        x = self.fc1(x)
        x = self.fc2(x)
        bbox_deltas = self.bbox_pred(x)
        
        tf.print('bbox_deltas shape:', tf.shape(bbox_deltas))
        return bbox_deltas
    