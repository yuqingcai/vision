import tensorflow as tf
from tensorflow.keras import layers

class ROIBBoxHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.bbox_pred = layers.Dense(num_classes * 4)

    def call(self, inputs):
        """
        inputs shape: [batch, (N), sample_size, sample_size, feature_size], 
                       it's a RaggedTensor
        """
        bbox_deltas = tf.map_fn(
            self.forward, 
            inputs, 
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[ None, self.num_classes * 4 ], 
                dtype=tf.float32
            )
        )
        return bbox_deltas
    
    def forward(self, x):
        """
        x shape: [ N, sample_size, sample_size, feature_size ], 
        bbox_deltas shape: [ N, num_classes*4 ]
        """
        x_shape = tf.shape(x)
        feature_dim = x_shape[1] * x_shape[2] * x_shape[3]
        
        # flatten x to shape [ N, feature_dim ]
        x = tf.reshape(x, [ -1,  feature_dim ])

        x = self.fc1(x)
        x = self.fc2(x)
        bbox_deltas = self.bbox_pred(x)
        return tf.RaggedTensor.from_tensor(bbox_deltas)