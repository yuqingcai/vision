import tensorflow as tf
from tensorflow.keras import layers

class ROIClassifierHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.class_logits = layers.Dense(num_classes, activation=None)
    
    def call(self, inputs):
        """
        inputs shape: [batch, (N), sample_size, sample_size, feature_size], 
                       it's a RaggedTensor
        """
        class_logits = tf.map_fn(
            self.forward, 
            inputs, 
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[ None, self.num_classes ], 
                dtype=tf.float32
            )
        )
        return class_logits

    def forward(self, x):
        """
        x shape: [ N, sample_size, sample_size, feature_size ], 
        class_logits shape: [ N, num_classes ]
        """
        x_shape = tf.shape(x)
        feature_dim = x_shape[1] * x_shape[2] * x_shape[3]
        
        # flatten x to shape [ N, feature_dim ]
        x = tf.reshape(x, [ -1,  feature_dim ])

        x = self.fc1(x)
        x = self.fc2(x)
        class_logits = self.class_logits(x)
        return tf.RaggedTensor.from_tensor(class_logits)
    