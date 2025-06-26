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

    def call(self, inputs):

        masks = tf.map_fn(
            self.forward, 
            inputs, 
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[ 
                    None, 
                    self.resolution, 
                    self.resolution, 
                    self.num_classes 
                ], 
                dtype=tf.float32,
                ragged_rank=1
            )
        )
        return masks
    
    def forward(self, x):
        """
        x shape: [ N, sample_size, sample_size, feature_size ], 
        mask shape: [ N, resolution, resolution, num_classes ]
        """
        masks = tf.map_fn(
            self.generate,
            x,
            fn_output_signature=tf.TensorSpec(
                shape=[
                    self.resolution, 
                    self.resolution, 
                    self.num_classes 
                ], 
                dtype=tf.float32
            )
        )
        return tf.RaggedTensor.from_tensor(masks)

    def generate(self, x):
        """
        x shape: [ sample_size, sample_size, feature_size ], 
        mask shape: [ resolution, resolution, num_classes ]
        """
        # expand dims to add batch dimension
        x = tf.expand_dims(x, axis=0)
        for conv in self.conv_layers:
            x = conv(x)
        x = self.upsample(x)
        x = self.mask_pred(x)
        # squeezing the batch dimension
        x = tf.squeeze(x, axis=0)
        return x
    