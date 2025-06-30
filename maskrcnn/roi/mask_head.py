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

    def call(self, features):
        """
            features shape: [N, sample_size, sample_size, feature_size]
        """
        for conv in self.conv_layers:
            x = conv(features)
        x = self.upsample(x)
        x = self.mask_pred(x)
        
        tf.print('mask shape: ', tf.shape(x))
        return x
