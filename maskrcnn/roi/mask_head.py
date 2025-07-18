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
            layers.Conv2D(
                conv_dim, 
                3, 
                padding='same', 
                dtype=tf.float32, 
                activation='relu'
            ) for _ in range(num_convs)
        ]

        self.upsample = layers.Conv2DTranspose(
            conv_dim, 
            2, 
            strides=2, 
            padding='valid', 
            dtype=tf.float32,
            activation='relu'
        )

        self.mask_pred = tf.keras.layers.Conv2D(
            num_classes, 
            1, 
            dtype=tf.float32, 
            activation=None 
        )

        self.resolution = roi_output_size * 2

    def call(self, features, valid_mask):
        """features shape: [B, N, S, S, F]
        valid_mask shape: [B, N]
        where B is batch size, N is number of ROIs,
        S is the size of the ROI (e.g., 14 for 14x14),
        and F is the feature dimension (e.g., 256).
        """
        B = tf.shape(features)[0]
        N = tf.shape(features)[1]
        S = tf.shape(features)[2]
        F = tf.shape(features)[4]
        
        # valid_mask_float shape: [B, N, 1]
        valid_mask_float = tf.cast(valid_mask, tf.float32)
        valid_mask_float = tf.reshape(valid_mask_float, [B, N, 1, 1, 1])
        
        # masks shape [num_valid, resolution, resolution, num_classes]
        x = tf.reshape(features, [B*N, S, S, F])
        for conv in self.conv_layers:
            x = conv(x)
        x = self.upsample(x)
        masks = self.mask_pred(x)

        # masks shape: [B, N, resolution, resolution, num_classes]
        masks = tf.reshape(
            masks, 
            [B, N, self.resolution, self.resolution, self.num_classes]
        )
        masks = masks * valid_mask_float
        
        # tf.print('masks:', tf.shape(masks))
        # masks_padded: [B, N, resolution, resolution, num_classes]
        
        return masks
