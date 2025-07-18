import tensorflow as tf
from tensorflow.keras import layers

class ROIClassifierHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim, **kwargs):

        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.fc1 = layers.Dense(
            hidden_dim, 
            activation='relu', 
            dtype=tf.float32
        )
        self.fc2 = layers.Dense(
            hidden_dim, 
            activation='relu', 
            dtype=tf.float32
        )
        self.class_logits_pred = layers.Dense(
            num_classes, 
            activation=None,
            dtype=tf.float32
            )
    
    def call(self, features, valid_mask):
        """features shape: [B, N, S, S, F]
        valid_mask shape: [B, N]
        where B is batch size, N is number of ROIs,
        S is the size of the ROI (e.g., 7 for 7x7),
        and F is the feature dimension (e.g., 256).
        """
        B = tf.shape(features)[0]
        N = tf.shape(features)[1]
        S = tf.shape(features)[2]
        F = tf.shape(features)[4]

        # flatten features to shape: [B, N, S * S * F]
        features_flat = tf.reshape(features, [B, N, S * S * F])

        # valid_mask_float shape: [B, N, 1]
        valid_mask_float = tf.cast(valid_mask, tf.float32)
        valid_mask_float = tf.expand_dims(valid_mask_float, axis=-1)
        
        # class_logits shape: [B, N, num_classes]
        x = self.fc1(features_flat)
        x = self.fc2(x)
        class_logits = self.class_logits_pred(x)  

        # mask invalid class logits, set to -1e8
        # This is to ensure that invalid ROIs do not contribute to the loss.
        class_logits = class_logits * valid_mask_float + \
            (1.0 - valid_mask_float) * (-1e8)

        # tf.print('class_logits:', tf.shape(class_logits))

        # class_logits shape: [B, N, num_classes]
        return class_logits  
    