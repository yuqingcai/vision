import tensorflow as tf
from tensorflow.keras import layers, Model


class FPNGenerator(layers.Layer):
    def __init__(self, feature_size=256, **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        # 1x1 convs
        self.conv_c5 = layers.Conv2D(feature_size, 1)
        self.conv_c4 = layers.Conv2D(feature_size, 1)
        self.conv_c3 = layers.Conv2D(feature_size, 1)
        self.conv_c2 = layers.Conv2D(feature_size, 1)
        # 3x3 convs
        self.smooth_p5 = layers.Conv2D(feature_size, 3, padding="same")
        self.smooth_p4 = layers.Conv2D(feature_size, 3, padding="same")
        self.smooth_p3 = layers.Conv2D(feature_size, 3, padding="same")
        self.smooth_p2 = layers.Conv2D(feature_size, 3, padding="same")
        self.resize = layers.Lambda(lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method='nearest'))

    def call(self, inputs):
        c2, c3, c4, c5 = inputs
        # 1x1 conv reduce
        p5 = self.conv_c5(c5)
        p4 = self.conv_c4(c4)
        p3 = self.conv_c3(c3)
        p2 = self.conv_c2(c2)
        # top-down
        p4 = self.resize([p5, p4]) + p4
        p3 = self.resize([p4, p3]) + p3
        p2 = self.resize([p3, p2]) + p2
        # 3x3 conv smooth
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)

        return [p2, p3, p4, p5]
    