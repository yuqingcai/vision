from tensorflow.keras import layers


class ROIMaskHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim=256, mask_size=28, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(
            hidden_dim, 
            3, 
            padding='same', 
            activation='relu'
        )
        self.conv2 = layers.Conv2D(
            hidden_dim, 
            3, 
            padding='same', 
            activation='relu'
        )
        self.conv3 = layers.Conv2D(
            hidden_dim, 
            3, 
            padding='same', 
            activation='relu'
        )
        self.conv4 = layers.Conv2D(
            hidden_dim, 
            3, 
            padding='same', 
            activation='relu'
        )
        self.deconv = layers.Conv2DTranspose(
            hidden_dim, 
            2, 
            strides=2, 
            activation='relu'
        )
        self.mask_pred = layers.Conv2D(
            num_classes, 
            1, 
            activation='sigmoid'
        )

        self.mask_size = mask_size

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.deconv(x)
        masks = self.mask_pred(x)
        return masks