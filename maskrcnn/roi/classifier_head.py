from tensorflow.keras import layers


class ROIClassifierHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.class_logits = layers.Dense(
            num_classes, 
            activation='softmax'
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True
    
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        class_logits = self.class_logits(x)
        return class_logits
    