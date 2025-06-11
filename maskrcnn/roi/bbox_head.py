from tensorflow.keras import layers


class ROIBBoxHead(layers.Layer):
    def __init__(self, num_classes, hidden_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        # 每个类别4个回归参数
        self.bbox_pred = layers.Dense(num_classes * 4)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas
    