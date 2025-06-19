from tensorflow import keras
from tensorflow.keras import layers, Model


class ResNet50Backbone(Model):
    def __init__(self, 
                 input_shape, 
                 batch_size, 
                 trainable=True,
                 weights="imagenet", 
                 **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.Input(shape=input_shape, 
                                       batch_size=batch_size,
                                       dtype='float32')
        self.base_model = keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_tensor=self.input_layer,
            pooling=None
        )

        self.out_layers = [
            self.base_model.get_layer('conv2_block3_out').output,  # C2
            self.base_model.get_layer('conv3_block4_out').output,  # C3
            self.base_model.get_layer('conv4_block6_out').output,  # C4
            self.base_model.get_layer('conv5_block3_out').output   # C5
        ]

        self.feature_extractor = keras.Model(
            inputs=self.base_model.input,
            outputs=self.out_layers,
            name='resnet50_backbone'
        )
        self.feature_extractor.trainable = trainable


    def call(self, inputs, training=False):
        return self.feature_extractor(inputs, training=training)
    

class ResNet101Backbone(Model):
    def __init__(self, 
                 input_shape, 
                 batch_size, 
                 trainable=True,
                 weights="imagenet", 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_layer = keras.Input(shape=input_shape, 
                                       batch_size=batch_size)
        
        self.base_model = keras.applications.ResNet101(
            include_top=False,
            weights=weights,
            input_tensor=self.input_layer,
            pooling=None
        )

        self.out_layers = [
            self.base_model.get_layer('conv2_block3_out').output,  # C2
            self.base_model.get_layer('conv3_block4_out').output,  # C3
            self.base_model.get_layer('conv4_block23_out').output, # C4
            self.base_model.get_layer('conv5_block3_out').output   # C5
        ]
        

        self.feature_extractor = keras.Model(
            inputs=self.base_model.input,
            outputs=self.out_layers,
            name='resnet101_backbone'
        )
        self.feature_extractor.trainable = trainable


    def call(self, inputs, training=False):        
        return self.backbone(inputs, training=training)
    