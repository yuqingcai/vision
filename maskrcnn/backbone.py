import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def build_resnet101(
        input_shape=(None, None, 3), 
        barch_size=1, 
        weights="imagenet"):
    
    inputs = keras.Input(shape=input_shape, batch_size=barch_size)
    base_model = keras.applications.ResNet101(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        pooling=None
    )
    layer_names = [
        "conv2_block3_out",
        "conv3_block4_out", 
        "conv4_block23_out",
        "conv5_block3_out",
    ]
    # get C2, C3, C4, C5 layer outputs
    outputs = [base_model.get_layer(name).output for name in layer_names]
    model = Model(
        inputs=base_model.input, 
        outputs=outputs,
        name='resnet101')
    return model


def build_resnet50(
        input_shape=(None, None, 3), 
        barch_size=1, 
        weights="imagenet"):
    
    inputs = keras.Input(shape=input_shape, batch_size=barch_size)
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        pooling=None
    )
    layer_names = [
        "conv2_block3_out",
        "conv3_block4_out", 
        "conv4_block6_out",
        "conv5_block3_out",
    ]
    # get C2, C3, C4, C5 layer outputs
    outputs = [base_model.get_layer(name).output for name in layer_names]
    model = Model(
        inputs=base_model.input, 
        outputs=outputs,
        name='resnet50')
    return model