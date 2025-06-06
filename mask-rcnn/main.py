import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import json
from collections import defaultdict
import time
import matplotlib.pyplot as plt


def build_resnet101(input_shape=(None, None, 3), weights="imagenet"):
    inputs = keras.Input(shape=input_shape, batch_size=None)
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
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model


def build_fpn(c2, c3, c4, c5, feature_size=256):

    class ResizeLike(layers.Layer):
        def call(self, x, y):
            # x: to be upsampled, y: target shape
            target_shape = tf.shape(y)[1:3]
            return tf.image.resize(x, size=target_shape, method='nearest')
    
    # 1x1 conv to reduce channel dims
    p5 = layers.Conv2D(feature_size, 1)(c5)
    p4 = layers.Conv2D(feature_size, 1)(c4)
    p3 = layers.Conv2D(feature_size, 1)(c3)
    p2 = layers.Conv2D(feature_size, 1)(c2)

    resize_like = ResizeLike()

    def upsample_add(x, y):
        x = resize_like(x, y)
        return layers.Add()([x, y])
    
    # Top-down pathway
    p4 = upsample_add(p5, p4)
    p3 = upsample_add(p4, p3)
    p2 = upsample_add(p3, p2)
    
    # 3x3 conv to smooth
    p5 = layers.Conv2D(feature_size, 3, padding="same")(p5)
    p4 = layers.Conv2D(feature_size, 3, padding="same")(p4)
    p3 = layers.Conv2D(feature_size, 3, padding="same")(p3)
    p2 = layers.Conv2D(feature_size, 3, padding="same")(p2)

    # output: P2, P3, P4, P5
    return [p2, p3, p4, p5]


class RPN(layers.Layer):
    def __init__(self, anchors_per_location=3, feature_size=256, **kwargs):
        super().__init__(**kwargs)
        self.shared_conv = layers.Conv2D(feature_size, 3, 
                padding="same", activation="relu")
        self.objectness_conv = layers.Conv2D(anchors_per_location, 1)
        self.box_conv = layers.Conv2D(anchors_per_location * 4, 1)

    def call(self, feature_maps):
        # feature_maps: [P2, P3, P4, P5]
        t2 = self.shared_conv(feature_maps[0])
        t3 = self.shared_conv(feature_maps[1])
        t4 = self.shared_conv(feature_maps[2])
        t5 = self.shared_conv(feature_maps[3])

        obj2 = self.objectness_conv(t2)
        obj3 = self.objectness_conv(t3)
        obj4 = self.objectness_conv(t4)
        obj5 = self.objectness_conv(t5)

        box2 = self.box_conv(t2)
        box3 = self.box_conv(t3)
        box4 = self.box_conv(t4)
        box5 = self.box_conv(t5)

        return [obj2, obj3, obj4, obj5], [box2, box3, box4, box5]


def build_model(input_shape=(None, None, 3), anchors_per_location=3, 
                    feature_size=256, roi_pool_size=7):
    backbone = build_resnet101(input_shape)
    c2, c3, c4, c5 = backbone.output
    fpn_outputs = build_fpn(c2, c3, c4, c5)
    rpn = RPN(anchors_per_location=anchors_per_location, 
              feature_size=feature_size)
    rpn_objectness, rpn_bbox = rpn(fpn_outputs)

    return Model(
        inputs=backbone.input,
        outputs=rpn_objectness + rpn_bbox,
        name="fpn_rpn_resnet101"
    )


def resize_image(img, min_size=800, max_size=1333):
    h, w = img.shape[:2]
    scale = min(min_size / h, max_size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = tf.image.resize(img, (new_h, new_w), method='bilinear')

    # 如果缩放后尺寸大于目标，直接中心裁剪
    if new_h > min_size or new_w > max_size:
        img_cropped = tf.image.resize_with_crop_or_pad(img_resized, min_size, max_size)
        return img_cropped, scale

    # 否则padding
    pad_h = min_size - new_h
    pad_w = max_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_padded = tf.pad(
        img_resized,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        mode='CONSTANT',
        constant_values=0
    )
    return img_padded, scale


def build_coco_ann_index(ann_path):
    f = open(ann_path, "r")
    coco_anns = json.load(f)
    f.close()

    filename_to_id = {img["file_name"]: img["id"] for img in coco_anns["images"]}
    id_to_anns = defaultdict(list)
    for ann in coco_anns["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)
    return filename_to_id, id_to_anns


def expand_batch_axis(img):
    return tf.expand_dims(img, axis=0)


if __name__ == "__main__":
    # filename_to_id, id_to_anns = build_coco_ann_index(
    #     '../../dataset/coco/annotations/instances_train2017.json')

    model = build_model(input_shape=(None, None, 3))
    model.summary()

    img_name = 'input1.jpg'
    img_path = f'../../dataset/coco/train2017/{img_name}'
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img_resized, scale = resize_image(img, min_size=800, max_size=1333)
    plt.imshow(img_resized.numpy())
    plt.title("img_resized")
    plt.show()

    img_input = expand_batch_axis(img_resized)

    start_time = time.time()
    outputs = model(img_input)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.4f} s")
    
    # 输出各层特征的 shape
    for i, output in enumerate(outputs):
        print(output.shape)

    model.export('model_export')
    # img_id = filename_to_id[img_name]
    # anns = id_to_anns[img_id]
    # print(anns)