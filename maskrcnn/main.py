import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import json
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# tf.config.set_visible_devices([], 'GPU')

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

    return [p2, p3, p4, p5]

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
    
"""
生成 anchors
anchors（锚框）：是按照特定规则（位置、尺度、比例）在特征图上密集生成的一组“参考框”，
它们和图片内容无关，只和特征图的空间位置、ratios、scales 有关。anchors 数量通常非
常多（几十万）。

proposals（候选框）：是通过 RPN Head 对 anchors 进行前景/背景分类和边框回归后，
经过解码、筛选（如 NMS、top-N）得到的一组“高质量候选区域”。proposals 数量远小于
anchors（通常几百到几千），用于后续 ROI Head 的
精细分类和回归。
"""
class AnchorGenerator(layers.Layer):
    def __init__(self, ratios=[0.5, 1, 2], scales=[8, 16, 32], **kwargs):
        super().__init__(**kwargs)
        self.ratios = tf.constant(ratios, dtype=tf.float32)
        self.scales = tf.constant(scales, dtype=tf.float32)

    def call(self, strides, base_sizes, feature_maps):
        anchors_all = []
        for feature_map, stride, base_size in zip(feature_maps, strides, base_sizes):
            shape = tf.shape(feature_map)
            H = tf.cast(shape[1], tf.int32)
            W = tf.cast(shape[2], tf.int32)

            # 生成所有 ratio/scale 组合
            base_size = tf.cast(base_size, tf.float32) 
            ratios = tf.reshape(self.ratios, [-1, 1])
            scales = tf.reshape(self.scales, [1, -1])
            ws = tf.sqrt(base_size * base_size * scales * scales / ratios)
            hs = ws * ratios
            ws = tf.reshape(ws, [-1])
            hs = tf.reshape(hs, [-1])

            # [A, 4]
            x1 = -ws / 2
            y1 = -hs / 2
            x2 = ws / 2
            y2 = hs / 2
            base_anchors = tf.stack([x1, y1, x2, y2], axis=1)

            # 平移到所有位置
            # anchor 的数量只与特征图的空间尺寸（高 H，宽 W）和每个位置的 
            # base_anchors 数量有关，与特征图的深度（通道数）无关。
            # 每层 anchor 数量的计算公式为：
            # base_anchor_num_per_dot = len(ratios) * len(scales)
            # anchor_num_per_layer = H × W × base_anchor_num_per_dot
            shift_x = (tf.range(W, dtype=tf.float32) + 0.5) * stride
            shift_y = (tf.range(H, dtype=tf.float32) + 0.5) * stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
            shifts = tf.stack([
                tf.reshape(shift_x, [-1]),
                tf.reshape(shift_y, [-1]),
                tf.reshape(shift_x, [-1]),
                tf.reshape(shift_y, [-1])
            ], axis=1)  # [K, 4]
            A = tf.shape(base_anchors)[0]
            K = tf.shape(shifts)[0]
            anchors = tf.reshape(base_anchors, [1, A, 4]) + \
                tf.reshape(shifts, [K, 1, 4])
            anchors = tf.reshape(anchors, [K * A, 4])
            anchors_all.append(anchors)
        # 合并所有层的 anchors

        return tf.concat(anchors_all, axis=0)


class RPNHead(layers.Layer):
    def __init__(self, num_anchors=9, feature_size=256, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(feature_size, 3, padding="same", 
                                  activation="relu")
        # 前景/背景分数 (batch, H, W, num_anchors)
        self.cls_logits = layers.Conv2D(num_anchors, 1)
        # 边框回归 (batch, H, W, num_anchors*4)
        self.bbox_deltas = layers.Conv2D(num_anchors * 4, 1)

    def call(self, feature_maps):
        logits_all = []
        bbox_deltas_all = []
        for feature_map in feature_maps:
            x = self.conv(feature_map)
            logits = self.cls_logits(x)
            bbox_deltas = self.bbox_deltas(x)
            # reshape
            logits = tf.reshape(logits, [tf.shape(logits)[0], -1, 1])   # (batch, HW*A, 1)
            bbox_deltas = tf.reshape(bbox_deltas, [tf.shape(bbox_deltas)[0], -1, 4])  # (batch, HW*A, 4)
            logits_all.append(logits)
            bbox_deltas_all.append(bbox_deltas)

        logits_all = tf.concat(logits_all, axis=1)         # (batch, N, 1)
        bbox_deltas_all = tf.concat(bbox_deltas_all, axis=1)  # (batch, N, 4)
        
        return logits_all, bbox_deltas_all


class ProposalGenerator(layers.Layer):
    def __init__(self, pre_nms_topk=6000, post_nms_topk=1000, 
                 nms_thresh=0.7, min_size=16, **kwargs):
        super().__init__(**kwargs)
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_size = min_size

    def call(self, inputs):

        image, anchors, bbox_deltas, logits = inputs

        # 1. 解码 proposals
        wa = anchors[:, 2] - anchors[:, 0]
        ha = anchors[:, 3] - anchors[:, 1]
        xa = anchors[:, 0] + 0.5 * wa
        ya = anchors[:, 1] + 0.5 * ha

        dx = bbox_deltas[:, 0]
        dy = bbox_deltas[:, 1]
        dw = bbox_deltas[:, 2]
        dh = bbox_deltas[:, 3]

        x = dx * wa + xa
        y = dy * ha + ya
        w = tf.exp(dw) * wa
        h = tf.exp(dh) * ha

        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        x2 = x + 0.5 * w
        y2 = y + 0.5 * h
        proposals = tf.stack([x1, y1, x2, y2], axis=1)

        # 2. 裁剪到图片边界
        shape = tf.shape(image)
        height = tf.cast(shape[1], tf.float32)
        width = tf.cast(shape[2], tf.float32)
        
        proposals = tf.stack([
            tf.clip_by_value(proposals[:, 0], 0, width - 1),
            tf.clip_by_value(proposals[:, 1], 0, height - 1),
            tf.clip_by_value(proposals[:, 2], 0, width - 1),
            tf.clip_by_value(proposals[:, 3], 0, height - 1)
        ], axis=1)

        # 3. 去除过小的框
        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        valid = tf.where((ws >= self.min_size) & (hs >= self.min_size))
        proposals = tf.gather(proposals, valid[:, 0])
        scores = tf.gather(tf.sigmoid(logits[:, 0]), valid[:, 0])

        # 4. 取 top-k
        topk = tf.math.top_k(scores, k=tf.minimum(self.pre_nms_topk, tf.shape(scores)[0]))
        proposals = tf.gather(proposals, topk.indices)
        scores = tf.gather(scores, topk.indices)

        # 5. NMS
        keep = tf.image.non_max_suppression(
            proposals, scores,
            max_output_size=self.post_nms_topk,
            iou_threshold=self.nms_thresh
        )
        proposals = tf.gather(proposals, keep)

        return proposals


def build_resnet_fpn_rpn(
        input_shape=(None, None, 3), 
        batch_size=1, 
        backbone_type='resnet50'):
    
    if backbone_type == 'resnet101':
        backbone = build_resnet101(input_shape, batch_size)
    elif backbone_type == 'resnet50':
        backbone = build_resnet50(input_shape, batch_size)
    else:
        raise ValueError("backbone_type must be 'resnet50' or 'resnet101'")
    
    c2, c3, c4, c5 = backbone.output
    p2, p3, p4, p5 = FPNGenerator(feature_size=256)([c2, c3, c4, c5])
    anchors = AnchorGenerator(ratios=[0.5, 1, 2], scales=[8, 16, 32])(
        strides=[4, 8, 16, 32],
        base_sizes=[4, 8, 16, 32],
        feature_maps=[p2, p3, p4, p5],
    )

    logits, bbox_deltas = RPNHead(num_anchors=9, feature_size=256)(
        feature_maps=[p2, p3, p4, p5],
    )

    proposals = ProposalGenerator(
        pre_nms_topk=6000,
        post_nms_topk=1000,
        nms_thresh=0.7,
        min_size=1
    )([backbone.input[0], anchors, bbox_deltas[0], logits[0]])

    return Model(
        inputs=backbone.input,
        outputs=proposals,
        name=f'{backbone_type}_fpn_rpn'
    )


def resize_image(img, min_size=800, max_size=1333):
    h, w = img.shape[:2]
    scale = min(min_size / min(h, w), max_size / max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = tf.image.resize(img, (new_h, new_w), method='bilinear')
    return img_resized, scale


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


def image_tensor(file_name):
    path = f'../../dataset/coco2017/train2017/{file_name}'
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img_resized, scale = resize_image(img, min_size=800, max_size=1333)
    img_expand = expand_batch_axis(img_resized)
    return img_expand, scale


if __name__ == "__main__":
    # filename_to_id, id_to_anns = build_coco_ann_index(
    #     '../../dataset/coco/annotations/instances_train2017.json')

    img_0, scale_0 = image_tensor('000000000049.jpg')

    # print('output from resnet50:')
    # outputs = resnet50(img_0)
    # for i, output in enumerate(outputs):
    #     print(output.shape)

    # print('output from resnet50_fpn:')
    # outputs = resnet50_fpn(img_0)
    # for i, output in enumerate(outputs):
    #     print(output.shape)

    # for i, feature in enumerate(outputs):
    #     # 取第一个batch和第0通道
    #     for j in range(5):
    #         fmap = feature[0, :, :, j].numpy()
    #         plt.figure()
    #         plt.imshow(fmap, cmap='viridis')
    #         plt.title(f'FPN P{i+2} Feature Map (channel {j})')
    #         plt.colorbar()
    # plt.show()

    model = build_resnet_fpn_rpn()
    model.summary()
    output = model(img_0)
    print(output.shape)
    # for i, feature in enumerate(output):
    #     print(f'{i}:{feature}')

    # 输出各层特征的 shape
    # for i, output in enumerate(outputs):
    #     print(output.shape)

    # start_time = time.time()
    # for i in range(100):
    #     print(f'eval:{i}')
    #     outputs = resnet50_fpn(img_0)
    # end_time = time.time()
    # print(f"elapsed time: {end_time - start_time:.4f} s")
    
    # model.export('model_export')
    # img_id = filename_to_id[img_name]
    # anns = id_to_anns[img_id]
    # print(anns)

    # plt.imshow(img_0[0].numpy())
    # plt.title("img_0")
    # plt.show()

    img_np = img_0[0].numpy()  # 去掉 batch 维，转为 numpy
    plt.figure(figsize=(12, 12))
    plt.imshow(img_np)
    ax = plt.gca()

    for box in output.numpy():
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor='r', facecolor='none', alpha=0.4
        )
        ax.add_patch(rect)

    plt.title(f"Proposals ({output.shape[0]} boxes)")
    plt.axis('off')
    plt.show()