import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import json
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from model import MaskRCNN
from dataset import create_dataset


# tf.config.set_visible_devices([], 'GPU')

coco_root = "../../dataset/coco2017/"
train_img_dir = os.path.join(coco_root, "train2017")
ann_file = os.path.join(coco_root, "annotations/instances_train2017.json")


def resize_image(img, min_size=800, max_size=1333):
    h, w = img.shape[:2]
    scale = min(min_size / min(h, w), max_size / max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = tf.image.resize(img, (new_h, new_w), method='bilinear')
    return img_resized, scale


def padding_image(img, target_height=1400, target_width=1400):
    h, w = img.shape[:2]
    pad_h = target_height - h
    pad_w = target_width - w
    if pad_h < 0 or pad_w < 0:
        print(f'w: {w}, h: {h}')
        raise ValueError("Image is larger than target size.")
    img_padded = tf.image.pad_to_bounding_box(
        img, 0, 0, target_height, target_width)
    return img_padded


def expand_batch_axis(img):
    return tf.expand_dims(img, axis=0)


def image_tensor(file_name):
    path = f'../../dataset/coco2017/train2017/{file_name}'
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img_resized, scale = resize_image(img, min_size=800, max_size=1333)
    return img_resized, scale


if __name__ == "__main__":

    ann_path = '../../dataset/coco2017/annotations/instances_train2017.json'
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()

    batch = []
    max_width = 0
    max_height = 0

    n = len(image_ids)
    # indices = np.random.choice(n, size=10, replace=False)
    
    indices = range(0, 2)
    for i in indices:
        img_id = image_ids[i]
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        origin_height = img_info['height']
        origin_width = img_info['width']
        img, scale = image_tensor(file_name)
        target_height = int(round(origin_height * scale))
        target_width = int(round(origin_width * scale))

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns_origin = coco.loadAnns(ann_ids)

        anns = []
        for ann_origin in anns_origin:
            if ann_origin['iscrowd'] == 1:
                continue
            
            ann = ann_origin.copy()

            bbox = ann_origin['bbox']
            bbox_resized = [
                bbox[0] * scale,  # x
                bbox[1] * scale,  # y
                bbox[2] * scale,  # w
                bbox[3] * scale   # h
            ]
            ann['bbox'] = bbox_resized
            
            segmentation = ann_origin['segmentation']
            if isinstance(segmentation, list):
                # polygon format
                segmentation_resized = []
                for poly in segmentation:
                    poly_resized = [coord * scale for coord in poly]
                    segmentation_resized.append(poly_resized)
                rles = maskUtils.frPyObjects(
                     segmentation_resized, target_height, target_width)
                rle = maskUtils.merge(rles)
                ann['segmentation'] = rle
            elif isinstance(segmentation['counts'], list):
                # uncompressed RLE
                rle = maskUtils.frPyObjects(
                    segmentation, origin_height, origin_width)
                rle_resized = maskUtils.frPyObjects(
                    rle, target_height, target_width)
                ann['segmentation'] = rle_resized
            else:
                # compressed RLE
                rle = segmentation
                rle['counts'] = rle['counts'].decode('utf-8')
                rle_resized = maskUtils.frPyObjects(
                    rle, origin_height, origin_width)
                rle_resized = maskUtils.frPyObjects(
                    rle_resized, target_height, target_width)
                ann['segmentation'] = rle_resized

            anns.append(ann)

        batch.append({
            'image': img, 
            'scale': scale, 
            'image_id': img_id, 
            'file_name': file_name, 
            'origin_height': origin_height, 
            'origin_width': origin_width,
            'target_height': target_height,
            'target_width': target_width,
            'anns': anns
            })
        
        if target_height > max_height:
            max_height = target_height

        if target_width > max_width:
            max_width = target_width

    images = []
    for item in batch:
        img = item['image']
        img = padding_image(
            img, 
            target_height=max_height, 
            target_width=max_width
            )
        img = expand_batch_axis(img)
        item['image'] = img
        images.append(img)
    
    images = tf.concat(images, axis=0)
    print(f'images shape: {images.shape}')

    for item in batch:
        img = item['image']
        img_np = img[0].numpy()
        anns = item['anns']
    
        plt.figure()
        plt.title(f"Image ID: {item['image_id']}")
        plt.imshow(img_np)
        
        ax = plt.gca()
        ax.set_xlim([0, img_np.shape[1]])
        ax.set_ylim([img_np.shape[0], 0])

        for ann in anns:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=0.7, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            rle = ann['segmentation']
            mask = maskUtils.decode(rle)  # (H, W), 0/1
            # 只显示mask区域，透明度可调
            ax.imshow(np.ma.masked_where(mask == 0, mask), 
                      alpha=0.4, cmap='spring')
            
        plt.axis('on')

    plt.show()



    # # 选择一张图像 ID（比如第一张）
    # img_id = image_ids[1]
    # # 获取图像信息
    # img_info = coco.loadImgs(img_id)[0]
    # print("图像信息：", img_info)
    # # 获取该图像的所有标注 ID
    # ann_ids = coco.getAnnIds(imgIds=img_id)
    # # 加载所有标注
    # anns = coco.loadAnns(ann_ids)
    # # 遍历打印标注信息
    # for i, ann in enumerate(anns):
    #     print(f'ann{i}:')
    #     print("类别ID:", ann['category_id'])
    #     print("边框: ", ann['bbox'])  # 格式为 [x, y, width, height]
    #     print("分割: ", ann.get('segmentation'))  # 可能是 polygon 或 RLE
    #     print("是否是 crowd:", ann.get('iscrowd', 0))

    # # 假设 anns 是 COCO 的 annotation 列表，img_info 是图片信息
    # height, width = img_info['height'], img_info['width']
    # binary_mask = np.zeros((height, width), dtype=np.uint8)

    # for ann in anns:
    #     segm = ann['segmentation']
    #     if isinstance(segm, list):  # polygon
    #         rles = maskUtils.frPyObjects(segm, height, width)
    #         rle = maskUtils.merge(rles)
    #     elif isinstance(segm['counts'], list):  # uncompressed RLE
    #         rle = maskUtils.frPyObjects(segm, height, width)
    #     else:  # compressed RLE
    #         rle = segm
    #     m = maskUtils.decode(rle)
    #     binary_mask = np.logical_or(binary_mask, m)

    # plt.imshow(binary_mask, cmap='gray')
    # plt.axis('off')
    # plt.title('Binary Mask')
    # plt.show()
    
    
    # img_0, scale_0 = image_tensor('000000000049.jpg')
    # model = MaskRCNN()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    # model.summary()

    # # output: [logits, bbox_deltas, proposals]
    # output = model(img_0)
    # proposals = output[2]
    # print(proposals.shape)

    # img_np = img_0[0].numpy()  # 去掉 batch 维，转为 numpy
    # plt.imshow(img_np)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img_np)

    # ax = plt.gca()
    # for box in proposals.numpy():
    #     x1, y1, x2, y2 = box
    #     rect = patches.Rectangle(
    #         (x1, y1), x2 - x1, y2 - y1,
    #         linewidth=1, edgecolor='r', facecolor='none', alpha=0.4
    #     )
    #     ax.add_patch(rect)

    # plt.title(f"Proposals ({proposals.shape[0]} boxes)")
    # plt.axis('off')
    # plt.show()

def preprocess_batch(batch):
    
    file_paths = batch['file_path']
    bboxes_batch = batch['bboxes']
    classes_batch = batch['classes']

    # 1. 计算每张图片的缩放系数
    scales = []
    new_shapes = []
    for img in images:
        h, w = img.shape[:2]
        scale = min(min_size / h, max_size / w, 1.0)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        scales.append(scale)
        new_shapes.append((new_h, new_w))

    # 2. 计算 batch 内目标高宽
    target_h = max([s[0] for s in new_shapes])
    target_w = max([s[1] for s in new_shapes])

    # 3. resize+pad 所有图片
    images_out, bboxes_out, masks_out = [], [], []
    for img, bboxes, masks, scale in \
        zip(images, bboxes_list, masks_list, scales):
        img_resized = tf.image.resize(
            img, 
            (int(round(img.shape[0]*scale)), 
             int(round(img.shape[1]*scale))))
        img_padded = tf.image.pad_to_bounding_box(
            img_resized, 0, 0, target_h, target_w)
        images_out.append(img_padded)
        # bbox缩放
        bboxes_out.append(bboxes * scale)
        # mask缩放
        masks_resized = tf.image.resize(
            masks, 
            (int(round(img.shape[0]*scale)), 
             int(round(img.shape[1]*scale))), 
             method='nearest')
        masks_padded = tf.image.pad_to_bounding_box(
            masks_resized, 0, 0, target_h, target_w)
        masks_out.append(masks_padded)
    # 4. 堆叠成 batch
    images_out = tf.stack(images_out)
    bboxes_out = tf.stack(bboxes_out)
    masks_out = tf.stack(masks_out)
    return images_out, bboxes_out, masks_out




    # print(f'walk through batches in dataset...')
    # time_0 = time.time()
    # for i, data in enumerate(ds_train):
    #     if i == 0:
    #         for j in range(0, len(data['image'])):
    #             img = data['image'][j]
    #             size = data['size'][j]
    #             bboxes = data['bboxes'][j]
    #             category_ids = data['category_ids'][j]
    #             # segmentations = data['segmentations'][j]
    #             plt.figure()
    #             plt.imshow(img.numpy())
    #             plt.axis('on')

    #             # size border
    #             rect = patches.Rectangle((0, 0), size[1], size[0], 
    #                                      linewidth=1, \
    #                                      edgecolor='red', \
    #                                      facecolor='none', \
    #                                      alpha=1.0
    #             )
    #             plt.gca().add_patch(rect)
                
    #             for bbox, category_id in zip(bboxes, category_ids):
    #                 x1, y1, x2, y2 = bbox
    #                 rect = patches.Rectangle((x1, y1), x2 - x1, \
    #                                          y2 - y1, linewidth=1, \
    #                                          edgecolor='r', \
    #                                          facecolor='none', \
    #                                          alpha=0.4
    #                 )
    #                 plt.gca().add_patch(rect)

    #                 cid = int(category_id.numpy()) if \
    #                     hasattr(category_id, 'numpy') else int(category_id)
    #                 plt.gca().text(x1, y1 - 2, str(cid), \
    #                                color='yellow', fontsize=6, \
    #                                va='bottom',  ha='left', \
    #                                bbox=dict(facecolor='black', 
    #                                          alpha=0.5, pad=0))
    #         plt.show()
    #         print(f'first batch image: {data["images"][0].numpy()}')

    # print(f'batches: {i}, duration: {time.time() - time_0:.2f} s')

        # print(f'file_path:')
        # for image_path in data['file_path']:
        #     print(f'{image_path.numpy().decode()}')

        # print(f'category_ids:')
        # for category_id in data['category_ids']:
        #     print(f'{category_id.numpy().tolist()}')

        # print(f'bboxes:')
        # for bbox in data['bboxes']:
        #     print(f'{bbox.numpy().tolist()}')


        # print(f'segmentations:')
        # for segmentation in data['segmentations']:
        #     print(f'{segmentation.numpy().tolist()}')

        # print(f'')

    # # 选择一张图像 ID（比如第一张）
    # img_id = image_ids[1]
    # # 获取图像信息
    # img_info = coco.loadImgs(img_id)[0]
    # print('图像信息：', img_info)
    # # 获取该图像的所有标注 ID
    # ann_ids = coco.getAnnIds(imgIds=img_id)
    # # 加载所有标注
    # anns = coco.loadAnns(ann_ids)
    # # 遍历打印标注信息
    # for i, ann in enumerate(anns):
    #     print(f'ann{i}:')
    #     print('类别ID:', ann['category_id'])
    #     print('边框: ', ann['bbox'])  # 格式为 [x, y, width, height]
    #     print('分割: ', ann.get('segmentation'))  # 可能是 polygon 或 RLE
    #     print('是否是 crowd:', ann.get('iscrowd', 0))

    # # 假设 anns 是 COCO 的 annotation 列表，img_info 是图片信息
    # height, width = img_info['height'], img_info['width']
    # binary_mask = np.zeros((height, width), dtype=np.uint8)

    # for ann in anns:
    #     segm = ann['segmentation']
    #     if isinstance(segm, list):  # polygon
    #         rles = maskUtils.frPyObjects(segm, height, width)
    #         rle = maskUtils.merge(rles)
    #     elif isinstance(segm['counts'], list):  # uncompressed RLE
    #         rle = maskUtils.frPyObjects(segm, height, width)
    #     else:  # compressed RLE
    #         rle = segm
    #     m = maskUtils.decode(rle)
    #     binary_mask = np.logical_or(binary_mask, m)

    # plt.imshow(binary_mask, cmap='gray')
    # plt.axis('off')
    # plt.title('Binary Mask')
    # plt.show()
    
    
    # img_0, scale_0 = image_tensor('000000000049.jpg')
    # model = MaskRCNN()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    # model.summary()

    # # output: [logits, bbox_deltas, proposals]
    # output = model(img_0)
    # proposals = output[2]
    # print(proposals.shape)

    # img_np = img_0[0].numpy()  # 去掉 batch 维，转为 numpy
    # plt.imshow(img_np)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img_np)

    # ax = plt.gca()
    # for box in proposals.numpy():
    #     x1, y1, x2, y2 = box
    #     rect = patches.Rectangle(
    #         (x1, y1), x2 - x1, y2 - y1,
    #         linewidth=1, edgecolor='r', facecolor='none', alpha=0.4
    #     )
    #     ax.add_patch(rect)

    # plt.title(f'Proposals ({proposals.shape[0]} boxes)')
    # plt.axis('off')
    # plt.show()


def resize_rle_masks(rel_masks, scale):

    def decode_and_resize(rle_str, scale):

        def py_fn(rle_str_np, scale_np):
            rle = json.loads(rle_str_np.decode('utf-8'))
            mask = maskUtils.decode(rle)  # shape=(H, W), dtype=uint8

            h, w = mask.shape
            new_h = int(h * scale_np)
            new_w = int(w * scale_np)

            resized_mask = cv2.resize(mask, (new_w, new_h), \
                                      interpolation=cv2.INTER_NEAREST)

            rle_new = maskUtils.encode(np.asfortranarray(resized_mask))
            rle_new['size'] = [new_h, new_w]

            if isinstance(rle_new['counts'], bytes):
                rle_new['counts'] = rle_new['counts'].decode('ascii')

            return json.dumps(rle_new).encode('utf-8')

        return tf.py_function(
            func=py_fn, inp=[rle_str, scale], Tout=tf.string
        )
    
    resized_rel_masks = tf.map_fn(
        lambda args: decode_and_resize(args[0], args[1]),
        (rel_masks, scale),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )

    return resized_rel_masks



def rle_mask_from_seg(seg, height, width):
    # Check if segmentation is a polygon or RLE, 
    # then make compress
    if isinstance(seg, list):
        # polygon
        rles = maskUtils.frPyObjects(seg, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(seg['counts'], list):
        # uncompressed
        rle = maskUtils.frPyObjects(seg, height, width)
        rle = maskUtils.merge(rle)
    else:
        # compressed
        rle = seg
    if isinstance(rle, list):
        rle = rle[0]
    
    # Ensure counts is a string
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    
    rle_str = json.dumps(rle)

    return rle_str



    shape = tf.shape(sample['image'][0])
    for i in range(1, batch_size):
        img = sample['image'][i]
        print(f'shape of image {i}: {tf.shape(img)}')

        if tf.reduce_any(tf.not_equal(tf.shape(img), shape)):
            print(f"Image {i} has different shape: {tf.shape(img)}")
            break
        shape = tf.shape(img)



def show_image(sample, i):
    img = sample['image'][i]
    size = sample['size'][i]
    bboxes = sample['bboxes'][i]
    category_ids = sample['category_ids'][i]
    masks = sample['masks'][i]
    padded_size = tf.shape(img)

    print(f'padded_size: {padded_size[0]}x{padded_size[1]}')

    plt.figure()
    plt.imshow(img.numpy())
    plt.axis('off')
    plt.gca().set_aspect('auto')

    # size border
    rect = patches.Rectangle((0, 0), size[1], size[0], 
                             linewidth=1, 
                             edgecolor='green',
                             facecolor='none',
                             alpha=1.0
    )
    plt.gca().add_patch(rect)
    
    for bbox, category_id, mask in \
        zip(bboxes, category_ids, masks):

        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, 
                                 y2 - y1, linewidth=1,
                                 edgecolor='r', 
                                 facecolor='none', 
                                 alpha=0.4
        )
        plt.gca().add_patch(rect)

        cid = int(category_id.numpy()) if \
            hasattr(category_id, 'numpy') else int(category_id)
        plt.gca().text(x1, y1 - 2, str(cid), 
                        color='yellow', fontsize=6, 
                        va='bottom', ha='left', 
                        bbox=dict(facecolor='black', alpha=0.5, pad=0))
        
        mask = mask.numpy() if hasattr(mask, 'numpy') else mask
        highlight = np.zeros((*mask.shape, 4), dtype=np.float32)
        highlight[mask == 1] = [1, 0, 0, 0.5]
        plt.imshow(highlight)


from tensorflow import keras
from tensorflow.keras import Model


    


def build_resnet50(
        input_shape, 
        batch_size, 
        weights="imagenet"):
    
    inputs = keras.Input(shape=input_shape, batch_size=batch_size)
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
        inputs=inputs, 
        outputs=outputs,
        name='resnet50')
    
    return model


def build_resnet101(
        input_shape, 
        batch_size, 
        weights="imagenet"):
    
    inputs = keras.Input(shape=input_shape, batch_size=batch_size)
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
        inputs=inputs, 
        outputs=outputs,
        name='resnet101')
    
    return model




        # anchors_all = []
        # for feature_map, stride, base_size in \
        #     zip(feature_maps, strides, base_sizes):

        #     shape = tf.shape(feature_map)
        #     H = tf.cast(shape[1], tf.int32)
        #     W = tf.cast(shape[2], tf.int32)

        #     # 生成所有 ratio/scale 组合
        #     base_size = tf.cast(base_size, tf.float32) 
        #     ratios = tf.reshape(self.ratios, [-1, 1])
        #     scales = tf.reshape(self.scales, [1, -1])
        #     ws = tf.sqrt(base_size * base_size * scales * scales / ratios)
        #     hs = ws * ratios
        #     ws = tf.reshape(ws, [-1])
        #     hs = tf.reshape(hs, [-1])

        #     # [A, 4]
        #     x1 = -ws / 2
        #     y1 = -hs / 2
        #     x2 = ws / 2
        #     y2 = hs / 2
        #     base_anchors = tf.stack([x1, y1, x2, y2], axis=1)

        #     # 平移到所有位置
        #     # anchor 的数量只与特征图的空间尺寸（高 H，宽 W）和每个位置的 
        #     # base_anchors 数量有关，与特征图的深度（通道数）无关。
        #     # 每层 anchor 数量的计算公式为：
        #     # base_anchor_num_per_dot = len(ratios) * len(scales)
        #     # anchor_num_per_layer = H × W × base_anchor_num_per_dot
        #     shift_x = (tf.range(W, dtype=tf.float32) + 0.5) * stride
        #     shift_y = (tf.range(H, dtype=tf.float32) + 0.5) * stride
        #     shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        #     shifts = tf.stack([
        #         tf.reshape(shift_x, [-1]),
        #         tf.reshape(shift_y, [-1]),
        #         tf.reshape(shift_x, [-1]),
        #         tf.reshape(shift_y, [-1])
        #     ], axis=1)  # [K, 4]
        #     A = tf.shape(base_anchors)[0]
        #     K = tf.shape(shifts)[0]
        #     anchors = tf.reshape(base_anchors, [1, A, 4]) + \
        #         tf.reshape(shifts, [K, 1, 4])
        #     anchors = tf.reshape(anchors, [K * A, 4])
        #     anchors_all.append(anchors)
        # # 合并所有层的 anchors

        # return tf.concat(anchors_all, axis=0)

base_size: 32 
ws: [
    [45.2548332 67.8822479 90.5096664]
    [32 48 64]
    [22.6274166 33.941124 45.2548332]
] 
hs: [
    [22.6274166 33.941124 45.2548332]
    [32 48 64]
    [45.2548332 67.8822479 90.5096664]
] 

wsxhs: [
    [1023.99994 2303.99976 4095.99976]
    [1024 2304 4096]
    [1023.99994 2303.99976 4095.99976]
]


base size: 32 area: [[1024 2304 4096]]

ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]
base size: 32 area: [[1024 2304 4096]]
ws: [[64 96 128]
 [32 48 64]
 [16 24 32]]
hs: [[32 48 64]
 [32 48 64]
 [32 48 64]]