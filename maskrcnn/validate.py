import tensorflow as tf
import time
import numpy as np
import json
from pycocotools import mask as mask_utils
from utils import decode_bbox
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib

def validate_image(
    id, 
    proposals, 
    valid_mask,
    classifier_logits,
    class_bbox_deltas,
    class_masks,
    score_thresh
):
    """
    Validate a single image.
    proposals shape: [M, 4]
    classifier_logits shape: [M, 81]
    class_bbox_deltas shape: [M, 324]
    class_masks shape: [M, 28, 28, 81]
    """
    proposals = tf.boolean_mask(proposals, valid_mask)
    classifier_logits = tf.boolean_mask(classifier_logits, valid_mask)
    class_bbox_deltas = tf.boolean_mask(class_bbox_deltas, valid_mask)
    class_masks = tf.boolean_mask(class_masks, valid_mask)
    
    # classifier_logits 的形状通常为 (M, num_classes)，M 为 proposal 数量，每一行是一
    # 个 proposal 对所有类别的原始得分（logits），每一行有81个数值。
    # tf.nn.softmax(classifier_logits, axis=-1) 会对每个 proposal 的 logits 进行
    # softmax 操作，使得每一行的分数加和为1，得到每个proposal属于每个类别的概率（预测置信度）。
    # 例如，classifier_logits[i] 经过softmax后得到 pred_scores[i]，代表第i个 proposal
    # 属于各类别的概率。pred_scores 形状为 (M, num_classes)。
    pred_scores = tf.nn.softmax(classifier_logits, axis=-1).numpy()
    
    # 对每个 proposal 选得分最高的类别（忽略背景类0）
    pred_classes = tf.argmax(pred_scores[:, 1:], axis=-1) + 1       # [M] 类别1~80
    pred_confidences = tf.reduce_max(pred_scores[:, 1:], axis=-1)   # [M]
    n = tf.reduce_sum(tf.cast(pred_confidences >= score_thresh, tf.int32)).numpy()

    mask_thresh = 0.5

    results = []
    for i in range(proposals.shape[0]):
        cls = pred_classes[i]
        score = pred_confidences[i]
        
        if score < score_thresh:
            continue 
        
        # x1, y1, x2, y2
        start = cls * 4
        end = (cls + 1) * 4
        bbox_delta = class_bbox_deltas[i, start:end]    # [4]
        proposal = proposals[i]                         # [4]
        bbox = decode_bbox(proposal, bbox_delta)   # [4]
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        # mask
        mask_pred = class_masks[i, :, :, cls]           # [28,28]
        mask_bin = tf.cast(mask_pred > mask_thresh, tf.uint8).numpy()
        rle = mask_utils.encode(np.asfortranarray(mask_bin))
        rle["counts"] = rle["counts"].decode("utf-8")

        results.append({
            "image_id": int(id.numpy()),
            "category_id": int(cls),
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(score),
            "segmentation": rle
        })
    
    return results


def validate(
        epoch,
        model,
        ann_file, 
        dataset,
        dataset_length,
        batch_size):
    
    epoch_score_thresh_map = {
        0 : 0.02,
        1 : 0.03,
        2 : 0.06,
        3 : 0.05,
        4 : 0.06,
        5 : 0.07,
        6 : 0.08,
        7 : 0.1,
        8 : 0.2,
        9 : 0.3,
        10: 0.4,
    }

    score_thresh = epoch_score_thresh_map.get(epoch, 0.5)
    
    results = []
    t_0 = time.time()

    steps = dataset_length // batch_size
    iterator = iter(dataset)

    for step in range(steps):
        t_1 = time.time()

        batch = iterator.get_next()
        ids = batch['id']
        images = batch['image']
        sizes = batch['size']

        # d2 = time.time()

        proposals, \
            valid_mask, \
            rpn_objectness_logits, \
            rpn_bbox_deltas, \
            classifier_logits, \
            class_bbox_deltas, \
            class_masks = model(images, sizes, training=False)
        
        # d2 = time.time() - d2
        # print(f'eval_t:{d2:.2f}s')

        for i in range(len(ids)):
            # d3 = time.time()

            results += validate_image(
                ids[i], 
                proposals[i], 
                valid_mask[i],
                classifier_logits[i],
                class_bbox_deltas[i],
                class_masks[i],
                score_thresh
            )
            # d3 = time.time() - d3
            # print(f'validate image {ids[i]}: {d3:.2f}s')

        d0 = time.time() - t_0
        d1 = time.time() - t_1
        
        print(
            f'validate step {step}, '
            f'setp_t: {d1:.2f}s, '
            f'total_t: {d0:.2f}s, '
            f'av_t: {d0/(step+1):.2f}s '
        )
    
    del iterator
    
    if not results:
        print("No valid proposals found, validation returning zero.")
        return

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(results)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")  # "bbox"或"segm"根据任务选择
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    with open('eval_results.txt', 'a') as f:
        f.write(f'Epoch {epoch} validation results:\n')
        with contextlib.redirect_stdout(f):
            cocoEval.summarize()
        f.write(f'\n\n')

