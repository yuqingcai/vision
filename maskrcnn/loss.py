import tensorflow as tf

def class_loss_fn(proposals, class_logits, gt_class_ids, gt_bboxes):
    """
    将 proposals 和 gt_bboxes 进行IoU计算，找到与每个 proposal 匹配的 ground truth，将与ground truth 匹配的 gt_class_id 赋值给该 proposal，并把 proposal 的 class_logits（预期值）和 gt_class_id（实际值）进行比较，计算损失。
    """
    return 0.0

def bbox_loss_fn(proposals, bbox_deltas, gt_bboxes):
    return 0.0

def mask_loss_fn(proposals, masks, gt_masks):
    return 0.0
