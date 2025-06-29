import tensorflow as tf

def loss_objectness_fn():
    pass

def loss_rpn_box_reg_fn():
    pass

def loss_class_fn(proposals, class_logits, gt_class_ids, gt_bboxes):
    """
    将 proposals 和 gt_bboxes 进行IoU计算，找到与每个 proposal 匹配的 ground truth，将与ground truth 匹配的 gt_class_id 赋值给该 proposal，并把 proposal 的 class_logits（预期值）和 gt_class_id（实际值）进行比较，计算损失。
    """
    pass

def loss_bbox_fn(proposals, bbox_deltas, gt_bboxes):
    pass

def loss_mask_fn(proposals, masks, gt_masks):
    pass
