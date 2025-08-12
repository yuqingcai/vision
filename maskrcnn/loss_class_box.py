import tensorflow as tf
from utils import compute_iou

def loss_class_box_reg_fn(
        proposals, 
        valid_mask, 
        bbox_deltas_pred, 
        gt_labels,
        gt_bboxes,
        iou_pos_thresh=0.7,
        iou_neg_thresh=0.3):
    
    def loss_per_image(
            proposals, 
            valid_mask, 
            bbox_deltas_pred, 
            gt_labels,
            gt_bboxes):
        
        # bbox_deltas_pred shape: [N, num_classes * 4]

        proposals_valid = tf.boolean_mask(proposals, valid_mask)
        bbox_deltas_pred_valid = tf.boolean_mask(bbox_deltas_pred, valid_mask)
        
        gt_bboxes = gt_bboxes.to_tensor()
        
        # Find the best matching ground truth for each proposal
        ious = compute_iou(proposals_valid, gt_bboxes)
        best_gt_ious = tf.reduce_max(ious, axis=1)
        best_gt_inds = tf.argmax(ious, axis=1, output_type=tf.int32)

        pos_mask = best_gt_ious >= iou_pos_thresh

        proposals_pos = tf.boolean_mask(proposals_valid, pos_mask)
        if tf.size(proposals_pos) == 0:
            # tf.print('No positive proposals found, class_box loss returning zero.')
            return tf.constant(0.0, dtype=tf.float32)
        
        bbox_deltas_pred_pos = tf.boolean_mask(
            bbox_deltas_pred_valid, 
            pos_mask
        )

        best_gt_inds_pos = tf.boolean_mask(best_gt_inds, pos_mask)  # [M]
        gt_boxes_pos = tf.gather(gt_bboxes, best_gt_inds_pos)       # [M, 4]
        gt_labels_pos = tf.gather(gt_labels, best_gt_inds_pos)
        
        # compute the target deltas
        px, py, pw, ph = \
            (proposals_pos[:, 0] + proposals_pos[:, 2]) / 2, \
            (proposals_pos[:, 1] + proposals_pos[:, 3]) / 2, \
            proposals_pos[:, 2] - proposals_pos[:, 0], \
            proposals_pos[:, 3] - proposals_pos[:, 1]
        
        gx, gy, gw, gh = \
            (gt_boxes_pos[:, 0] + gt_boxes_pos[:, 2]) / 2, \
            (gt_boxes_pos[:, 1] + gt_boxes_pos[:, 3]) / 2, \
            gt_boxes_pos[:, 2] - gt_boxes_pos[:, 0], \
            gt_boxes_pos[:, 3] - gt_boxes_pos[:, 1]
        
        tx = (gx - px) / tf.maximum(pw, 1e-6)
        ty = (gy - py) / tf.maximum(ph, 1e-6)
        tw = tf.math.log(gw / tf.maximum(pw, 1e-6))
        th = tf.math.log(gh / tf.maximum(ph, 1e-6))

        target_deltas = tf.stack([tx, ty, tw, th], axis=1)
        
        num_classes = tf.shape(bbox_deltas_pred_pos)[1] // 4
        bbox_deltas_pred_pos_reshaped = tf.reshape(
            bbox_deltas_pred_pos, 
            [ -1, num_classes, 4 ]
        ) # [M, C, 4]

        # 使用 batch gather: 对每个样本从对应类别维度提取 delta
        pred_deltas = tf.gather(
            bbox_deltas_pred_pos_reshaped, 
            gt_labels_pos, 
            axis=1, 
            batch_dims=1)  # [M, 4]
        
        diff = pred_deltas - target_deltas
        abs_diff = tf.abs(diff)
        loss = tf.where(abs_diff < 1.0,
                        0.5 * tf.square(abs_diff),
                        abs_diff - 0.5)
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    
    losses = tf.map_fn(
        lambda args: loss_per_image(
            args[0], 
            args[1], 
            args[2], 
            args[3],
            args[4]
        ),
        elems=(
            proposals, 
            valid_mask, 
            bbox_deltas_pred, 
            gt_labels,
            gt_bboxes
        ),
        fn_output_signature=tf.TensorSpec(
            shape=(), 
            dtype=tf.float32
        ),
    )

    return tf.reduce_mean(losses)
