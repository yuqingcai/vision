import tensorflow as tf
from utils import compute_iou

def loss_classifier_reg_fn(
        proposals, 
        valid_mask, 
        class_logits_pred, 
        gt_labels, 
        gt_bboxes,
        iou_pos_thresh=0.7,
        iou_neg_thresh=0.3):
    
    def loss_per_image(
            proposals, 
            valid_mask, 
            class_logits_pred, 
            gt_labels, 
            gt_boxes):
        
        proposals_valid = tf.boolean_mask(proposals, valid_mask)
        class_logits_pred_valid = tf.boolean_mask(
            class_logits_pred, 
            valid_mask
        )
        
        gt_boxes = gt_boxes.to_tensor()

         # Find the best matching ground truth for each proposal
        ious = compute_iou(proposals_valid, gt_boxes)
        best_gt_inds = tf.argmax(ious, axis=1, output_type=tf.int32)
        best_gt_ious = tf.reduce_max(ious, axis=1)
        pos_mask = best_gt_ious >= iou_pos_thresh
        neg_mask = best_gt_ious < iou_neg_thresh
        
        target_class = tf.zeros_like(best_gt_inds, dtype=tf.int32)
        pos_indices = tf.where(pos_mask)[:, 0]
        target_class_pos = tf.gather(
            gt_labels, 
            tf.gather(best_gt_inds, pos_indices)
        )
        target_class = tf.tensor_scatter_nd_update(
            target_class, 
            tf.expand_dims(pos_indices, 1), 
            target_class_pos
        )

        keep = tf.logical_or(pos_mask, neg_mask)
        keep_indices = tf.where(keep)[:, 0]
        if tf.size(keep_indices) == 0:
            # tf.print('No valid proposals found, classifier loss returning zero.')
            return tf.constant(0.0, dtype=tf.float32)

        sel_logits = tf.gather(class_logits_pred_valid, keep_indices)
        sel_targets = tf.gather(target_class, keep_indices)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sel_targets, logits=sel_logits
            )
        )
        return loss

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
            class_logits_pred, 
            gt_labels, 
            gt_bboxes
        ),
        fn_output_signature=tf.TensorSpec(
            shape=(), 
            dtype=tf.float32
        ),
    )

    return tf.reduce_mean(losses)
