import tensorflow as tf
from utils import compute_iou

def loss_rpn_objectness_fn(
        proposals, 
        valid_mask, 
        objectness_logits, 
        gt_bboxes,
        sample_num_pos=128,
        sample_num_neg=128,
        seed=None,
        iou_pos_thresh=0.7,
        iou_neg_thresh=0.3):
    # proposals shape: [B, N, 4]
    # valid_mask shape: [B, N]
    # objectness_logits shape: [B, N, 1]
    # gt_bboxes shape: [B, M, 4]
    
    def loss_per_image(
            proposals, 
            valid_mask, 
            objectness_logits_pred, 
            gt_boxes):
        
        # proposals shape: [N, 4]
        # objectness_logits_pred shape: [N, 1]
        # gt_boxes shape: [M, 4]
        # valid_mask shape: [N]
        # ious shape: [N, M]
        
        proposals_valid = tf.boolean_mask(proposals, valid_mask)
        objectness_logits_valid = tf.boolean_mask(
            objectness_logits_pred, valid_mask
        )
        objectness_logits_valid = tf.squeeze(
            objectness_logits_valid, axis=-1
        )

        # Find the best matching ground truth for each proposal
        ious = compute_iou(proposals_valid, gt_boxes.to_tensor())
        best_gt_ious = tf.reduce_max(ious, axis=1)
        pos_mask = best_gt_ious >= iou_pos_thresh
        neg_mask = best_gt_ious < iou_neg_thresh
        labels = tf.where(pos_mask, 1.0, tf.where(neg_mask, 0.0, -1.0))

        # sample positive and negative proposals
        pos_inds = tf.where(labels == 1.0)[:, 0]
        neg_inds = tf.where(labels == 0.0)[:, 0]
        num_pos_sample = tf.minimum(tf.size(pos_inds), sample_num_pos)
        num_neg_sample = tf.minimum(tf.size(neg_inds), sample_num_neg)
        pos_inds = tf.random.shuffle(pos_inds, seed=seed)[:num_pos_sample]
        neg_inds = tf.random.shuffle(neg_inds, seed=seed)[:num_neg_sample]
        sampled_inds = tf.concat([pos_inds, neg_inds], axis=0)
        sampled_labels = tf.gather(labels, sampled_inds)
        sampled_logits = tf.gather(objectness_logits_valid, sampled_inds)

        loss = tf.cond(
            tf.size(sampled_labels) > 0,
            lambda: tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=sampled_labels, 
                    logits=sampled_logits
                )
            ),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        return loss

    losses = tf.map_fn(
        lambda args: loss_per_image(
            args[0], 
            args[1], 
            args[2], 
            args[3]
        ),
        elems=(
            proposals, 
            valid_mask, 
            objectness_logits, 
            gt_bboxes
        ),
        fn_output_signature=tf.TensorSpec(
            shape=(), 
            dtype=tf.float32
        ),
    )
    
    return tf.reduce_mean(losses)
