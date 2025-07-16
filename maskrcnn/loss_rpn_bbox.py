import tensorflow as tf
from utils import compute_iou

def loss_rpn_box_reg_fn(
        proposals, 
        valid_mask, 
        bbox_deltas_pred, 
        gt_bboxes,
        sample_num_pos=128, 
        iou_pos_thresh=0.7, 
        seed=None):
    
    # proposals shape: [B, N, 4]
    # valid_mask shape: [B, N]
    # bbox_deltas_pred shape: [B, N, 4]
    # gt_bboxes shape: [B, M, 4]
    
    def loss_per_image(
            proposals, 
            valid_mask, 
            bbox_deltas_pred, 
            gt_boxes):

        # proposals shape: [N, 4]
        # valid_mask shape: [N]
        # bbox_deltas_pred shape: [N, 4]
        # gt_boxes shape: [M, 4]
        
        proposals_valid = tf.boolean_mask(proposals, valid_mask)
        bbox_deltas_pred_valid = tf.boolean_mask(bbox_deltas_pred, valid_mask)
         # Find the best matching ground truth for each proposal
        ious = compute_iou(proposals_valid, gt_boxes.to_tensor())
        best_gt_inds = tf.argmax(ious, axis=1)
        best_gt_ious = tf.reduce_max(ious, axis=1)
        pos_mask = best_gt_ious >= iou_pos_thresh
        pos_inds = tf.where(pos_mask)[:, 0]
        num_pos = tf.minimum(tf.size(pos_inds), sample_num_pos)
        pos_inds = tf.random.shuffle(pos_inds, seed=seed)[:num_pos]
        
        def loss_fn():
            proposals_pos = tf.gather(proposals_valid, pos_inds)
            bbox_deltas_pred_pos = tf.gather(bbox_deltas_pred_valid, pos_inds)
            best_gt_inds_pos = tf.gather(best_gt_inds, pos_inds)
            gt_boxes_pos = tf.gather(gt_boxes.to_tensor(), best_gt_inds_pos)
            
            # compute the target deltas
            px, py, pw, ph = (proposals_pos[:, 0] + proposals_pos[:, 2]) * 0.5, \
                (proposals_pos[:, 1] + proposals_pos[:, 3]) * 0.5, \
                proposals_pos[:, 2] - proposals_pos[:, 0], \
                proposals_pos[:, 3] - proposals_pos[:, 1]

            gx, gy, gw, gh = (gt_boxes_pos[:, 0] + gt_boxes_pos[:, 2]) * 0.5, \
                (gt_boxes_pos[:, 1] + gt_boxes_pos[:, 3]) * 0.5, \
                gt_boxes_pos[:, 2] - gt_boxes_pos[:, 0], \
                gt_boxes_pos[:, 3] - gt_boxes_pos[:, 1]

            tx = (gx - px) / tf.maximum(pw, 1e-6)
            ty = (gy - py) / tf.maximum(ph, 1e-6)
            tw = tf.math.log(gw / tf.maximum(pw, 1e-6))
            th = tf.math.log(gh / tf.maximum(ph, 1e-6))
            bbox_deltas_target =  tf.stack([tx, ty, tw, th], axis=-1)
            
            # compute the loss
            loss = tf.reduce_mean(
                tf.losses.huber(
                    bbox_deltas_target, 
                    bbox_deltas_pred_pos, 
                    delta=1.0
                )
            )
            
            return loss
        
        loss = tf.cond(
            tf.size(pos_inds) > 0,
            loss_fn,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        return loss


    loss = tf.map_fn(
        lambda args: loss_per_image(
            args[0], 
            args[1], 
            args[2], 
            args[3]
        ),
        elems=(
            proposals, 
            valid_mask, 
            bbox_deltas_pred, 
            gt_bboxes
        ),
        fn_output_signature=tf.TensorSpec(
            shape=(), 
            dtype=tf.float32
        ),
    )
    
    return tf.reduce_mean(loss)