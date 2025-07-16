import tensorflow as tf
from utils import compute_iou

def loss_mask_fn(
        proposals, 
        valid_mask, 
        masks_pred, 
        gt_labels, 
        gt_bboxes, 
        gt_masks,
        iou_pos_thresh=0.7,
        iou_neg_thresh=0.3):
    
    def loss_per_image(
            proposals, 
            valid_mask, 
            masks_pred, 
            gt_labels, 
            gt_bboxes, 
            gt_masks):
        
        proposals_valid = tf.boolean_mask(proposals, valid_mask)
        masks_pred_valid = tf.boolean_mask(
            masks_pred, 
            valid_mask
        )
        gt_bboxes = gt_bboxes.to_tensor()
        gt_masks = gt_masks.to_tensor()

        # Find the best matching ground truth for each proposal
        ious = compute_iou(proposals_valid, gt_bboxes)
        best_gt_ious = tf.reduce_max(ious, axis=1)
        best_gt_inds = tf.argmax(ious, axis=1, output_type=tf.int32)

        pos_mask = best_gt_ious >= iou_pos_thresh

        proposals_pos = tf.boolean_mask(proposals_valid, pos_mask)
        if tf.size(proposals_pos) == 0:
            # tf.print('No positive proposals found, mask loss returning zero.')
            return tf.constant(0.0, dtype=tf.float32)
        
        masks_pred_pos = tf.boolean_mask(
            masks_pred_valid, 
            pos_mask
        )
        best_gt_inds_pos = tf.boolean_mask(best_gt_inds, pos_mask)  # [M]
        gt_boxes_pos = tf.gather(gt_bboxes, best_gt_inds_pos)       # [M, 4]
        gt_masks_pos = tf.gather(gt_masks, best_gt_inds_pos)        # [M, H, W]
        gt_labels_pos = tf.gather(gt_labels, best_gt_inds_pos)

        # compute the target masks
        N = tf.shape(gt_boxes_pos)[0]
        H = tf.cast(tf.shape(gt_masks_pos)[1], tf.float32)
        W = tf.cast(tf.shape(gt_masks_pos)[2], tf.float32)
        M = masks_pred_pos.shape[1]  # MxM is the output size of the mask

        y1 = gt_boxes_pos[:, 1] / H
        x1 = gt_boxes_pos[:, 0] / W
        y2 = gt_boxes_pos[:, 3] / H
        x2 = gt_boxes_pos[:, 2] / W
        boxes_norm = tf.stack([y1, x1, y2, x2], axis=1)  # [N, 4]

        # gt_masks_pos: [N, H, W] -> [N, H, W, 1]
        # crop_and_resize
        gt_masks_pos_exp = tf.expand_dims(gt_masks_pos, -1)
        box_indices = tf.range(N)
        gt_masks_cropped = tf.image.crop_and_resize(
            gt_masks_pos_exp,   # [N, H, W, 1]
            boxes_norm,         # [N, 4]
            box_indices,        # [N]
            crop_size=[M, M]    # output size
        )                       # [N, M, M, 1]
        # squeeze the last dimension
        # gt_masks_cropped: [N, M, M, 1] -> [N, M, M]
        gt_masks_cropped = tf.squeeze(gt_masks_cropped, axis=-1)

        # pick up the masks corresponding to the gt_labels
        # masks_pred_pos: [N, M, M, C] -> [N, C, M, M]
        masks_trans = tf.transpose(masks_pred_pos, [0, 3, 1, 2]) # [N, C, M, M]
        batch_indices = tf.range(tf.shape(gt_labels_pos)[0])     # [N]
        gather_indices = tf.stack([batch_indices, gt_labels_pos], axis=1)  # [N, 2]
        masks_pred_sel = tf.gather_nd(masks_trans, gather_indices)  # [N, M, M]

        # tf.print('gt_labels_pos shape:', tf.shape(gt_labels_pos))
        # tf.print('masks_cropped shape:', tf.shape(gt_masks_cropped))
        # tf.print('masks_pred_sel shape:', tf.shape(masks_pred_sel))

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=gt_masks_cropped,
                logits=masks_pred_sel
            )
        )
        
        return loss
    
    
    loss = tf.map_fn(
        lambda args: loss_per_image(
            args[0], 
            args[1], 
            args[2], 
            args[3],
            args[4],
            args[5]
        ),
        elems=(
            proposals, 
            valid_mask, 
            masks_pred, 
            gt_labels, 
            gt_bboxes, 
            gt_masks
        ),
        fn_output_signature=tf.TensorSpec(
            shape=(), 
            dtype=tf.float32
        ),
    )

    return loss
