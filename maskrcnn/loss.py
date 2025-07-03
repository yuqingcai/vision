import tensorflow as tf

def compute_iou(boxes1, boxes2):
    """
    boxes1: [P, 4], boxes2: [G, 4]
    """
    boxes1 = tf.expand_dims(boxes1, 1)  # [P,1,4]
    boxes2 = tf.expand_dims(boxes2, 0)  # [1,G,4]

    inter_x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * \
        tf.maximum(inter_y2 - inter_y1, 0)
    
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])
    
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / tf.maximum(union_area, 1e-8)

    # shape [P, G]
    return iou

def loss_objectness_fn(
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

    
    def loss_per_image(proposals, valid_mask, objectness_logits_pred, gt_boxes):
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

        # compute loss
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

    loss = tf.map_fn(
        lambda args: loss_per_image(args[0], args[1], args[2], args[3]),
        elems=(proposals, valid_mask, objectness_logits, gt_bboxes),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    
    return tf.reduce_mean(loss)


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
    
    def loss_per_image(proposals, valid_mask, bbox_deltas_pred, gt_boxes):

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
        lambda args: loss_per_image(args[0], args[1], args[2], args[3]),
        elems=(proposals, valid_mask, bbox_deltas_pred, gt_bboxes),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    
    return tf.reduce_mean(loss)


def loss_class_fn(proposals, class_logits, gt_class_ids, gt_bboxes):
    """
    将 proposals 和 gt_bboxes 进行IoU计算，找到与每个 proposal 匹配的 ground truth，将与ground truth 匹配的 gt_class_id 赋值给该 proposal，并把 proposal 的 class_logits（预期值）和 gt_class_id（实际值）进行比较，计算损失。
    """
    pass

def loss_bbox_fn(proposals, bbox_deltas, gt_bboxes):
    pass

def loss_mask_fn(proposals, masks, gt_masks):
    pass
