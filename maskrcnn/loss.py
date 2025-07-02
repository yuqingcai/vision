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
        batch_indices, 
        objectness_logits, 
        gt_bboxes):
    
    # proposals shape: [N, 4]
    # batch_indices shape: [N]
    # objectness_logits shape: [N, 1]
    # gt_bboxes shape: [B, M, 4]
    
    batch_size = tf.reduce_max(batch_indices) + 1

    def loss_per_image(index):
        mask = tf.equal(batch_indices, index)
        proposals_selected = tf.boolean_mask(proposals, mask)
        objectness_logits_selected = tf.boolean_mask(objectness_logits, mask)

        gt_bboxes_selected = gt_bboxes[index]
        if isinstance(gt_bboxes_selected, tf.RaggedTensor):
            gt_bboxes_selected = gt_bboxes_selected.to_tensor()

        # proposals_selected shape: [N, 4]
        # gt_bboxes_selected shape: [M, 4]
        # ious shape: [N, M]
        ious = compute_iou(proposals_selected, gt_bboxes_selected)
        # Find the best matching ground truth for each proposal
        best_gt_ious = tf.reduce_max(ious, axis=1)
        pos_mask = best_gt_ious >= 0.7
        neg_mask = best_gt_ious < 0.3
        labels = tf.where(pos_mask, 1.0, tf.where(neg_mask, 0.0, -1.0))

        valid_mask = labels >= 0

        objectness_logits_valid = tf.boolean_mask(
            objectness_logits_selected, valid_mask
        )
        objectness_logits_valid = tf.squeeze(
            objectness_logits_valid, axis=-1
        )

        labels_valid = tf.boolean_mask(labels, valid_mask)

        loss = tf.cond(
            tf.size(labels_valid) > 0,
            lambda: tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels_valid, 
                    logits=objectness_logits_valid
                )
            ),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        return loss

    loss = tf.map_fn(
        loss_per_image,
        tf.range(batch_size),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    
    loss = tf.reduce_mean(loss)

    return loss

def loss_rpn_box_reg_fn(
    proposals, 
    batch_indices, 
    bbox_deltas, 
    gt_bboxes):
    
    # proposals shape: [N, 4]
    # batch_indices shape: [N]
    # bbox_deltas shape: [N, 4]
    # gt_bboxes shape: [B, M, 4]
    
    batch_size = tf.reduce_max(batch_indices) + 1

    def loss_per_image(index):
        mask = tf.equal(batch_indices, index)
        proposals_selected = tf.boolean_mask(proposals, mask)
        bbox_deltas_selected = tf.boolean_mask(bbox_deltas, mask)
    
        gt_bboxes_selected = gt_bboxes[index]
        if isinstance(gt_bboxes_selected, tf.RaggedTensor):
            gt_bboxes_selected = gt_bboxes_selected.to_tensor()


        return 0.0
    
    loss = tf.map_fn(
        loss_per_image,
        tf.range(batch_size),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    
    loss = tf.reduce_mean(loss)

    return loss

def loss_class_fn(proposals, class_logits, gt_class_ids, gt_bboxes):
    """
    将 proposals 和 gt_bboxes 进行IoU计算，找到与每个 proposal 匹配的 ground truth，将与ground truth 匹配的 gt_class_id 赋值给该 proposal，并把 proposal 的 class_logits（预期值）和 gt_class_id（实际值）进行比较，计算损失。
    """
    pass

def loss_bbox_fn(proposals, bbox_deltas, gt_bboxes):
    pass

def loss_mask_fn(proposals, masks, gt_masks):
    pass
