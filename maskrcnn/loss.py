import tensorflow as tf


def rpn_class_loss(rpn_logits, rpn_labels):
    indices = tf.where(rpn_labels >= 0)
    logits = tf.gather_nd(rpn_logits, indices)
    labels = tf.gather_nd(rpn_labels, indices)
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            labels, logits, from_logits=True)
        )


def rpn_bbox_loss(rpn_deltas, rpn_targets, rpn_labels):
    indices = tf.where(rpn_labels == 1)
    deltas = tf.gather_nd(rpn_deltas, indices)
    targets = tf.gather_nd(rpn_targets, indices)
    return tf.reduce_mean(tf.keras.losses.Huber()(targets, deltas))


def roi_class_loss(class_logits, class_labels):
    indices = tf.where(class_labels >= 0)
    logits = tf.gather_nd(class_logits, indices)
    labels = tf.gather_nd(class_labels, indices)
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True)
        )


def roi_bbox_loss(bbox_deltas, bbox_targets, class_labels):
    indices = tf.where(class_labels >= 0)
    deltas = tf.gather_nd(bbox_deltas, indices)
    targets = tf.gather_nd(bbox_targets, indices)
    return tf.reduce_mean(tf.keras.losses.Huber()(targets, deltas))


def roi_mask_loss(masks, mask_targets, class_labels):
    indices = tf.where(class_labels >= 0)
    masks = tf.gather_nd(masks, indices)
    targets = tf.gather_nd(mask_targets, indices)
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            targets, masks, from_logits=True)
        )