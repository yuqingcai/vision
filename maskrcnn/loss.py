import tensorflow as tf

def class_loss_fn(class_logits, class_labels):
    indices = tf.where(class_labels >= 0)
    logits = tf.gather_nd(class_logits, indices)
    labels = tf.gather_nd(class_labels, indices)
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True)
        )

def bbox_loss_fn(bbox_deltas, bbox_targets, class_labels):
    indices = tf.where(class_labels >= 0)
    deltas = tf.gather_nd(bbox_deltas, indices)
    targets = tf.gather_nd(bbox_targets, indices)
    return tf.reduce_mean(tf.keras.losses.Huber()(targets, deltas))


def mask_loss_fn(masks, mask_targets, class_labels):
    indices = tf.where(class_labels >= 0)
    masks = tf.gather_nd(masks, indices)
    targets = tf.gather_nd(mask_targets, indices)
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            targets, masks, from_logits=True)
        )