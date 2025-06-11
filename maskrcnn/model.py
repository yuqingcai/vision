import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from fpn import FPNGenerator
from backbone import build_resnet50, build_resnet101
from rpn import AnchorGenerator, RPNHead, ProposalGenerator
from loss import rpn_class_loss, rpn_bbox_loss, roi_class_loss, \
    roi_bbox_loss, roi_mask_loss
from roi import ROIAlign, ROIClassifierHead, ROIBBoxHead, ROIMaskHead
from utils import sample_and_assign_targets

class MaskRCNN(Model):
    def __init__(self, 
                 input_shape=(None, None, 3), 
                 batch_size=1, 
                 backbone_type='resnet50', 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.backbone = self.build_backbone(
            input_shape, 
            batch_size, 
            backbone_type
        )
        
        self.fpn = FPNGenerator(feature_size=256)

        self.anchor_generator = AnchorGenerator(
            ratios=[0.5, 1, 2], 
            scales=[8, 16, 32]
        )
        self.roi_align = ROIAlign(output_size=7, sampling_ratio=2)
        self.rpn_head = RPNHead(num_anchors=9, feature_size=256)
        self.proposal_generator = ProposalGenerator(
            pre_nms_topk=6000,
            post_nms_topk=1000,
            nms_thresh=0.7,
            min_size=16
        )

        self.roi_classifier_head = ROIClassifierHead(
            num_classes=80, 
            hidden_dim=1024
        )

        self.roi_bbox_head = ROIBBoxHead(
            num_classes=80, 
            hidden_dim=1024
        )
        
        self.roi_mask_head = ROIMaskHead(
            num_classes=80,
            hidden_dim=256,
            mask_size=28
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def build(self, input_shape):
        super().build(input_shape)
        self.built = True
        self.fpn.build(input_shape)
        self.anchor_generator.build(input_shape)
        self.roi_align.build(input_shape)
        self.rpn_head.build(input_shape)
        self.proposal_generator.build(input_shape)

    def build_backbone(self, input_shape, batch_size, backbone_type):
        if backbone_type == 'resnet101':
            return build_resnet101(input_shape, batch_size)
        elif backbone_type == 'resnet50':
            return build_resnet50(input_shape, batch_size)
        else:
            raise ValueError("backbone_type must be 'resnet50' or 'resnet101'")
    
    def call(self, image, training=False):
        c2, c3, c4, c5 = self.backbone(image, training=training)
        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
        anchors = self.anchor_generator(
            strides=[4, 8, 16, 32],
            base_sizes=[4, 8, 16, 32],
            feature_maps=[p2, p3, p4, p5]
        )
        rpn_logits, rpn_bbox_deltas = self.rpn_head([p2, p3, p4, p5])
        proposals = self.proposal_generator(
            image, anchors, rpn_bbox_deltas, rpn_logits
        )
        proposal_features = self.roi_align(
            [p2, p3, p4, p5], 
            proposals, 
            strides=[4, 8, 16, 32]
        )
        roi_class_logits = self.roi_classifier_head(
            proposal_features, 
            training=training
        )
        roi_bbox_deltas = self.roi_bbox_head(
            proposal_features, 
            training=training
        )
        roi_masks = self.roi_mask_head(
            proposal_features, 
            training=training
        )

        return (
            rpn_logits, 
            rpn_bbox_deltas, 
            proposals, 
            roi_class_logits, 
            roi_bbox_deltas, 
            roi_masks
        )
    
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, inputs):
        
        image, \
        (rpn_labels, \
         rpn_targets, \
         gt_boxes, \
         gt_classes, \
         gt_masks) = inputs

        with tf.GradientTape() as tape:
            rpn_logits, \
            rpn_bbox_deltas, \
            proposals, \
            roi_class_logits, \
            roi_bbox_deltas, \
            roi_masks = self.call(image, training=True)

            roi_boxes, roi_labels, roi_bbox_targets, roi_mask_targets = \
                sample_and_assign_targets(
                    proposals, 
                    gt_boxes, 
                    gt_classes, 
                    gt_masks)

            # 计算 RPN 损失
            loss_0 = rpn_class_loss(rpn_logits, rpn_labels)
            loss_1 = rpn_bbox_loss(
                rpn_bbox_deltas, 
                rpn_targets, 
                rpn_labels
            )
            loss_2 = roi_class_loss(roi_class_logits, roi_labels)
            loss_3 = roi_bbox_loss(
                roi_bbox_deltas, 
                roi_bbox_targets, 
                roi_labels
            )
            loss_4 = roi_mask_loss(
                roi_masks, 
                roi_mask_targets, 
                roi_labels
            )

            total_loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)

        return {"loss": self.loss_tracker.result()}
