import tensorflow as tf
from tensorflow.keras import Model
from fpn import FPNGenerator
from backbone import ResNet50Backbone, ResNet101Backbone
from rpn import AnchorGenerator, RPNHead, ProposalGenerator
from loss import rpn_class_loss_fn, rpn_bbox_loss_fn, \
    roi_class_loss_fn, roi_bbox_loss_fn, roi_mask_loss_fn
from roi import ROIAlign, ROIClassifierHead, ROIBBoxHead, ROIMaskHead
from utils import sample_and_assign_targets
import os

class MaskRCNN(Model):
    def __init__(self, 
                 input_shape, 
                 batch_size, 
                 backbone_type='resnet50', 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.backbone = self.build_backbone(
            input_shape, 
            batch_size, 
            backbone_type
        )
        
        # ratios is height:width
        self.anchor_ratios = [0.5, 1, 2]
        self.anchor_scales = [1.0, 1.5, 2.0]
        self.fpn_feature_size = 256
        
        self.fpn = FPNGenerator(feature_size=self.fpn_feature_size)

        self.anchor_generator = AnchorGenerator()

        self.rpn_head = RPNHead(
            anchors_per_location=len(self.anchor_ratios) * \
                len(self.anchor_scales), 
            feature_size=self.fpn_feature_size)
        
        self.proposal_generator = ProposalGenerator(
            pre_nms_topk=6000,
            post_nms_topk=1000,
            nms_thresh=0.5,
            min_size=16
        )
        
        self.roi_align = ROIAlign(
            output_size=7, 
            sampling_ratio=2,
            feature_strides=self.fpn.strides(),
            feature_size=self.fpn_feature_size
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
        
        self.rpn_class_loss_tracker = tf.keras.metrics.Mean(name="rpn_class_loss")
        self.rpn_bbox_loss_tracker = tf.keras.metrics.Mean(name="rpn_bbox_loss")
        self.roi_class_loss_tracker = tf.keras.metrics.Mean(name="roi_class_loss")
        self.roi_bbox_loss_tracker = tf.keras.metrics.Mean(name="roi_bbox_loss")
        self.roi_mask_loss_tracker = tf.keras.metrics.Mean(name="roi_mask_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    

    def build_backbone(self, input_shape, batch_size, backbone_type):
        if backbone_type == 'resnet50':
            return ResNet50Backbone(input_shape, batch_size)
        elif backbone_type == 'resnet101':
            return ResNet101Backbone(input_shape, batch_size)
        else:
            raise ValueError("backbone_type must be 'resnet50' or 'resnet101'")
    
    
    def call(self, images, image_sizes, training=False):
        if os.environ.get("GPU_ENABLE", "FALSE") == "TRUE":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                print("Before backbone, GPU memory:", info)

        c2, c3, c4, c5 = self.backbone(images, training=training)

        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
        
        anchors = self.anchor_generator(
            feature_maps=[p2, p3, p4, p5],
            strides=self.fpn.strides(),
            base_sizes=self.fpn.base_sizes(),
            ratios=self.anchor_ratios, 
            scales=self.anchor_scales,
            image_sizes=image_sizes
        )

        rpn_class_logits, rpn_bbox_deltas = self.rpn_head([p2, p3, p4, p5])
        
        proposals = self.proposal_generator(
            image_sizes, anchors, rpn_class_logits, rpn_bbox_deltas
        )

        features = self.roi_align(
            feature_maps=[p2, p3, p4, p5], 
            rois=proposals
        )
        # tf.print('features shape:', tf.shape(features))
        
        # roi_class_logits = self.roi_classifier_head(
        #     features, 
        #     training=training
        # )
        # 
        # roi_bbox_deltas = self.roi_bbox_head(
        #     features, 
        #     training=training
        # )
        # 
        # roi_masks = self.roi_mask_head(
        #     features, 
        #     training=training
        # )

        if os.environ.get("GPU_ENABLE", "FALSE") == "TRUE":
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                print("After backbone, GPU memory:", info)

        # return (
        #     rpn_class_logits, 
        #     rpn_bbox_deltas, 
        #     proposals, 
        #     roi_class_logits, 
        #     roi_bbox_deltas, 
        #     roi_masks
        # )

        return (
            0.0, 
            0.0, 
            0.0, 
            0.0, 
            0.0, 
            0.0
        )
    
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    @tf.function(reduce_retracing=True)
    def train_step(self, image, size):
        # image = inputs['image']
        # size = inputs['size']
        # bboxes = inputs['bboxes']
        # masks = inputs['masks']
        
        with tf.GradientTape() as tape:

            rpn_logits, rpn_bbox_deltas, proposals, roi_class_logits, \
            roi_bbox_deltas, roi_masks = self.call(image, size, training=True)
            
            # roi_boxes, roi_labels, roi_bbox_targets, roi_mask_targets = \
            #     sample_and_assign_targets(
            #         proposals, 
            #         gt_boxes, 
            #         gt_classes, 
            #         gt_masks)

            # # losses
            # rpn_class_loss = rpn_class_loss_fn(rpn_logits, rpn_labels)
            # rpn_bbox_loss = rpn_bbox_loss_fn(
            #     rpn_bbox_deltas, 
            #     rpn_targets, 
            #     rpn_labels
            # )
            # roi_class_loss = roi_class_loss_fn(roi_class_logits, roi_labels)
            # roi_bbox_loss = roi_bbox_loss_fn(
            #     roi_bbox_deltas, 
            #     roi_bbox_targets, roi_labels
            # )
            # roi_mask_loss = roi_mask_loss_fn(
            #     roi_masks, 
            #     roi_mask_targets, 
            #     roi_labels
            # )

            # total_loss = rpn_class_loss + rpn_bbox_loss + \
            #     roi_class_loss + roi_bbox_loss + roi_mask_loss
        

        # grads = tape.gradient(total_loss, self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # self.rpn_class_loss_tracker.update_state(rpn_class_loss)
        # self.rpn_bbox_loss_tracker.update_state(rpn_bbox_loss)
        # self.roi_class_loss_tracker.update_state(roi_class_loss)
        # self.roi_bbox_loss_tracker.update_state(roi_bbox_loss)
        # self.roi_mask_loss_tracker.update_state(roi_mask_loss)
        # self.total_loss_tracker.update_state(total_loss)
        
        # return {
        #     'rpn_class_loss': self.rpn_class_loss_tracker.result(),
        #     'rpn_bbox_loss': self.rpn_bbox_loss_tracker.result(),
        #     'roi_class_loss': self.roi_class_loss_tracker.result(),
        #     'roi_bbox_loss': self.roi_bbox_loss_tracker.result(),
        #     'roi_mask_loss': self.roi_mask_loss_tracker.result(),
        #     'total_loss': self.total_loss_tracker.result(),
        # }
        return {
            'rpn_class_loss': 0.0,
            'rpn_bbox_loss': 0.0,
            'roi_class_loss': 0.0,
            'roi_bbox_loss': 0.0,
            'roi_mask_loss': 0.0,
            'total_loss': 0.0,
        }
        
    def reset_metrics(self):
        self.rpn_class_loss_tracker.reset_states()
        self.rpn_bbox_loss_tracker.reset_states()
        self.roi_class_loss_tracker.reset_states()
        self.roi_bbox_loss_tracker.reset_states()
        self.roi_mask_loss_tracker.reset_states()
        self.total_loss_tracker.reset_states()
