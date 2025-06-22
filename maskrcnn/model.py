import tensorflow as tf
from tensorflow.keras import Model
from fpn import FPNGenerator
from backbone import ResNet50Backbone, ResNet101Backbone
from rpn import AnchorGenerator, RPNHead, ProposalGenerator
from loss import rpn_class_loss_fn, rpn_bbox_loss_fn, \
    roi_class_loss_fn, roi_bbox_loss_fn, roi_mask_loss_fn
from roi import ROIAlign, ROIClassifierHead, ROIBBoxHead, ROIMaskHead
from utils import sample_and_assign_targets
import time

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
        
        self.fpn = FPNGenerator(feature_size=256)

        self.anchor_generator = AnchorGenerator()

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

        self.rpn_class_loss_tracker = tf.keras.metrics.Mean(name="rpn_class_loss")
        self.rpn_bbox_loss_tracker = tf.keras.metrics.Mean(name="rpn_bbox_loss")
        self.roi_class_loss_tracker = tf.keras.metrics.Mean(name="roi_class_loss")
        self.roi_bbox_loss_tracker = tf.keras.metrics.Mean(name="roi_bbox_loss")
        self.roi_mask_loss_tracker = tf.keras.metrics.Mean(name="roi_mask_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    
    # def build(self, input_shape):
    #     super().build(input_shape)
    #     self.built = True
    #     self.fpn.build(input_shape)
    #     self.anchor_generator.build(input_shape)
    #     self.roi_align.build(input_shape)
    #     self.rpn_head.build(input_shape)
    #     self.proposal_generator.build(input_shape)

    def build_backbone(self, input_shape, batch_size, backbone_type):
        if backbone_type == 'resnet50':
            return ResNet50Backbone(input_shape, batch_size)
        elif backbone_type == 'resnet101':
            return ResNet101Backbone(input_shape, batch_size)
        else:
            raise ValueError("backbone_type must be 'resnet50' or 'resnet101'")
    
    
    def call(self, images, sizes, training=False):

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info = tf.config.experimental.get_memory_info('GPU:0')
            print("Before backbone, GPU memory:", info)
        c2, c3, c4, c5 = self.backbone(images, training=training)
        if gpus:
            info = tf.config.experimental.get_memory_info('GPU:0')
            print("After backbone, GPU memory:", info)
        # tf.print('c2 shape:', tf.shape(c2), 
        #          'c3 shape:', tf.shape(c3), 
        #          'c4 shape:', tf.shape(c4), 
        #          'c5 shape:', tf.shape(c5))
        
        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
        # tf.print('p2 shape:', tf.shape(p2), 
        #          'p3 shape:', tf.shape(p3), 
        #          'p4 shape:', tf.shape(p4), 
        #          'p5 shape:', tf.shape(p5))
        
        # strides of p2, p3, p4, p5 are 4, 8, 16, 32 respectively
        # base_sizes of p2, p3, p4, p5 are 32, 64, 128, 256 respectively
        # ratios is height:width
        anchors = self.anchor_generator(
            feature_maps=[p2, p3, p4, p5],
            strides=[4, 8, 16, 32],
            base_sizes=[32, 64, 128, 256],
            ratios=[0.5, 1, 2], 
            scales=[1.0, 1.5, 2.0],
            origin_sizes=sizes
        )
        # tf.print('anchors shape:', tf.shape(anchors))
        

        # rpn_logits, rpn_bbox_deltas = self.rpn_head([p2, p3, p4, p5])
        # proposals = self.proposal_generator(
        #     image, anchors, rpn_bbox_deltas, rpn_logits
        # )
        # proposal_features = self.roi_align(
        #     [p2, p3, p4, p5], 
        #     proposals, 
        #     strides=[4, 8, 16, 32]
        # )
        # roi_class_logits = self.roi_classifier_head(
        #     proposal_features, 
        #     training=training
        # )
        # roi_bbox_deltas = self.roi_bbox_head(
        #     proposal_features, 
        #     training=training
        # )
        # roi_masks = self.roi_mask_head(
        #     proposal_features, 
        #     training=training
        # )

        # return (
        #     rpn_logits, 
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

    @tf.function
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
