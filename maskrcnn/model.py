import tensorflow as tf
from tensorflow.keras import Model
from fpn import FPNGenerator
from backbone import ResNet50Backbone, ResNet101Backbone
from rpn import AnchorGenerator, RPNHead, ProposalGenerator
from loss import loss_class_fn, loss_bbox_fn, loss_mask_fn, \
    loss_objectness_fn, loss_rpn_box_reg_fn
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
        self.class_num = 80  # COCO dataset has 80 classes
        
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
        
        self.roi_align_class = ROIAlign(
            output_size=7, 
            sampling_ratio=2,
            feature_strides=self.fpn.strides(),
            feature_size=self.fpn_feature_size
        )

        self.roi_align_mask = ROIAlign(
            output_size=14, 
            sampling_ratio=2,
            feature_strides=self.fpn.strides(),
            feature_size=self.fpn_feature_size
        )

        self.roi_classifier_head = ROIClassifierHead(
            num_classes=self.class_num + 1, # +1 for background class
            hidden_dim=1024
        )

        self.roi_bbox_head = ROIBBoxHead(
            num_classes=self.class_num + 1, # +1 for background class
            hidden_dim=1024
        )
        
        self.roi_mask_head = ROIMaskHead(
            num_classes=self.class_num,
            conv_dim=256,
            num_convs=4,
            roi_output_size=self.roi_align_mask.output_size
        )
        
        self.loss_objectness_tracker = tf.keras.metrics.Mean(name="objectness_loss")
        self.loss_rpn_box_reg_tracker = tf.keras.metrics.Mean(name="loss_rpn_box_reg")
        self.loss_class_tracker = tf.keras.metrics.Mean(name="loss_class")
        self.loss_box_reg_tracker = tf.keras.metrics.Mean(name="loss_box_reg")
        self.loss_mask_tracker = tf.keras.metrics.Mean(name="loss_mask")
        self.loss_total_tracker = tf.keras.metrics.Mean(name="loss_total")
    

    def build_backbone(self, input_shape, batch_size, backbone_type):
        if backbone_type == 'resnet50':
            return ResNet50Backbone(input_shape, batch_size)
        elif backbone_type == 'resnet101':
            return ResNet101Backbone(input_shape, batch_size)
        else:
            raise ValueError("backbone_type must be 'resnet50' or 'resnet101'")
    
    
    def call(self, images, image_sizes, training=False):
        """
        rois shape: [B, (None), 4]
        features_class shape: [B, (None), 7, 7, 256]
        class_logits shape: [B, (None), 81]
        bbox_deltas shape: [B, (None), 81*4]
        features_mask shape: [B, (None), 14, 14, 256]
        masks shape: [B, (None), 28, 28, 80]
        """
        if os.environ.get("GPU_ENABLE", "FALSE") == "TRUE":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                print("Before backbone, GPU memory:", info)

        c2, c3, c4, c5 = self.backbone(images, training=training)
        
        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
        
        anchors, batch_indices = self.anchor_generator(
            feature_maps=[p2, p3, p4, p5],
            strides=self.fpn.strides(),
            base_sizes=self.fpn.base_sizes(),
            ratios=self.anchor_ratios, 
            scales=self.anchor_scales,
            image_sizes=image_sizes
        )
        
        rpn_class_logits, rpn_bbox_deltas = self.rpn_head([p2, p3, p4, p5])

        proposals, batch_indices, rpn_class_logits, rpn_bbox_deltas = \
            self.proposal_generator(
                anchors, batch_indices, image_sizes, rpn_class_logits, 
                rpn_bbox_deltas
            )
        
        # roi class and bbox head
        features_class, feature_batch_indices = self.roi_align_class(
            feature_maps=[p2, p3, p4, p5], 
            rois=proposals,
            batch_indices=batch_indices
        )
        class_logits = 0.0
        # class_logits = self.roi_classifier_head(
        #     features_class, 
        #     training=training
        # )
        
        bbox_deltas = 0.0
        # bbox_deltas = self.roi_bbox_head(
        #     features_class, 
        #     training=training
        # )
        
        # # roi mask head
        # features_mask, _ = self.roi_align_mask(
        #     feature_maps=[p2, p3, p4, p5], 
        #     rois=proposals,
        #     batch_indices=batch_indices
        # )
        masks = 0.0
        # masks = self.roi_mask_head(
        #     features_mask, 
        #     training=training
        # )

        if os.environ.get("GPU_ENABLE", "FALSE") == "TRUE":
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                print("After backbone, GPU memory:", info)

        return (
            proposals, 
            rpn_class_logits,
            rpn_bbox_deltas,
            class_logits, 
            bbox_deltas, 
            masks,
            feature_batch_indices
        )

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    @tf.function(reduce_retracing=True)
    def train_step(self, batch):
        images = batch['image']
        sizes = batch['size']
        gt_bboxes = batch['bbox']
        gt_masks = batch['mask']
        gt_class_ids = batch['category_id']

        with tf.GradientTape() as tape:
            rois, rpn_class_logits, rpn_box_deltas, \
                class_logits, bbox_deltas, masks, batch_indices = \
                    self.call(images, sizes, training=True)

            # loss_objectness = loss_objectness_fn()
            # loss_rpn_box_reg = loss_rpn_box_reg_fn()
            # loss_class = loss_class_fn(
            #     proposals, class_logits, gt_class_ids, gt_bboxes)
            
            # loss_box_reg = loss_bbox_reg_fn(
            #     proposals, bbox_deltas, gt_bboxes
            # )
            # loss_mask = loss_mask_fn(
            #     proposals,
            #     masks, 
            #     gt_masks
            # )

            loss_objectness = tf.reduce_mean(tf.square(rpn_class_logits))
            loss_rpn_box_reg = tf.reduce_mean(tf.square(rpn_box_deltas))
            loss_class = tf.reduce_mean(tf.square(rpn_class_logits))
            loss_box_reg = tf.reduce_mean(tf.square(rpn_class_logits))
            loss_mask = tf.reduce_mean(tf.square(rpn_class_logits))

            loss_total = loss_objectness + loss_rpn_box_reg + \
                loss_class + loss_box_reg + loss_mask

        grads = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update metrics
        self.loss_objectness_tracker.update_state(loss_objectness)
        self.loss_rpn_box_reg_tracker.update_state(loss_rpn_box_reg)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_box_reg_tracker.update_state(loss_box_reg)
        self.loss_mask_tracker.update_state(loss_mask)
        self.loss_total_tracker.update_state(loss_total)
        
        return {
            'loss_objectness': self.loss_objectness_tracker.result(),
            'loss_rpn_box_reg': self.loss_rpn_box_reg_tracker.result(),
            'loss_class': self.loss_class_tracker.result(),
            'loss_box_reg': self.loss_box_reg_tracker.result(),
            'loss_mask': self.loss_mask_tracker.result(),
            'loss_total': self.loss_total_tracker.result(),
        }

    def reset_metrics(self):
        self.loss_objectness_tracker.reset_states()
        self.loss_rpn_box_reg_tracker.reset_states()
        self.loss_class_tracker.reset_states()
        self.loss_box_reg_tracker.reset_states()
        self.loss_mask_tracker.reset_states()
        self.loss_total_tracker.reset_states()
