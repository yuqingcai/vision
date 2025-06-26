import tensorflow as tf
from tensorflow.keras import Model
from fpn import FPNGenerator
from backbone import ResNet50Backbone, ResNet101Backbone
from rpn import AnchorGenerator, RPNHead, ProposalGenerator
from loss import class_loss_fn, bbox_loss_fn, mask_loss_fn
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
        
        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.bbox_loss_tracker = tf.keras.metrics.Mean(name="bbox_loss")
        self.mask_loss_tracker = tf.keras.metrics.Mean(name="mask_loss")
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

        # rois shape: [B, (None), 4]
        # features_class shape: [B, (None), sample_size, sample_size, feature_size]
        features_class = self.roi_align_class(
            feature_maps=[p2, p3, p4, p5], 
            rois=proposals
        )
        tf.print('features shape:', tf.shape(features_class))

        # class_logits shape: [B, (None), 81]
        class_logits = self.roi_classifier_head(
            features_class,   
            training=training
        )
        tf.print('class_logits shape:', tf.shape(class_logits))
        
        # bbox_deltas shape: [B, (None), 81*4]
        bbox_deltas = self.roi_bbox_head(
            features_class, 
            training=training
        )
        tf.print('bbox_deltas shape:', tf.shape(bbox_deltas))

        # rois shape: [B, (None), 4]
        # features_mask shape: [B, (None), sample_size, sample_size, feature_size]
        features_mask = self.roi_align_mask(
            feature_maps=[p2, p3, p4, p5], 
            rois=proposals
        )
        # masks shape: [B, (None), 28, 28, 80]
        masks = self.roi_mask_head(
            features_mask, 
            training=training
        )
        tf.print('masks shape:', tf.shape(masks))

        if os.environ.get("GPU_ENABLE", "FALSE") == "TRUE":
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                print("After backbone, GPU memory:", info)

        return (
            proposals, 
            class_logits, 
            bbox_deltas, 
            masks
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
        gt_classes = batch['category_id']

        # tf.print('gt_bboxes shape:', tf.shape(gt_bboxes))
        # tf.print('gt_masks shape:', tf.shape(gt_masks))
        # tf.print('gt_classes shape:', tf.shape(gt_classes))

        with tf.GradientTape() as tape:

            proposals, class_logits, bbox_deltas, masks = \
                self.call(images, sizes, training=True)
            
            # bboxes, labels, bbox_targets, mask_targets = \
            #     sample_and_assign_targets(
            #         proposals, 
            #         gt_boxes, 
            #         gt_classes, 
            #         gt_masks)
            # 
            # class_loss = class_loss_fn(class_logits, labels)
            # 
            # bbox_loss = bbox_loss_fn(
            #     bbox_deltas, 
            #     bbox_targets, 
            #     labels
            # )
            # 
            # mask_loss = mask_loss_fn(
            #     masks, 
            #     mask_targets, 
            #     labels
            # )

            # total_loss = class_loss + bbox_loss + mask_loss
        
        # grads = tape.gradient(total_loss, self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # self.class_loss_tracker.update_state(class_loss)
        # self.bbox_loss_tracker.update_state(bbox_loss)
        # self.mask_loss_tracker.update_state(mask_loss)
        # self.total_loss_tracker.update_state(total_loss)
        
        # return {
        #     'class_loss': self.class_loss_tracker.result(),
        #     'bbox_loss': self.bbox_loss_tracker.result(),
        #     'mask_loss': self.mask_loss_tracker.result(),
        #     'total_loss': self.total_loss_tracker.result(),
        # }
        return {
            'class_loss': 0.0,
            'bbox_loss': 0.0,
            'mask_loss': 0.0,
            'total_loss': 0.0,
        }
        
    def reset_metrics(self):
        self.class_loss_tracker.reset_states()
        self.bbox_loss_tracker.reset_states()
        self.mask_loss_tracker.reset_states()
        self.loss_tracker.reset_states()
