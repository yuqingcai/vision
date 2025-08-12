import tensorflow as tf
from tensorflow.keras import Model
from fpn import FPNGenerator
from backbone import ResNet50Backbone, ResNet101Backbone
from rpn import AnchorGenerator, RPNHead, ProposalGenerator
from loss_rpn_objectness import loss_rpn_objectness_fn
from loss_rpn_bbox import loss_rpn_box_reg_fn
from loss_classifier import loss_classifier_reg_fn
from loss_class_box import loss_class_box_reg_fn
from loss_mask import loss_mask_fn
from roi import ROIAlign, ROIClassifierHead, ROIBBoxHead, ROIMaskHead
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
        self.pre_nms_topk = 2000
        self.post_nms_topk = 1000
        self.roi_sample_ratio = 2
        
        self.fpn = FPNGenerator(feature_size=self.fpn_feature_size)

        self.anchor_generator = AnchorGenerator()

        self.rpn_head = RPNHead(
            anchors_per_location=len(self.anchor_ratios) * \
                len(self.anchor_scales), 
            feature_size=self.fpn_feature_size)
        
        self.proposal_generator = ProposalGenerator(
            pre_nms_topk=self.pre_nms_topk,
            post_nms_topk=self.post_nms_topk,
            nms_thresh=0.5,
            min_size=16
        )
        
        self.roi_align = ROIAlign(
            output_size=7, 
            sampling_ratio=self.roi_sample_ratio,
            feature_strides=self.fpn.strides(),
            feature_size=self.fpn_feature_size
        )

        self.roi_align_mask = ROIAlign(
            output_size=14, 
            sampling_ratio=self.roi_sample_ratio,
            feature_strides=self.fpn.strides(),
            feature_size=self.fpn_feature_size
        )

        self.classifier_head = ROIClassifierHead(
            num_classes=self.class_num + 1, # +1 for background class
            hidden_dim=1024
        )

        self.class_spec_bbox_head = ROIBBoxHead(
            num_classes=self.class_num + 1, # +1 for background class
            hidden_dim=1024
        )
        
        self.mask_head = ROIMaskHead(
            num_classes=self.class_num + 1, # +1 for background class
            conv_dim=256,
            num_convs=4,
            roi_output_size=self.roi_align_mask.output_size
        )
        
        self.loss_rpn_objectness_reg_tracker = tf.keras.metrics.Mean(
            name='loss_rpn_objectness_reg_tracker'
        )
        self.loss_rpn_box_reg_tracker = tf.keras.metrics.Mean(
            name='loss_rpn_box_reg_tracker'
        )
        self.loss_classifier_reg_tracker = tf.keras.metrics.Mean(
            name='loss_classifier_reg_tracker'
        )
        self.loss_class_spec_box_reg_tracker = tf.keras.metrics.Mean(
            name='loss_class_spec_box_reg_tracker'
        )
        self.loss_class_spec_mask_reg_tracker = tf.keras.metrics.Mean(
            name='loss_class_spec_mask_reg_tracker'
        )
        self.loss_total_tracker = tf.keras.metrics.Mean(
            name='loss_total_tracker'
        )
    

    def build_backbone(self, input_shape, batch_size, backbone_type):
        if backbone_type == 'resnet50':
            return ResNet50Backbone(input_shape, batch_size)
        elif backbone_type == 'resnet101':
            return ResNet101Backbone(input_shape, batch_size)
        else:
            raise ValueError("backbone_type must be 'resnet50' or 'resnet101'")
    
    
    def call(self, images, sizes, training=False):
        """rois shape: [B, (None), 4]
        features_class shape: [B, (None), 7, 7, 256]
        class_logits shape: [B, (None), 81]
        bbox_deltas shape: [B, (None), 81*4]
        features_mask shape: [B, (None), 14, 14, 256]
        masks shape: [B, (None), 28, 28, 80]
        sizes is not equal to images.shape[:2] because
        images is padded, image_sizes is the original size (i.e the 
        size of the content in a padded image).
        """
        
        c2, c3, c4, c5 = self.backbone(
            images, 
            training=training
        )

        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])

        # if training == False:
        #     print(f"model training == False -> c2: {c2.numpy()}")
        #     print(f"model training == False -> c3: {c3.numpy()}")
        #     print(f"model training == False -> c4: {c4.numpy()}")
        #     print(f"model training == False -> c5: {c5.numpy()}")
        #     print(f"model training == False -> p2: {p2.numpy()}")
        #     print(f"model training == False -> p3: {p3.numpy()}")
        #     print(f"model training == False -> p4: {p4.numpy()}")
        #     print(f"model training == False -> p5: {p5.numpy()}")

        anchors = self.anchor_generator(
            feature_maps=[p2, p3, p4, p5],
            strides=self.fpn.strides(),
            base_sizes=self.fpn.base_sizes(),
            ratios=self.anchor_ratios, 
            scales=self.anchor_scales,
            sizes=sizes
        )

        rpn_objectness_logits, \
            rpn_bbox_deltas = self.rpn_head(
            [p2, p3, p4, p5], 
            training=training
        )

        proposals, \
        rpn_objectness_logits, \
        rpn_bbox_deltas, \
        valid_mask = self.proposal_generator(
            anchors, 
            sizes, 
            rpn_objectness_logits, 
            rpn_bbox_deltas,
            training=training
        )


        # roi class and bbox head
        features = self.roi_align(
            feature_maps=[p2, p3, p4, p5], 
            rois=proposals, 
            valid_mask=valid_mask, 
            roi_size_pred=self.post_nms_topk,
            training=training
        )

        classifier_logits = self.classifier_head(
            features, 
            valid_mask, 
            training=training
        )

        class_bbox_deltas = self.class_spec_bbox_head(
            features, 
            valid_mask,
            training=training
        )
        
        # roi mask head
        features_mask = self.roi_align_mask(
            feature_maps=[p2, p3, p4, p5], 
            rois=proposals,
            valid_mask=valid_mask,
            roi_size_pred=self.post_nms_topk,
            training=training
        )

        class_masks = self.mask_head(
            features_mask, 
            valid_mask,
            training=training
        )
        
        return (
            proposals, 
            valid_mask,
            rpn_objectness_logits, 
            rpn_bbox_deltas,
            classifier_logits, 
            class_bbox_deltas, 
            class_masks, 
        )

    def compile(self, optimizer, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.optimizer = optimizer

    def build(self, input_shape):
        super().build(input_shape)
        self.gradient_accumulator = [
            tf.Variable(tf.zeros_like(var), trainable=False) \
                for var in self.trainable_variables
        ]

    @tf.function(reduce_retracing=True)
    def train_step(self, accumulation_steps, iterator):

        mean_loss_rpn_objectness_reg = 0.0
        mean_loss_rpn_box_reg = 0.0
        mean_loss_classifier_reg = 0.0
        mean_loss_class_spec_box_reg = 0.0
        mean_loss_mask = 0.0
        mean_loss_total = 0.0

        for i in tf.range(accumulation_steps):
            # Get the next batch of data
            batch = iterator.get_next()

            images = batch['image']     # shape: [B, H, W, C]
            sizes = batch['size']       # shape: [B, 2]
            gt_bboxes = batch['bbox']   # shape: [B, (None), 4]
            gt_masks = batch['mask']    # shape: [B, (None), H, W]
            gt_labels = batch['label']  # shape: [B, (None)]

            with tf.GradientTape() as tape:
                proposals, \
                    valid_mask, \
                    rpn_objectness_logits, \
                    rpn_bbox_deltas, \
                    classifier_logits, \
                    class_bbox_deltas, \
                    class_masks = self.call(images, sizes, training=True)

                loss_rpn_objectness_reg = loss_rpn_objectness_fn(
                    proposals,
                    valid_mask,
                    rpn_objectness_logits,
                    gt_bboxes
                )
                
                loss_rpn_box_reg = loss_rpn_box_reg_fn(
                    proposals,
                    valid_mask,
                    rpn_bbox_deltas,
                    gt_bboxes
                )

                loss_classifier_reg = loss_classifier_reg_fn(
                    proposals, 
                    valid_mask,
                    classifier_logits, 
                    gt_labels, 
                    gt_bboxes
                )

                loss_class_spec_box_reg = loss_class_box_reg_fn(
                    proposals, 
                    valid_mask, 
                    class_bbox_deltas, 
                    gt_labels, 
                    gt_bboxes
                )
                
                loss_mask = loss_mask_fn(
                    proposals, 
                    valid_mask, 
                    class_masks, 
                    gt_labels, 
                    gt_bboxes, 
                    gt_masks
                )

                # reduce losses by accumulation steps
                loss_total = (loss_rpn_objectness_reg + \
                    loss_rpn_box_reg + \
                    loss_classifier_reg + \
                    loss_class_spec_box_reg + \
                    loss_mask) / accumulation_steps

            # Backpropagation and accumulation
            grads = tape.gradient(loss_total, self.trainable_variables)
            for acc, grad in zip(self.gradient_accumulator, grads):
                if grad is not None:
                    acc.assign_add(grad)

            # accumulate loss
            mean_loss_rpn_objectness_reg += loss_rpn_objectness_reg
            mean_loss_rpn_box_reg += loss_rpn_box_reg
            mean_loss_classifier_reg += loss_classifier_reg
            mean_loss_class_spec_box_reg += loss_class_spec_box_reg
            mean_loss_mask += loss_mask
            mean_loss_total += (
                loss_rpn_objectness_reg + 
                loss_rpn_box_reg + 
                loss_classifier_reg + 
                loss_class_spec_box_reg + 
                loss_mask
            )

        # update weights
        self.optimizer.apply_gradients(
            zip(
                [acc.read_value() for acc in self.gradient_accumulator], 
                self.trainable_variables
            )
        )

        for acc in self.gradient_accumulator:
            acc.assign(tf.zeros_like(acc))

        # mean loss
        mean_loss_rpn_objectness_reg /= accumulation_steps
        mean_loss_rpn_box_reg /= accumulation_steps
        mean_loss_classifier_reg /= accumulation_steps
        mean_loss_class_spec_box_reg /= accumulation_steps
        mean_loss_mask /= accumulation_steps
        mean_loss_total /= accumulation_steps
        
        # Update metrics
        self.loss_rpn_objectness_reg_tracker.update_state(
            mean_loss_rpn_objectness_reg
        )
        self.loss_rpn_box_reg_tracker.update_state(
            mean_loss_rpn_box_reg
            )
        self.loss_classifier_reg_tracker.update_state(
            mean_loss_classifier_reg
        )
        self.loss_class_spec_box_reg_tracker.update_state(
            mean_loss_class_spec_box_reg
        )
        self.loss_class_spec_mask_reg_tracker.update_state(
            mean_loss_mask
        )
        self.loss_total_tracker.update_state(
            mean_loss_total
        )
        
        return {
            'loss_objectness': self.loss_rpn_objectness_reg_tracker.result(),
            'loss_rpn_box_reg': self.loss_rpn_box_reg_tracker.result(),
            'loss_class': self.loss_classifier_reg_tracker.result(),
            'loss_box_reg': self.loss_class_spec_box_reg_tracker.result(),
            'loss_mask': self.loss_class_spec_mask_reg_tracker.result(),
            'loss_total': self.loss_total_tracker.result(),
        }

    def reset_metrics(self):
        self.loss_rpn_objectness_reg_tracker.reset_state()
        self.loss_rpn_box_reg_tracker.reset_state()
        self.loss_classifier_reg_tracker.reset_state()
        self.loss_class_spec_box_reg_tracker.reset_state()
        self.loss_class_spec_mask_reg_tracker.reset_state()
        self.loss_total_tracker.reset_state()
