#!/usr/bin/env python
import torch
from typing import Dict, List, Tuple
from torch import Tensor, nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.export import (
    scripting_with_instances,
    TracingAdapter,
)
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.structures import Boxes
from detectron2.projects.point_rend import add_pointrend_config

def export_scripting(torch_model):
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }

    class ScriptableAdapter(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.eval()

        def forward(self, inputs: Tuple[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
            # Detectron2 batch输入格式
            instances = self.model.inference(inputs, do_postprocess=False)
            # 转为 list[dict]，每个dict包含上述fields
            return [i.get_fields() for i in instances]

    adapter = ScriptableAdapter(torch_model)
    ts_model = scripting_with_instances(adapter, fields)
    torch.jit.save(ts_model, 'mask_rcnn_R_50_FPN_3x_scripting.pt')


# deprecated
def export_tracing(torch_model):
    image = torch.rand(3, 800, 800)
    inputs = [{"image": image}]

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=True)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    ts_model = torch.jit.trace(traceable_model, (image,))
    torch.jit.save(ts_model, 'mask_rcnn_R_50_FPN_3x_tracing.pt')


def mask_rcnn_mdoel():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()

    return model


def pointrend_rcnn_model():
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file("/Volumes/VolumeEXT/Project/NNDL/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    # cfg.MODEL.ROI_MASK_HEAD.FC_DIM = 1024
    cfg.freeze()

    # 模型构建
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()


def main() -> None:
    torch._C._jit_set_bailout_depth(1)

    # model = mask_rcnn_mdoel()
    model = pointrend_rcnn_model()
    export_scripting(model)

    # deprecated
    # export_tracing(torch_model)


if __name__ == "__main__":
    main()
