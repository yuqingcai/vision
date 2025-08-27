import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


"""validate_trained_weights 函数加载预训练参数/模型后直接进行评估
annotations_file: 验证集标签文件
image_files: 验证集图片目录
weights: 预训练权重文件 .pth/.pkl

需要注意。weights 参数使用 .pth/.pkl 文件，config_file 参数使用 .yaml 文件。
.pth/.pkl 文件必须使用和 config_file 一致的 .yaml文件训练得到，比如当 config_file 是 
"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" 时，weights 也应该是由使用
该 .yaml 文件训练得到的模型权重，否则模型在加载参数时会发生错误。
"""
def validate(
    annotations_file,
    image_files,
    config_file,
    weights
):
    register_coco_instances(
        "val_dataset", 
        {}, 
        annotations_file, 
        image_files
    )
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = "cuda"

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    evaluator = COCOEvaluator(
        "val_dataset", 
        cfg, 
        False, 
        output_dir="./output/"
    )
    val_loader = build_detection_test_loader(cfg, "val_dataset")
    results = inference_on_dataset(model, val_loader, evaluator)
    print(results)
    

if __name__ == "__main__":
    # 直接使用预训练模型进行验证
    # 注意：这里的权重文件需要是经过训练的模型权重
    # validate(
    #     annotations_file='../../dataset/coco2017/annotations/instances_val2017.json',
    #     image_files='../../dataset/coco2017/val2017',
    #     config_file=model_zoo.get_config_file(
    #         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    #     ),
    #     weights=model_zoo.get_checkpoint_url(
    #         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    #     )
    # )

    # 验证 model_final.pth 权重文件
    validate(
        annotations_file='../../dataset/coco2017/annotations/instances_val2017.json',
        image_files='../../dataset/coco2017/val2017',
        config_file=model_zoo.get_config_file(
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        ),
        weights='./output/model_final.pth'
    )
    