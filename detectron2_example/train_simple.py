import os
import torch
import math
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pycocotools.coco import COCO
from detectron2.evaluation import COCOEvaluator

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 加载数据集
    train_annotations_file = '../../dataset/coco2017/annotations/instances_train2017.json'
    train_image_files = '../../dataset/coco2017/train2017'

    val_annotations_file = '../../dataset/coco2017/annotations/instances_val2017.json'
    val_image_files = '../../dataset/coco2017/val2017'

    register_coco_instances(
        "train_dataset", 
        {}, 
        train_annotations_file, 
        train_image_files
    )

    register_coco_instances(
        "val_dataset", 
        {}, 
        val_annotations_file, 
        val_image_files
    )

    # 计算迭代次数 max_iter，detectron2 的迭代次数是指所有 epoch 的迭代次数。
    # 例如，假设每个 epoch 有 1000 个 batch，epoch 数为 10，
    # max_iter = 1000 * 10 = 10000。
    # 
    # 在深度学习训练中，最后一个 batch 不足 batch size 是很常见的情况。大多数
    # 框架（包括 Detectron2 和 PyTorch）会自动处理最后一个 batch 的图片数少于
    # 你设置的 IMS_PER_BATCH（即 batch size）的情况。
    # 
    coco = COCO(train_annotations_file)
    image_ids = coco.getImgIds()
    items_per_batch = 20
    epoch = 1
    max_iter = math.ceil(len(image_ids) / items_per_batch) * epoch

    print(f'image num: {len(image_ids)}')
    print(f'items per batch: {items_per_batch}')
    print(f'epoch num: {epoch}')
    print(f'iteration num: {max_iter}')

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.DATASETS.TRAIN = ('train_dataset',)
    cfg.DATASETS.TEST = ('val_dataset',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = "cuda"

    # 这里初始化模型权重，事实上mask_rcnn_R_50_FPN_3x.yaml文件中已经指定
    # 使用detectron2://ImageNetPretrained/MSRA/R-50.pkl初始化模型权重，
    # 但这个权重只给 backbone（ResNet-50 主干部分），不包括 FPN 和 
    # Mask R-CNN 头部。
    # 这里的初始化包括了 backbone，FPN 和 Mask R-CNN 头部。
    # 
    # 这里的 model_zoo.get_checkpoint_url 有问题！！！
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # )
    cfg.MODEL.WEIGHTS = weights=model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    # '/home/qing/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    
    cfg.SOLVER.IMS_PER_BATCH = items_per_batch
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.TEST.EVAL_PERIOD = 1000  # 每1000次iter评估一次val集

    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
