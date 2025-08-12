import os
import math
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pycocotools.coco import COCO

if __name__ == "__main__":

    # 加载数据集
    annotations_file = '../dataset/coco2017/annotations/instances_train2017.json'
    image_files = '../dataset/coco2017/train2017'
    register_coco_instances("train_dataset", 
        {}, 
        annotations_file, 
        image_files)


    # 计算迭代次数 max_iter，detectron2 的迭代次数是指所有 epoch 的迭代次数。
    # 例如，假设每个 epoch 有 1000 个 batch，epoch 数为 10，则 max_iter = 1000 * 10 = 10000。
    # 
    # 在深度学习训练中，最后一个 batch 不足 batch size 是很常见的情况。大多数
    # 框架（包括 Detectron2 和 PyTorch）会自动处理最后一个 batch 的图片数少于
    # 你设置的 IMS_PER_BATCH（即 batch size）的情况。
    # 
    coco = COCO(annotations_file)
    image_ids = coco.getImgIds()
    items_per_batch = 16
    epoch = 10
    max_iter = math.ceil(len(image_ids) / items_per_batch) * epoch

    print(f'image num: {len(image_ids)}')
    print(f'items per batch: {items_per_batch}')
    print(f'epoch num: {epoch}')
    print(f'iteration num: {max_iter}')

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # 这里初始化模型权重，事实上mask_rcnn_R_50_FPN_3x.yaml文件中已经指定
    # 使用detectron2://ImageNetPretrained/MSRA/R-50.pkl初始化模型权重，
    # 但这个权重只给 backbone（ResNet-50 主干部分），不包括 FPN 和 
    # Mask R-CNN 头部。
    # 这里的初始化包括了 backbone，FPN 和 Mask R-CNN 头部。
    # 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    cfg.SOLVER.IMS_PER_BATCH = items_per_batch
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.SOLVER.BASE_LR = 0.00025

    # COCO类别数，如自定义数据集需修改
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81

    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    