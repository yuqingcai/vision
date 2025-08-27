import os
import torch
import math
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, \
    default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pycocotools.coco import COCO
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.events import EventWriter, get_event_storage

class CustomWriter(EventWriter):
    def write(self):
        storage = get_event_storage()
        histories = storage.histories()
        l = []
        for k, v in histories.items():
            l.append(f'{k}: {v}')
        print('CustomWriter:', ', '.join(l))            


class COCOTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name)

    # @classmethod
    # def test(cls, cfg, model, evaluators=None):
    #     print('Testing model...')

    # def build_writers(self):
    #     writers = super().build_writers()
    #     writers.append(CustomWriter())
    #     return writers


def setup(local_rank, args):
    # Register datasets
    train_annotations_file = '../../datasets/coco2017/annotations/instances_train2017.json'
    train_image_files = '../../datasets/coco2017/coco/images/train2017'
    val_annotations_file = '../../datasets/coco2017/annotations/instances_val2017.json'
    val_image_files = '../../datasets/coco2017/coco/images/val2017'

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

    # Calculate max_iter
    coco = COCO(train_annotations_file)
    image_ids = coco.getImgIds()
    items_per_batch = 16
    epoch = 36
    max_iter = math.ceil(len(image_ids) / items_per_batch) * epoch
    lr = 0.04
    
    print(
        f'local_rank: {local_rank}\n'
        f'image num: {len(image_ids)}\n'
        f'items per batch: {items_per_batch}\n'
        f'epoch num: {epoch}\n'
        f'iteration num: {max_iter}\n'
    )
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.DATASETS.TRAIN = ('train_dataset',)
    cfg.DATASETS.TEST = ('val_dataset',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = "cuda"
    # total batch size, will be split
    cfg.SOLVER.IMS_PER_BATCH = items_per_batch
    cfg.SOLVER.MAX_ITER = max_iter
    # WARMUP_ITERS 步数内的 lr 会很小，逐步恢复到设定的 lr
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.BASE_LR = lr / args.num_gpus
    cfg.SOLVER.GAMMA = 0.1
    # 分别在 131000 和 149000 次迭代调整学习率
    # cfg.SOLVER.STEPS = (131000, 149000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.SOLVER.LOG_PERIOD = 20
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.TEST.EVAL_PERIOD = 5000
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # 从 model_0129999.pth 处初始化权重继续训练
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0129999.pth")
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # )
    return cfg


def main(args):
    local_rank = comm.get_local_rank()
    cfg = setup(local_rank, args)
    trainer = COCOTrainer(cfg)

    print(
        f'optimizer: {type(trainer.optimizer).__name__}\n'
        f'GAMMA: {cfg.SOLVER.GAMMA}\n'
    )

    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# nohup python -u train.py --num-gpus 2 --num-machines 1 --machine-rank 0 --dist-url "auto" --resume > train.log 2>&1 &
# tensorboard --logdir output --host 10.0.0.7 --port 6006
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(args)
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="auto",
        args=(args,)
    )
    