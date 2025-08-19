class COCOTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False)

def setup(local_rank, args):
    # Register datasets
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

    # Calculate max_iter
    coco = COCO(train_annotations_file)
    image_ids = coco.getImgIds()
    items_per_batch = 16
    epoch = 36
    max_iter = math.ceil(len(image_ids) / items_per_batch) * epoch

    # only main process prints
    print(f'local_rank: {local_rank}')
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
    cfg.DATALOADER.NUM_WORKERS = 4 # 这里设置成4~8可提升速度
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # )
    # total batch size, will be split
    cfg.SOLVER.IMS_PER_BATCH = items_per_batch
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.TEST.EVAL_PERIOD = 4000
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main(args):
    local_rank = comm.get_local_rank()
    cfg = setup(local_rank, args)
    trainer = COCOTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# nohup python -u train_dual.py --num-gpus 2 --num-machines 1 --machine-rank 0 --dist-url "auto" --resume True > train.log 2>&1 &
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="auto",
        args=(args,)
    )