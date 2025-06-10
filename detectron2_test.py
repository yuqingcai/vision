import cv2
import torch
from PIL import Image
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torchvision
import matplotlib.pyplot as plt


def detect_objects():
    # 设置配置
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 置信度阈值
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cuda"
    # 创建预测器
    predictor = DefaultPredictor(cfg)

    # 读取图片
    image_path = "../dataset/coco2017/test2017/000000000001.jpg"
    image = cv2.imread(image_path)

    # 推理
    outputs = predictor(image)

    # 可视化
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 显示结果
    # 这里不用cv2.imshow，在 windows 的 WSL 环境下会有问题。一般 WSL 上
    # 的 opencv 使用的是opencv-python-headless，不能显示图像,如果需要在
    # 本地显示图像，可以使用 matplotlib 或者直接保存图像。macOS 和 Linux
    # 上可以使用 cv2.imshow 显示图像。
    # cv2.imshow("Result", out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("result.jpg", out.get_image()[:, :, ::-1])
    print("Result saved to result.jpg")
    plt.imshow(out.get_image())
    plt.axis('off')
    plt.savefig("result.jpg")  # 可选：保存
    plt.show()


def show_version():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(detectron2.__version__)
    # print(detectron2._C)
    print(torchvision.__version__)


if __name__ == "__main__":
    show_version()
    detect_objects()


