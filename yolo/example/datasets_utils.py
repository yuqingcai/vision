from ultralytics.data.converter import convert_coco

# 经过实验，coco 官方提供的标签压缩包解压后包含如下文件
# annotations
#       ｜-captions_train2017.json
#       ｜-instances_train2017.json
#       ｜-person_keypoints_train2017.json
#       ｜-captions_val2017.json
#       ｜-instances_val2017.json
#       ｜-person_keypoints_val2017.json
# 
#  captions_train2017.json, person_keypoints_train2017.json, 
#  captions_val2017.json, person_keypoints_val2017.json 这4个文件如果出现在 
#  labels_dir 中会发生 'bbox' KeyError，正确的做法是新建一个目录，把 
#  instances_train2017.json 和 instances_val2017.json 单独提取出来放在该目录中然后
#  进行转换。如下示例中 datasets/coco2017/coco/annotations 只包含
#  instances_train2017.json 和 instances_val2017.json
# 
convert_coco(
    "/home/qing/project/NNDL/datasets/coco2017/coco/annotations", 
    use_segments=True, 
    use_keypoints=False, 
    cls91to80=False
)
