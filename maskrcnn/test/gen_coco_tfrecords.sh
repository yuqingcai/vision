python official/vision/data/create_coco_tf_record.py \
  --image_dir=/home/project/NNDL/dataset/coco2017/train2017 \
  --annotation_file=/home/project/NNDL/dataset/coco2017/annotations/instances_train2017.json \
  --output_file=/home/project/NNDL/dataset/coco2017/tfrecords/train2017.tfrecord



python official/vision/data/create_coco_tf_record.py --logtostderr \
  --image_dir="/home/qing/project/NNDL/dataset/coco2017/train2017" \
  --object_annotations_file="/home/qing/project/NNDL/dataset/coco2017/annotations/instances_train2017.json" \
  --caption_annotations_file="/home/qing/project/NNDL/dataset/coco2017/annotations/captions_train2017.json" \
  --output_file_prefix="/home/qing/project/NNDL/dataset/coco2017/tfrecords/coco-train" \
  --include_masks="true" \
  --num_shards=100


python official/vision/data/create_coco_tf_record.py --logtostderr \
  --image_dir="/home/qing/project/NNDL/dataset/coco2017/val2017" \
  --object_annotations_file="/home/qing/project/NNDL/dataset/coco2017/annotations/instances_val2017.json" \
  --caption_annotations_file="/home/qing/project/NNDL/dataset/coco2017/annotations/captions_val2017.json" \
  --output_file_prefix="/home/qing/project/NNDL/dataset/coco2017/tfrecords/coco-val" \
  --include_masks="true" \
  --num_shards=100



# TFRecord feature keys: ['image/width', 'image/object/bbox/xmax', 'image/object/class/label', 'image/object/class/text', 'image/caption', 'image/filename', 'image/object/bbox/xmin', 'image/encoded', 'image/object/area', 'image/object/bbox/ymin', 'image/height', 'image/key/sha256', 'image/object/is_crowd', 'image/object/bbox/ymax', 'image/format', 'image/source_id']


#  keys_to_features = {
#           'image/encoded':
#               tfrecord_lib.convert_to_feature(image.numpy()),
#           'image/filename':
#                tfrecord_lib.convert_to_feature(filename.encode('utf8')),
#           'image/format':
#               tfrecord_lib.convert_to_feature('jpg'.encode('utf8')),
#           'image/height':
#               tfrecord_lib.convert_to_feature(image_info['height']),
#           'image/width':
#               tfrecord_lib.convert_to_feature(image_info['width']),
#           'image/source_id':
#               tfrecord_lib.convert_to_feature(str(image_info['id']).encode('utf8')),
#           'image/object/bbox/xmin':
#               tfrecord_lib.convert_to_feature(data['xmin']),
#           'image/object/bbox/xmax':
#               tfrecord_lib.convert_to_feature(data['xmax']),
#           'image/object/bbox/ymin':
#               tfrecord_lib.convert_to_feature(data['ymin']),
#           'image/object/bbox/ymax':
#               tfrecord_lib.convert_to_feature(data['ymax']),
#           'image/object/class/text':
#               tfrecord_lib.convert_to_feature(data['category_names']),
#           'image/object/class/label':
#               tfrecord_lib.convert_to_feature(data['category_id']),
#           'image/object/is_crowd':
#               tfrecord_lib.convert_to_feature(data['is_crowd']),
#           'image/object/area':
#               tfrecord_lib.convert_to_feature(data['area'], 'float_list'),
#           'image/object/mask':
#               tfrecord_lib.convert_to_feature(data['encoded_mask_png'])
#       }