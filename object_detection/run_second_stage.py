import os
import os.path as osp
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import numpy as np
import functools
import bisect
import json
import gzip
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from google.protobuf import text_format
from utils import label_map_util
from utils import visualization_utils as vis_util

from object_detection.builders import model_builder
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.core import box_list
from object_detection.core import box_list_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
flags.DEFINE_string('model', '',
                     'The model on which to run the first stage.')
flags.DEFINE_integer('batch', 1,
                     'The model on which to run the first stage.')
FLAGS = flags.FLAGS



####################################################################
# MAIN PARAMETERS
####################################################################
MODELS = {
  'base_models_dir':  '/home/pierre/projects/deep_learning/foodDetectionAPI/models', 
  'base_configs_dir': 'samples/configs', 
  'frcnn': {
      'config':  'faster_rcnn_resnet101_foodinc.config', 
      'ckp_dir': 'frcnn', 
      'features_to_crop_shape': [1, None, None, 1024], 
  }, 
  'inception': { 
      'config':  'faster_rcnn_inception_resnet_v2_atrous_foodinc.config', 
      'ckp_dir': 'inception', 
      'features_to_crop_shape': [1, None, None, 1088], 
  }, 
  'rfcn': {
      'config':  'rfcn_resnet101_foodinc.config', 
      'ckp_dir': 'rfcn', 
      'features_to_crop_shape': [1, None, None, 1024], 
  }, 
  'ssd': {
      'config':  'ssd_mobilenet_v1_foodinc.config', 
      'ckp_dir': 'ssd', 
      'features_to_crop_shape': [1, None, None, 1024], 
  }, 
}

DATASET = {
  'foodinc': {
      'nb_classes': 67, 
      'base_dir': '/home/pierre/projects/datasets/MacFoodinc', 
      'images_dir': 'Images', 
      'annotations_dir': 'Annotations', 
      'labels_path': 'infos/foodinc_label_map.pbtxt',
      'list_path': 'ImageSets/all.txt', 
      'max_images': -1, 
      'image_extension': 'jpg', 
  }
}


CONFIDENCE_THRESH = 0.5
model = FLAGS.model
batch = FLAGS.batch
dataset = DATASET['foodinc']

images_dir = osp.join(dataset['base_dir'], dataset['images_dir'])
annotations_dir = osp.join(dataset['base_dir'], dataset['annotations_dir'])


####################################################################
# REQUIREMENTS
####################################################################
paths_check = {
    dataset['base_dir']: [ { 'name': '',                     'type': 'dir', }, 
                           { 'name': dataset['labels_path'], 'type': 'file', },
                           { 'name': dataset['images_dir'],  'type': 'dir', }, ], 
    MODELS['base_models_dir']: [], 
    MODELS['base_configs_dir']: [], 
}
paths_check[MODELS['base_configs_dir']].append({ 'name': MODELS[model]['config'],  'type': 'file' } )
paths_check[MODELS['base_models_dir']].append( { 'name': MODELS[model]['ckp_dir'], 'type': 'dir' } )
for k, v in paths_check.iteritems():
    for p in v:
        if (p['type'] == 'dir'  and not osp.exists(osp.join(k, p['name']))) or \
           (p['type'] == 'file' and not osp.isfile(osp.join(k, p['name']))):
            print ' '.join(['No', p['name'], 'given, please add / create it.'])
            raise SystemExit

if not osp.exists(annotations_dir):
    print 'Create the Annotations directory where to save the annotations.'
    os.makedirs(annotations_dir)


####################################################################
# Get the list of images to annotate
####################################################################
# List of images to annotate
list_path = osp.join(dataset['base_dir'], dataset['list_path'])
list_images_names = [line.rstrip('\n') for line in open(list_path)]
if dataset['max_images'] != -1 and dataset['max_images'] < len(list_images_names):
    list_images_names = list_images_names[:dataset['max_images']]

# Filtered images that should not be annotated
filtered_list = []
errors = { 'annotated': [], 'computed': [], 'notafile': [], 'missing': [] }
for img_name in list_images_names:
  if osp.isfile(osp.join(annotations_dir, img_name + '.txt')):
    errors['annotated'].append(img_name)
  elif osp.isfile(osp.join(annotations_dir, img_name + '.' + model + '.2')):
    errors['computed'].append(img_name)
  elif not osp.isfile(osp.join(images_dir, '{}.{}'.format(img_name, dataset['image_extension']))):
    errors['notafile'].append(img_name)
  elif not osp.isfile(osp.join(annotations_dir, img_name + '.1')):
    errors['missing'].append(img_name)
  elif not osp.isfile(osp.join(annotations_dir, img_name + '.' + model + '.1.prop')):
    errors['missing'].append(img_name)
  else:
    filtered_list.append(img_name)

if len(errors['annotated']) > 0:
  print len(errors['annotated']), "images are already annotated, will be skipped"
if len(errors['computed']) > 0:
  print len(errors['computed']), "images are already computed, will be skipped"
if len(errors['notafile']) > 0:
  print len(errors['notafile']), "images are in the list, but doesn't exist....."
if len(errors['missing']) > 0:
  print len(errors['missing']), "images have at least one missing annotation for this ensemble."
if len(filtered_list) == 0:
  print "No images to annotate."
  raise SystemExit


####################################################################
# Create the tensorflow methods for the scenarii
####################################################################
def classify_proposals_only(rpn_features, image_shape, boxes, num_proposals, isRfcn):
    # Run the second stage only and return the result
    detections = built_model.classify_proposals_only(rpn_features, image_shape, boxes, num_proposals, isRfcn)
    
    # Restore lost parameters between stages
    tmp_dict = { 'image_shape': image_shape }
    tmp_dict.update(detections)
    detections = tmp_dict
    
    return detections


####################################################################
# STORER
####################################################################
# Configs file for the model
model_config_file = osp.join(MODELS['base_configs_dir'], MODELS[model]['config'])
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.gfile.GFile(model_config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
model_config = pipeline_config.model
built_model = model_builder.build(model_config, False)


####################################################################
# The inputs
####################################################################
# Input for the second stage: the selected proposals
rpn_shape = MODELS[model]['features_to_crop_shape']
rpn_feat_tensor = tf.placeholder(tf.float32, shape=rpn_shape, name='rpn_feat')
prop_boxes_tensor = tf.placeholder(tf.float32, shape=[1, 300, 4], name='prop_boxes')
num_prop_tensor = tf.placeholder(tf.int32, shape=[1], name='num_prop')
image_shape_tensor = tf.placeholder(tf.int32, shape=[4], name='img_shape')


####################################################################
# Preparation
####################################################################
# Helper for the session
def create_session_for_model(model):
  variables_to_restore = tf.global_variables()
  saver = tf.train.Saver(variables_to_restore)
  sess = tf.Session('', graph=tf.get_default_graph())
  sess.run(tf.global_variables_initializer())
  path_to_ckpt = osp.join(MODELS['base_models_dir'], MODELS[model]['ckp_dir'])
  path_to_latest_ckpt = tf.train.latest_checkpoint(path_to_ckpt)
  latest_checkpoint = path_to_latest_ckpt
  saver.restore(sess, latest_checkpoint)
  return sess


####################################################################
# Start to annotate
####################################################################

# Do the batch
batch_list = filtered_list[:batch]

# First stage scenario
detections_tensor = classify_proposals_only(rpn_feat_tensor, image_shape_tensor, 
                                            prop_boxes_tensor, num_prop_tensor,
                                            (model == 'rfcn'))
# Create the session
sess = create_session_for_model(model)

# For each image in the batch
for img in batch_list:

  # Restore the ensemble
  proposals_path = osp.join(annotations_dir, img + '.1')
  loaded_results = json.load(open(proposals_path))

  model_characts_path = osp.join(annotations_dir, img + '.' + model + '.1.prop')
  loaded_characts = None
  with gzip.GzipFile(model_characts_path, 'r') as f:
    json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')
    loaded_characts = json.loads(json_str)

  results_proposals_dict = {
    'proposal_boxes_normalized': np.array(loaded_results['proposal_boxes_normalized']), 
    'proposal_scores': np.array(loaded_results['proposal_scores']), 
    'num_proposals': np.array(loaded_results['num_proposals']), 
  }

  model_characts = {
    'prediction_dict': {
        'image_shape': np.array(loaded_characts['prediction_dict']['image_shape']), 
        'rpn_features_to_crop': np.array(loaded_characts['prediction_dict']['rpn_features_to_crop']), 
    }, 
  }

  # Run the second stage
  detections = sess.run(
    detections_tensor, 
    feed_dict={rpn_feat_tensor:    model_characts['prediction_dict']['rpn_features_to_crop'],
               image_shape_tensor: model_characts['prediction_dict']['image_shape'],
               prop_boxes_tensor:  results_proposals_dict['proposal_boxes_normalized'],
               num_prop_tensor:    results_proposals_dict['num_proposals']}
  )

  result_to_save = {
    'class_predictions_with_background': detections['class_predictions_with_background'].tolist(), 
    'num_proposals': detections['num_proposals'].tolist(), 
    'image_shape': detections['image_shape'].tolist(), 
    'refined_box_encodings': detections['refined_box_encodings'].tolist(), 
    'proposal_boxes': detections['proposal_boxes'].tolist(), 
  }
  """
  result_to_save = {
    'detection_classes': detections['detection_classes'].tolist(), 
    'detection_boxes': detections['detection_boxes'].tolist(), 
    'detection_scores': detections['detection_scores'].tolist(), 
    'num_detections': detections['num_detections'].tolist(), 
  }
  """
  tmp_annotation_file = osp.join(annotations_dir, img + '.' + model + '.2')
  json.dump(result_to_save, open(tmp_annotation_file, 'w'))
  os.system("rm " + model_characts_path)

sess.close()


""" Output format:

{
  'detection_classes': array([], dtype=float32), # (1, 300)
  'detection_boxes': array([], dtype=float32),   # (1, 300, 4)
  'detection_scores': array([], dtype=float32),  # (1, 300)
  'num_detections': array([], dtype=float32)     # (1,)
}

"""





