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
import time
import cPickle as pickle

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
flags.DEFINE_string('ensemble', '',
                     'The ensemble of models (space as delimiter).')
FLAGS = flags.FLAGS



####################################################################
# MAIN PARAMETERS
####################################################################
MODELS = {
  #'base_models_dir':  '/home/pierre/projects/deep_learning/foodDetectionAPI/models', 
  'base_models_dir':  '/home/finc/tf_playground/results_models',  
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
      #'base_dir': '/home/pierre/projects/datasets/MacFoodinc', 
      'base_dir': '/home/finc/final_macFoodinc/0_25000', 
      'images_dir': 'Images', 
      'annotations_dir': 'Annotations', 
      'labels_path': 'infos/foodinc_label_map.pbtxt',
      'list_path': 'ImageSets/all.txt', 
      'max_images': -1, 
      'image_extension': 'jpg', 
  }
}


CONFIDENCE_THRESH = 0.5
# Only needed for the postprocessing function, which is similar for frcnn /
# rfcn / inception. Check that if you use another model
model_to_build = 'frcnn'
ensemble = FLAGS.ensemble.split()
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
for model in ensemble:
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
errors = { 'annotated': [], 'computed': [], 'missing': [] }
for img_name in list_images_names:
  if osp.isfile(osp.join(annotations_dir, img_name + '.txt')):
    errors['annotated'].append(img_name)
  elif osp.isfile(osp.join(annotations_dir, img_name + '.2')):
    errors['computed'].append(img_name)
  else:
    ok = [osp.isfile(osp.join(annotations_dir, img_name + '.' + m + '.2'))
                 for m in ensemble]
    if False in ok:
      errors['missing'].append(img_name)
    else:
      filtered_list.append(img_name)

if len(errors['annotated']) > 0:
  print len(errors['annotated']), "images are already annotated, will be skipped"
if len(errors['computed']) > 0:
  print len(errors['computed']), "images are already computed, will be skipped"
if len(errors['missing']) > 0:
  print len(errors['missing']), "images have at least one missing annotation for this ensemble."
if len(filtered_list) == 0:
  print "No images to annotate."
  raise SystemExit


####################################################################
# Create the tensorflow methods for the scenarii
####################################################################
def merge_detections(ensemble, detections):
    # {
    #   'class_predictions_with_background': <shape=(300, 68) dtype=float32 (tf.Tensor 'Squeeze_1:0')>, 
    #   'num_proposals': <shape=(1,) dtype=int32 (tf.Tensor 'num_prop:0')>, 
    #   'image_shape': <shape=(4,) dtype=int32 (tf.Tensor 'img_shape:0')>, 
    #   'refined_box_encodings': <shape=(300, 67, 4) dtype=float32 (tf.Tensor 'Squeeze:0')>, 
    #   'proposal_boxes': <shape=(1, 300, 4) dtype=float32 (tf.Tensor 'map/TensorArrayStack/TensorArrayGatherV3:0')>
    # }
    new_classes_predictions = detections[ensemble[0]]['class_predictions_with_background']
    new_refined_boxes = detections[ensemble[0]]['refined_box_encodings']
    new_proposals_boxes = detections[ensemble[0]]['proposal_boxes']
    for model in ensemble[1:]:
      new_classes_predictions += detections[model]['class_predictions_with_background']
      new_refined_boxes += detections[model]['refined_box_encodings']
      new_proposals_boxes += detections[model]['proposal_boxes']
    new_classes_predictions /= float(len(ensemble))
    new_refined_boxes /= float(len(ensemble))
    new_proposals_boxes /= float(len(ensemble))

    return {
      'class_predictions_with_background': new_classes_predictions, 
      'refined_box_encodings': new_refined_boxes, 
      'proposal_boxes': new_proposals_boxes, 
      'num_proposals': detections[ensemble[0]]['num_proposals'], 
      'image_shape': detections[ensemble[0]]['image_shape'], 
    }

def postprocess_detections(detections):
    # Post processing
    postprocessed_detections = built_model.postprocess(detections)

    # Returns them
    return postprocessed_detections


####################################################################
# STORER
####################################################################
# Configs file for the model
model_config_file = osp.join(MODELS['base_configs_dir'], MODELS[model_to_build]['config'])
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.gfile.GFile(model_config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
model_config = pipeline_config.model
built_model = model_builder.build(model_config, False)


####################################################################
# The inputs
####################################################################
# Input for the postprocessing
class_predictions_tensor = tf.placeholder(tf.float32, shape=[300, dataset['nb_classes']+1], name='class_predictions')
num_proposals_tensor = tf.placeholder(tf.int32, shape=[1, ], name='num_proposals')
image_shape_tensor = tf.placeholder(tf.int32, shape=[4, ], name='image_shape')
refined_box_encodings_tensor = tf.placeholder(tf.float32, shape=[300, dataset['nb_classes'], 4], name='refined_box_encodings')
proposal_boxes_tensor = tf.placeholder(tf.float32, shape=[1, 300, 4], name='proposal_boxes')


####################################################################
# Preparation
####################################################################
# Getting an image
def get_image(images_dir, image_name):
  path = osp.join(images_dir, '{}.{}'.format(image_name, dataset['image_extension']))
  return Image.open(path)

# Helper for the session
def create_session_for_model(model_to_build):
  sess = tf.Session('', graph=tf.get_default_graph())
  return sess


####################################################################
# Start to annotate
####################################################################

# No batch for the ensemble, process everything it cans
batch_list = filtered_list

# Create the dict of tensors
tensor_dict = {
    'class_predictions_with_background': class_predictions_tensor, 
    'refined_box_encodings': refined_box_encodings_tensor, 
    'proposal_boxes': proposal_boxes_tensor, 
    'num_proposals': num_proposals_tensor, 
    'image_shape': image_shape_tensor, 
}

# First stage scenario
ensembled_detections_tensor = postprocess_detections(tensor_dict)
# Create the session
sess = create_session_for_model(model)

# For each image in the batch
for img in batch_list:

  results_detections_dict = {}
  for model in ensemble:
    annotation_path = osp.join(annotations_dir, img + '.' + model + '.2')
    with open(annotation_path, 'rb') as f:
      loaded_results = pickle.load(f)
    """
    results_detections_dict[model] = {
      'detection_classes': np.array(loaded_results['detection_classes']), 
      'detection_boxes': np.array(loaded_results['detection_boxes']), 
      'detection_scores': np.array(loaded_results['detection_scores']), 
      'num_detections': np.array(loaded_results['num_detections']), 
    }
    """
    results_detections_dict[model] = {
      'class_predictions_with_background': np.array(loaded_results['class_predictions_with_background']), 
      'num_proposals': np.array(loaded_results['num_proposals']), 
      'image_shape': np.array(loaded_results['image_shape']), 
      'refined_box_encodings': np.array(loaded_results['refined_box_encodings']), 
      'proposal_boxes': np.array(loaded_results['proposal_boxes']), 
    }

  # For each image, merge the results of all the models
  ensembled_detections = merge_detections(ensemble, results_detections_dict)
  
  # Run the second stage
  postprocessed_detections = sess.run(
    ensembled_detections_tensor, 
    feed_dict={class_predictions_tensor:     ensembled_detections['class_predictions_with_background'],
               refined_box_encodings_tensor: ensembled_detections['refined_box_encodings'],
               proposal_boxes_tensor:        ensembled_detections['proposal_boxes'],
               num_proposals_tensor:         ensembled_detections['num_proposals'], 
               image_shape_tensor:           ensembled_detections['image_shape']}
  )

  # The image, for global characteristics
  image = get_image(images_dir, img)
  (im_width, im_height) = image.size
  
  # General post processing
  label_id_offset = 1
  scores = np.squeeze(postprocessed_detections['detection_scores'], axis=0)
  boxes = np.squeeze(postprocessed_detections['detection_boxes'], axis=0)
  classes = np.squeeze(postprocessed_detections['detection_classes'], axis=0) + label_id_offset
  
  confident_indices = scores > CONFIDENCE_THRESH
  scores = scores[confident_indices]
  boxes = boxes[confident_indices]
  classes = classes[confident_indices]

  absolute_boxes = []
  for box in boxes:
    y_min, x_min, y_max, x_max = box
    y_min = im_height * y_min
    y_max = im_height * y_max
    x_min = im_width * x_min
    x_max = im_width * x_max
    absolute_boxes.append([x_min, y_min, x_max, y_max])
  
  with open(osp.join(annotations_dir, img + '.txt'), 'a') as f:
    for i in range(len(absolute_boxes)):
      f.write(' '.join([str(int(classes[i])), 
                        ' '.join([str(absolute_boxes[i][0]), str(absolute_boxes[i][1]), 
                                  str(absolute_boxes[i][2]), str(absolute_boxes[i][3])]), 
                        str(scores[i])]) + '\n')

  for model in ensemble:
    annotation_path = osp.join(annotations_dir, img + '.' + model + '.2')
    os.system("rm " + annotation_path)
  if osp.isfile(osp.join(annotations_dir, img + '.1')):
    os.system("rm " + osp.join(annotations_dir, img + '.1'))

sess.close()

