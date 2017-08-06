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



####################################################################
# MAIN PARAMETERS
####################################################################
MODELS = {
  'base_models_dir':  '/home/pierre/projects/deep_learning/foodDetectionAPI/models', 
  'base_configs_dir': 'samples/configs', 
  'frcnn': {
      'config':  'faster_rcnn_resnet101_foodinc.config', 
      'ckp_dir': 'frcnn', 
  }, 
  'inception': { 
      'config':  'faster_rcnn_inception_resnet_v2_atrous_foodinc.config', 
      'ckp_dir': 'inception', 
  }, 
  'ssd': {
      'config':  'ssd_mobilenet_v1_foodinc.config', 
      'ckp_dir': 'ssd', 
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


DEBUG_MODE = False
CONFIDENCE_THRESH = 0.5
ensemble = ['frcnn',]
dataset = DATASET['foodinc']

images_dir = osp.join(dataset['base_dir'], dataset['images_dir'])
annotations_dir = osp.join(dataset['base_dir'], dataset['annotations_dir'])

# Test parameters
params_first_stage = { 'frcnn': 2, 'inception': 3, 'ssd': 4 }
params_second_stage = { 'frcnn': 10, 'inception': 5, 'ssd': 2 }


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
else:
    print 'Warning, the annotations directory already exists, will skip the images already annotated if any.'


####################################################################
# STORER
####################################################################
# Dictionary for each component of the ensemble
all_ = {}
for model in ensemble:
    all_[model] = {}

    # Configs file for each model into all_[model]['model']
    model_config_file = osp.join(MODELS['base_configs_dir'], MODELS[model]['config'])
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(model_config_file, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    model_config = pipeline_config.model
    all_[model]['model'] = model_builder.build(model_config, False)


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
errors = { 'annotated': [], 'notafile': [] }
for img_name in list_images_names:
  if osp.isfile(osp.join(annotations_dir, img_name + '.txt')):
    errors['annotated'].append(img_name)
  elif not osp.isfile(osp.join(images_dir, '{}.{}'.format(img_name, dataset['image_extension']))):
    errors['notafile'].append(img_name)
  else:
    filtered_list.append(img_name)

if len(errors['annotated']) > 0:
  print len(errors['annotated']), "images are already annotated, will be skipped"
if len(errors['notafile']) > 0:
  print len(errors['notafile']), "images are in the list, but doesn't exist....."
if len(filtered_list) == 0:
  print "No images to annotate."
  raise SystemExit

####################################################################
# Create the tensorflow methods for the scenarii
####################################################################
def propose_boxes_only(model, image_tensor):
    # For each model, compute the proposals
    preprocessed_image = model.preprocess(tf.to_float(image_tensor))

    # Run the first stage only (proposals)
    (prediction_dict, proposal_boxes_normalized, proposal_scores, num_proposals) = \
        model.propose_boxes_only(preprocessed_image)

    # Returns them
    return {
        'prediction_dict': prediction_dict, 
        'proposal_boxes_normalized': proposal_boxes_normalized, 
        'proposal_scores': proposal_scores, 
        'num_proposals': num_proposals, 
    }


def merge_proposals(ensemble, proposals):
    scores = proposals[ensemble[0]]['proposal_scores']
    boxes = proposals[ensemble[0]]['proposal_boxes_normalized']

    for model in ensemble[1:]:
      for batch in range(scores.shape[0]):
        for ix, s in enumerate(proposals[model]['proposal_scores'][batch]):
          if s > scores[batch][-1]:
            scores[batch] = scores[batch][::-1]
            index = len(scores[batch]) - bisect.bisect(scores[batch], s)
            scores[batch] = scores[batch][::-1]
            scores[batch].insert(index, s)
            boxes[batch].insert(index, proposals[model]['proposal_boxes_normalized'][batch][ix])
            scores[batch] = scores[batch][:-1]
            boxes[batch] = boxes[batch][:-1]
          else:
            break

    tmp_dict = proposals[ensemble[0]]
    tmp_dict.update({'proposal_scores': scores, 'proposal_boxes_normalized': boxes})
    return tmp_dict


def classify_proposals_only(model, old_pred_dict, rpn_features, image_shape, boxes, num_proposals):
    # Run the second stage only and return the result
    detections = model.classify_proposals_only(rpn_features, image_shape, boxes, num_proposals)
    
    # Restore lost parameters between stages
    tmp_dict = old_pred_dict
    tmp_dict.update(detections)
    detections = tmp_dict
    
    # Post processing
    postprocessed_detections = model.postprocess(detections)

    # Returns them
    return postprocessed_detections


def merge_detections(ensemble, detections):
    scores = detections[ensemble[0]]['detection_scores']
    boxes = detections[ensemble[0]]['detection_boxes']
    classes = detections[ensemble[0]]['detection_classes']

    for model in ensemble[1:]:
      for batch in range(scores.shape[0]):
        for ix, s in enumerate(detections[model]['detection_scores'][batch]):
          if s > scores[batch][-1]:
            scores[batch] = scores[batch][::-1]
            index = len(scores[batch]) - bisect.bisect(scores[batch], s)
            scores[batch] = scores[batch][::-1]
            scores[batch].insert(index, s)
            boxes[batch].insert(index, detections[model]['detection_boxes'][batch][ix])
            classes[batch].insert(index, detections[model]['detection_classes'][batch][ix])
            scores[batch] = scores[batch][:-1]
            boxes[batch] = boxes[batch][:-1]
            classes[batch] = classes[batch][:-1]
          else:
            break

    tmp_dict = detections[ensemble[0]]
    tmp_dict.update({'detection_scores': scores, 'detection_boxes': boxes})
    return tmp_dict

def first_stage(m, input):
    if DEBUG_MODE: return params_first_stage[m] * input
    else:          return propose_boxes_only(all_[m]['model'], input)

def postprocess_first_stage(ensemble, proposals):
    if DEBUG_MODE: return sum([proposals[m] for m in ensemble]) / float(len(ensemble))
    else:          return merge_proposals(ensemble, proposals)

def second_stage(m, old_pred_dict, rpn_features, image_shape, boxes, num_proposals):
    if DEBUG_MODE: return num_proposals[0] / float(params_second_stage[m])
    else:          return classify_proposals_only(all_[m]['model'], old_pred_dict, rpn_features, image_shape, boxes, num_proposals)

def postprocess_second_stage(ensemble, detections):
    if DEBUG_MODE: return sum([detections[m] for m in ensemble]) / float(len(ensemble))
    else:          return merge_detections(ensemble, detections)


####################################################################
# The inputs
####################################################################
# Input for the first stage: the image / second stage: the selected proposals
image_shape = [1, None, None, 3]
if DEBUG_MODE: image_shape=[]
image_tensor = tf.placeholder(tf.float32, shape=image_shape, name='image_tensor')
rpn_feat_tensor = tf.placeholder(tf.float32, shape=[1, None, None, 1024], name='rpn_feat')
prop_boxes_tensor = tf.placeholder(tf.float32, shape=[1, 300, 4], name='prop_boxes')
num_prop_tensor = tf.placeholder(tf.int32, shape=[1], name='num_prop')
image_shape_tensor = tf.placeholder(tf.int32, shape=[4], name='img_shape')



####################################################################
# Preparation
####################################################################
# Getting an image
def get_image(images_dir, image_name):
  path = osp.join(images_dir, '{}.{}'.format(image_name, dataset['image_extension']))
  return Image.open(path)

# Helper for the image
def get_image_for_odapi(images_dir, image_name):
  image = get_image(images_dir, image_name)
  (im_width, im_height) = image.size
  image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  extended_image = np.expand_dims(image_np, axis=0)
  return extended_image

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
# How many images will be processed at a time
batch = 1

# Do the batch
batch_list = filtered_list[:batch]

# Will store the proposals / detections for each image in these dicts
results_proposals_dict = {}
results_detections_dict = {}
for img in batch_list:
  results_proposals_dict[img] = {}
  results_detections_dict[img] = {}

# First stage, for each model, one by one
for m in ensemble:

  # First stage scenario
  tmp_proposals = first_stage(m, image_tensor)
  # Create the session
  sess = create_session_for_model(m)

  # For each image in the batch
  for img in batch_list:

    # The input image is the same for all the models in the ensemble
    readable_image = get_image_for_odapi(images_dir, img)

    # Run the first stage
    image_to_feed = readable_image
    if DEBUG_MODE: image_to_feed = 3
    results_proposals_dict[img][m] = sess.run(
      tmp_proposals, 
      feed_dict={image_tensor: image_to_feed}
    )

  sess.close()


# For each image, merge the results of all the models
ensembled_proposals = {}
for img in batch_list:
  ensembled_proposals[img] = postprocess_first_stage(ensemble, results_proposals_dict[img])
results_proposals_dict.clear()


# Second stage, for each model, one by one
for m in ensemble:

  # Second stage
  tmp_detections = second_stage(m, ensembled_proposals[img]['prediction_dict'],
                                   rpn_feat_tensor, image_shape_tensor, 
                                   prop_boxes_tensor, num_prop_tensor)
  # Create the session
  sess = create_session_for_model(m)

  # For each image in the batch
  for img in batch_list:

    # Run the second stage
    tmp_proposal = ensembled_proposals[img]
    tmp_prediction = tmp_proposal['prediction_dict']
    results_detections_dict[img][m] = sess.run(
      tmp_detections, 
      feed_dict={rpn_feat_tensor:    tmp_prediction['rpn_features_to_crop'],
                 image_shape_tensor: tmp_prediction['image_shape'],
                 prop_boxes_tensor:  tmp_proposal['proposal_boxes_normalized'],
                 num_prop_tensor:    tmp_proposal['num_proposals']}
    )
  
  sess.close()

# For each image, merge the results of all the models
ensembled_detections = {}
for img in batch_list:
  ensembled_detections[img] = postprocess_second_stage(ensemble, results_detections_dict[img])
results_detections_dict.clear()


# Save results
for img in batch_list:
  # The image, for global characteristics
  image = get_image(images_dir, img)
  (im_width, im_height) = image.size
  
  # General post processing
  label_id_offset = 1
  scores = np.squeeze(ensembled_detections[img]['detection_scores'], axis=0)
  boxes = np.squeeze(ensembled_detections[img]['detection_boxes'], axis=0)
  classes = np.squeeze(ensembled_detections[img]['detection_classes'], axis=0) + label_id_offset
  
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

# Go to next batch  
filtered_list = filtered_list[batch:]









