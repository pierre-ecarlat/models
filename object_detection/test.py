import os
import os.path as osp
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import numpy as np
import functools
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
  'base_models_dir':  '/mnt2/results', 
  'base_configs_dir': 'samples/configs', 
  'frcnn': {
      'config':  'faster_rcnn_resnet101_foodinc.config', 
      'ckp_dir': 'Foodinc/frcnn_res101_e2e_tf_ODAPI', 
  }, 
  'inception': { 
      'config':  'faster_rcnn_inception_resnet_v2_atrous_foodinc.config', 
      'ckp_dir': 'Foodinc/frcnn_inception_tf_ODAPI', 
  }, 
  'ssd': {
      'config':  'ssd_mobilenet_v1_foodinc.config', 
      'ckp_dir': 'Foodinc/ssd_mobilenet_tf_ODAPI', 
  }, 
}

DATASET = {
  'foodinc': {
      'nb_classes': 67, 
      'base_dir': '/home/finc/GodFoodinc', 
      'images_dir': 'Images', 
      'annotations_dir': 'Annotations', 
      'labels_path': 'infos/foodinc_label_map.pbtxt',
      'list_path': 'ImageSets/all.txt', 
      'max_images': 1, 
      'image_extension': 'jpg', 
  }
}


DEBUG_MODE = False
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
for img_name in list_images_names:
    if osp.isfile(osp.join(annotations_dir, img_name + '.txt')):
        print 'Found an annotation for img', img_name, ', will be skipped.'
    elif not osp.isfile(osp.join(images_dir, '{}.{}'.format(img_name, dataset['image_extension']))):
        print 'Unable to find img', img_name, ', will be skipped.'
    else:
        filtered_list.append(img_name)


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
    # TODO
    # Note, format:
    # {'prediction_dict': 
    #     {'rpn_box_encodings': <tf.Tensor 'Squeeze:0' shape=(1, ?, 4) dtype=float32>,
    #      'anchors': <tf.Tensor 'ClipToWindow/Gather/Gather:0' shape=(?, 4) dtype=float32>, 
    #      'image_shape': <tf.Tensor 'Shape:0' shape=(4,) dtype=int32>, 
    #      'rpn_features_to_crop': <tf.Tensor 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_23/bottleneck_v1/Relu:0' shape=(1, ?, ?, 1024) dtype=float32>, 
    #      'rpn_objectness_predictions_with_background': <tf.Tensor 'FirstStageBoxPredictor/Reshape_1:0' shape=(1, ?, 2) dtype=float32>, 
    #      'rpn_box_predictor_features': <tf.Tensor 'Conv/Relu6:0' shape=(1, ?, ?, 512) dtype=float32>
    #     }, 
    #  'proposal_scores': <tf.Tensor 'stack_3:0' shape=(1, 300) dtype=float32>, 
    #  'num_proposals': <tf.Tensor 'stack_4:0' shape=(1,) dtype=int32>, 
    #  'proposal_boxes_normalized': <tf.Tensor 'stack_2:0' shape=(1, 300, 4) dtype=float32>
    # }
    result = proposals[ensemble[0]]
    return result

def classify_proposals_only(model, proposals):
    # Run the second stage only and return the result
    return model.classify_proposals_only(\
          proposals['prediction_dict'], 
          proposals['proposal_boxes_normalized'], 
          proposals['num_proposals']
    )

def merge_detections(ensemble, detections):
    # TODO
    result = detections[ensemble[0]]
    return result

def postProcess(detections):
    # TODO
    # Should not depend on the model.... ?
    detections = all_['frcnn']['model'].postprocess(detections)
    return detections

def first_stage(m, input):
    if DEBUG_MODE: return params_first_stage[m] * input
    else:          return propose_boxes_only(m, input)

def postprocess_first_stage(ensemble, proposals):
    if DEBUG_MODE: return sum([proposals[m] for m in ensemble]) / float(len(ensemble))
    else:          return merge_proposals(ensemble, proposals)

def second_stage(m, proposals):
    if DEBUG_MODE: return proposals / float(params_second_stage[m])
    else:          return classify_proposals_only(m, proposals)

def postprocess_second_stage(ensemble, detections):
    if DEBUG_MODE: return sum([detections[m] for m in ensemble]) / float(len(ensemble))
    else:          return merge_detections(ensemble, detections)


####################################################################
# The inputs
####################################################################

# Input for the first stage: the image / second stage: the selected proposals
if DEBUG_MODE: image_shape=[]
else:          image_shape=[1, None, None, 3]
image_tensor = tf.placeholder(tf.float32, shape=image_shape, name='image_tensor')
proposals_tensor = tf.placeholder(tf.float32, shape=[], name='proposals_tensor')


####################################################################
# Preparation
####################################################################
nb_imgs = len(filtered_list)
print 'Will annotate', nb_imgs, 'image{}.'.format('s' if nb_imgs > 1 else '')

# Helper for the image
def get_image_for_odapi(images_dir, image_name):
  path = osp.join(images_dir, '{}.{}'.format(image_name, dataset['image_extension']))
  image = Image.open(path)
  (im_width, im_height) = image.size
  image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  extended_image = np.expand_dims(image_np, axis=0)
  return extended_image

# Helper for the session
def create_session_for_model(model):
  sess = tf.Session()
  path_to_ckpt = osp.join(MODELS['base_models_dir'], MODELS[model]['ckp_dir'])
  path_to_latest_ckpt = tf.train.latest_checkpoint(path_to_ckpt)
  path_to_meta = ".".join([path_to_latest_ckpt, "meta"])
  if not DEBUG_MODE:
    saver = tf.train.import_meta_graph(path_to_meta)
    saver.restore(sess, path_to_latest_ckpt)
  return sess


####################################################################
# Start to annotate
####################################################################
# How many images will be processed at a time
batch = 3

# Go over the list of images
while len(filtered_list) > 0:

  # Get the list of images for this batch
  batch_list = filtered_list[:batch]

  # Will store the proposals / detections for each image in these dicts
  results_proposals_dict = {}
  results_detections_dict = {}
  for img in batch_list:
    results_proposals_dict[img] = {}
    results_detections_dict[img] = {}


  # First stage, for each model, one by one
  for m in ensemble:

    # Create the session
    sess = create_session_for_model(m)

    # For each image in the batch
    for img in batch_list:

      # The input image is the same for all the models in the ensemble
      readable_image = get_image_for_odapi(images_dir, img)

      # Run the first stage
      if DEBUG_MODE: image_to_feed = 3
      else:          image_to_feed = readable_image
      results_proposals_dict[img][m] = sess.run(
        [first_stage(m, image_tensor)], 
        feed_dict={image_tensor: image_to_feed}
      )[0]


  # For each image, merge the results of all the models
  ensembled_proposals = {}
  for img in batch_list:
    ensembled_proposals[img] = postprocess_first_stage(ensemble, results_proposals_dict[img])
  results_proposals_dict.clear()


  # Second stage, for each model, one by one
  for m in ensemble:

    # Create the session
    sess = create_session_for_model(m)

    # For each image in the batch
    for img in batch_list:

      # Run the second stage
      results_detections_dict[img][m] = sess.run(
        [second_stage(m, proposals_tensor)], 
        feed_dict={proposals_tensor: ensembled_proposals[img]}
      )[0]


  # For each image, merge the results of all the models
  ensembled_detections = {}
  for img in batch_list:
    ensembled_detections[img] = postprocess_second_stage(ensemble, results_detections_dict[img])
  results_detections_dict.clear()
  

  # Get the format of first stage's dict for lost parameters
  if not DEBUG_MODE:
    for img in batch_list:
      tmp_predictions_dict = ensembled_proposals[img]['prediction_dict']
      tmp_predictions_dict.update(ensembled_detections[img])
      ensembled_detections[img] = tmp_predictions_dict

  # Postprocessing
  if not DEBUG_MODE:
    for img in batch_list:
      ensembled_detections[img] = postProcess(ensembled_detections[img])

  for img in batch_list:
    print ensembled_detections[img]

  # Save results
  # TODO
  
  filtered_list = filtered_list[batch:]





raise SystemExit


# For every image
for image_name in filtered_list:
    print '>> Image', image_name

    # The input image is the same for all the models in the ensemble
    image_path = osp.join(images_dir, '{}.{}'.format(image_name, dataset['image_extension']))
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    original_image = np.expand_dims(image_np, axis=0)


    # First stage
    proposals_dict = {}
    for m in ensemble:
        proposals_dict[m] = first_stage(m, image_tensor)

    # Run the first stage
    results_proposals_dict = {}
    for m in ensemble:
        results_proposals_dict[m] = sessions[m].run([proposals_dict[m]], feed_dict={image_tensor: 3})[0]

    # Merge the proposals
    proposals_dict['ensemble'] = postprocess_first_stage(ensemble, results_proposals_dict)


    # Second stage
    detections_dict = {}
    for m in ensemble:
      detections_dict[m] = second_stage(m, proposals_tensor)

    # Run the second stage
    results_detections_dict = {}
    for m in ensemble:
        results_detections_dict[m] = sessions[m].run([detections_dict[m]], feed_dict={proposals_tensor: proposals_dict['ensemble']})[0]

    # Merge the detections
    detections_dict['ensemble'] = postprocess_second_stage(ensemble, results_detections_dict)


    # Get the format of first stage's dict for lost parameters
    if not DEBUG_MODE:
        tmp_predictions_dict = proposals_dict['ensemble']['prediction_dict']
        tmp_predictions_dict.update(detections_dict['ensemble'])
        detections_dict['ensemble'] = tmp_predictions_dict

    # Postprocessing
    if not DEBUG_MODE:
        detections_dict['ensemble'] = postProcess(detections_dict['ensemble'])


    print(detections_dict['ensemble'])













