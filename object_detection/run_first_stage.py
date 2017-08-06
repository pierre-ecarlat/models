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
import gzip

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
  }, 
  'inception': { 
      'config':  'faster_rcnn_inception_resnet_v2_atrous_foodinc.config', 
      'ckp_dir': 'inception', 
  }, 
  'rfcn': {
      'config':  'rfcn_resnet101_foodinc.config', 
      'ckp_dir': 'rfcn', 
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
errors = { 'annotated': [], 'computed': [], 'notafile': [] }
for img_name in list_images_names:
  if osp.isfile(osp.join(annotations_dir, img_name + '.txt')):
    errors['annotated'].append(img_name)
  elif osp.isfile(osp.join(annotations_dir, img_name + '.' + model + '.1')) and \
       osp.isfile(osp.join(annotations_dir, img_name + '.' + model + '.1.prop')):
    errors['computed'].append(img_name)
  elif not osp.isfile(osp.join(images_dir, '{}.{}'.format(img_name, dataset['image_extension']))):
    errors['notafile'].append(img_name)
  else:
    filtered_list.append(img_name)

if len(errors['annotated']) > 0:
  print len(errors['annotated']), "images are already annotated, will be skipped"
if len(errors['computed']) > 0:
  print len(errors['computed']), "images are already computed, will be skipped"
if len(errors['notafile']) > 0:
  print len(errors['notafile']), "images are in the list, but doesn't exist....."
if len(filtered_list) == 0:
  print "No images to annotate."
  raise SystemExit


####################################################################
# Create the tensorflow methods for the scenarii
####################################################################
def propose_boxes_only(image_tensor):
    # For each model, compute the proposals
    preprocessed_image = built_model.preprocess(tf.to_float(image_tensor))

    # Run the first stage only (proposals)
    (prediction_dict, proposal_boxes_normalized, proposal_scores, num_proposals) = \
        built_model.propose_boxes_only(preprocessed_image)

    # Returns them
    return {
        'prediction_dict': prediction_dict, 
        'proposal_boxes_normalized': proposal_boxes_normalized, 
        'proposal_scores': proposal_scores, 
        'num_proposals': num_proposals, 
    }


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
# Input for the first stage: the image
image_tensor = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image_tensor')


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

# Do the batch
batch_list = filtered_list[:batch]

# First stage scenario
proposals_tensor = propose_boxes_only(image_tensor)
# Create the session
sess = create_session_for_model(model)

# For each image in the batch
for img in batch_list:

  # The input image is the same for all the models in the ensemble
  readable_image = get_image_for_odapi(images_dir, img)

  # Run the first stage
  proposals = sess.run(
    proposals_tensor, 
    feed_dict={image_tensor: readable_image}
  )

  # To keep for the second stage
  model_charact_to_save = {
    'prediction_dict': {
        'image_shape': proposals['prediction_dict']['image_shape'].tolist(), 
        'rpn_features_to_crop': proposals['prediction_dict']['rpn_features_to_crop'].tolist(), 
    }, 
  }

  # To ensemble
  predictions_to_save = {
    'proposal_boxes_normalized': proposals['proposal_boxes_normalized'].tolist(), 
    'proposal_scores': proposals['proposal_scores'].tolist(), 
    'num_proposals': proposals['num_proposals'].tolist(), 
  }

  tmp_annotation_file = osp.join(annotations_dir, img + '.' + model + '.1')
  json.dump(predictions_to_save, open(tmp_annotation_file, 'w'))

  tmp_model_annotation_file = osp.join(annotations_dir, img + '.' + model + '.1.prop')
  json_str = json.dumps(model_charact_to_save)
  json_bytes = json_str.encode('utf-8')
  with gzip.GzipFile(tmp_model_annotation_file, 'w') as f:
    f.write(json_bytes)

sess.close()


""" Output format:

{
  'proposal_boxes_normalized': array([], dtype=float32),                      # (1, 300, 4) 
  'prediction_dict': {                                                        
      'rpn_box_encodings': array([], dtype=float32),                          # (1, 23256, 4)
      'anchors': array([], dtype=float32),                                    # (23256, 4)
      'image_shape': array([], dtype=int32),                                  # (4,)
      'rpn_features_to_crop': array([], dtype=float32),                       # (1, 51, 38, 1024)
      'rpn_objectness_predictions_with_background': array([], dtype=float32), # (1, 23256, 2)
      'rpn_box_predictor_features': array([], dtype=float32)                  # (1, 51, 38, 512)
  }, 
  'proposal_scores': array([], dtype=float32),                                # (1, 300)
  'num_proposals': array([], dtype=int32)                                     # (1,)
}

"""





