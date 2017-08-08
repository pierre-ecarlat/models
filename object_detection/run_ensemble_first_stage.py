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
  elif osp.isfile(osp.join(annotations_dir, img_name + '.1')):
    errors['computed'].append(img_name)
  else:
    ok = [osp.isfile(osp.join(annotations_dir, img_name + '.' + m + '.1'))
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
def merge_proposals(ensemble, proposals):
    scores = proposals[ensemble[0]]['proposal_scores'].tolist()
    boxes = proposals[ensemble[0]]['proposal_boxes_normalized'].tolist()

    for model in ensemble[1:]:
      for batch in range(len(scores)):
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
    tmp_dict.update({'proposal_scores': np.array(scores), 
                     'proposal_boxes_normalized': np.array(boxes)})
    return tmp_dict


####################################################################
# Start to annotate
####################################################################

# No batch for the ensemble, process everything it cans
batch_list = filtered_list


# For each image in the batch
for img in batch_list:

  results_proposals_dict = {}
  for model in ensemble:
    annotation_path = osp.join(annotations_dir, img + '.' + model + '.1')
    loaded_results = json.load(open(annotation_path))

    results_proposals_dict[model] = {
      'proposal_boxes_normalized': np.array(loaded_results['proposal_boxes_normalized']), 
      'proposal_scores': np.array(loaded_results['proposal_scores']), 
      'num_proposals': np.array(loaded_results['num_proposals']), 
    }

  # For each image, merge the results of all the models
  ensembled_proposals = merge_proposals(ensemble, results_proposals_dict)

  # Store as list
  result_to_save = {
    'proposal_boxes_normalized': ensembled_proposals['proposal_boxes_normalized'].tolist(), 
    'proposal_scores': ensembled_proposals['proposal_scores'].tolist(), 
    'num_proposals': ensembled_proposals['num_proposals'].tolist(), 
  }

  tmp_annotation_file = osp.join(annotations_dir, img + '.1')
  json.dump(result_to_save, open(tmp_annotation_file, 'w'))

  for model in ensemble:
    annotation_path = osp.join(annotations_dir, img + '.' + model + '.1')
    os.system("rm " + annotation_path)


""" Output format:

{
  'proposal_boxes_normalized': array([], dtype=float32), # (1, 300, 4) 
  'proposal_scores': array([], dtype=float32),           # (1, 300)
}

"""





