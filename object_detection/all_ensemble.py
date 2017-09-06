import os
import os.path as osp
import numpy as np
from PIL import Image
import bisect
import time
import base64
import cStringIO
from io import BytesIO
import json

import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2

dataset_name = 'foodinc'
ensemble = ['frcnn', 'inception', 'rfcn']
CONFIDENCE_THRESH = 0.5


####################################################################
# MAIN PARAMETERS
####################################################################
MODELS = {
  'base_models_dir':  '/home/pierre/projects/deep_learning/models', 
  'base_configs_dir': 'samples/configs', 
  'frcnn': {
      'config':  'faster_rcnn_resnet101_foodinc.config', 
      'ckp_dir': 'frcnn_pref', 
      'model': 'Faster_RCNN_ResNet101_foodinc_950k.pb', 
      'features_to_crop_shape': [1, None, None, 1024], 
  }, 
  'inception': { 
      'config':  'faster_rcnn_inception_resnet_v2_atrous_foodinc.config', 
      'ckp_dir': 'inception_pref', 
      'model': 'Faster_RCNN_ResNet101_Inception_foodinc_340k.pb', 
      'features_to_crop_shape': [1, None, None, 1088], 
  }, 
  'rfcn': {
      'config':  'rfcn_resnet101_foodinc.config', 
      'ckp_dir': 'rfcn_pref', 
      'model': 'RFCN_ResNet101_foodinc_424k.pb', 
      'features_to_crop_shape': [1, None, None, 1024], 
  }, 
}

DATASET = {
  'foodinc': {
      'nb_classes': 67, 
  }
}

dataset = DATASET[dataset_name]


####################################################################
# STORER
####################################################################
built_model = {}
for model in ensemble:
  # Configs file for the model
  model_config_file = osp.join(MODELS['base_configs_dir'], MODELS[model]['config'])
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(model_config_file, 'r') as f:
      text_format.Merge(f.read(), pipeline_config)
  model_config = pipeline_config.model
  built_model[model] = model_builder.build(model_config, False)


####################################################################
# Preparation
####################################################################
# Helper for the image
def get_image_for_odapi(raw_image):
  image = Image.open(BytesIO(base64.b64decode(raw_image)))
  (im_width, im_height) = image.size
  image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  extended_image = np.expand_dims(image_np, axis=0)
  return extended_image

# Helper for the session
def create_session_for_model(model):
  variables_to_restore = [v for v in tf.global_variables() if v.name.startswith(model + "/")]
  path_to_ckpt = osp.join(MODELS['base_models_dir'], MODELS[model]['ckp_dir'])
  path_to_latest_ckpt = tf.train.latest_checkpoint(path_to_ckpt)

  saver = tf.train.Saver(variables_to_restore)
  sess = tf.Session('', graph=tf.get_default_graph())
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, path_to_latest_ckpt)
  return sess


####################################################################
# Create the tensorflow methods for the scenarii
####################################################################
def propose_boxes_only(bm, model, image_tensor):
  # For each model, compute the proposals
  preprocessed_image = bm.preprocess(tf.to_float(image_tensor))
  
  # Run the first stage only (proposals)
  (prediction_dict, proposal_boxes_normalized, proposal_scores, num_proposals) = \
      bm.propose_boxes_only(preprocessed_image, scope=model)
  
  # Returns them
  return {
    'prediction_dict': prediction_dict, 
    'proposal_boxes_normalized': proposal_boxes_normalized, 
    'proposal_scores': proposal_scores, 
    'num_proposals': num_proposals, 
  }

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

def classify_proposals_only(bm, model, rpn_features, image_shape, boxes, num_proposals, isRfcn):
  # Run the second stage only and return the result
  detections = bm.classify_proposals_only(rpn_features, image_shape, boxes, num_proposals, isRfcn, scope=model)
  
  # Restore lost parameters between stages
  tmp_dict = { 'image_shape': image_shape }
  tmp_dict.update(detections)
  detections = tmp_dict
  
  return detections

def merge_detections(ensemble, detections):
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

def postprocess_detections(bm, model, detections):
    # Post processing
    postprocessed_detections = bm.postprocess(detections)

    # Returns them
    return postprocessed_detections


####################################################################
# The inputs
####################################################################
# Input for the first stage: the image
image_tensor = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image_tensor')

# Input for the second stage: the selected proposals
rpn_feat_tensor = {}
for model in ensemble:
  rpn_shape = MODELS[model]['features_to_crop_shape']
  rpn_feat_tensor[model] = tf.placeholder(tf.float32, shape=rpn_shape, name='rpn_feat')
prop_boxes_tensor = tf.placeholder(tf.float32, shape=[1, 300, 4], name='prop_boxes')
num_prop_tensor = tf.placeholder(tf.int32, shape=[1], name='num_prop')
image_shape_tensor = tf.placeholder(tf.int32, shape=[4], name='img_shape')

# Input for the postprocessing
class_predictions_tensor = tf.placeholder(tf.float32, shape=[300, dataset['nb_classes']+1], name='class_predictions')
num_proposals_tensor = tf.placeholder(tf.int32, shape=[1, ], name='num_proposals')
image_shape_tensor = tf.placeholder(tf.int32, shape=[4, ], name='image_shape')
refined_box_encodings_tensor = tf.placeholder(tf.float32, shape=[300, dataset['nb_classes'], 4], name='refined_box_encodings')
proposal_boxes_tensor = tf.placeholder(tf.float32, shape=[1, 300, 4], name='proposal_boxes')


####################################################################
# Start to annotate
####################################################################

sessions = {}
proposals_tensor = {}
detections_tensor = {}
for model in ensemble:
  # First stage scenario
  proposals_tensor[model] = propose_boxes_only(built_model[model], model, image_tensor)

  # Second stage scenario
  detections_tensor[model] = classify_proposals_only(built_model[model], model, 
                                                     rpn_feat_tensor[model], image_shape_tensor, 
                                                     prop_boxes_tensor, num_prop_tensor,
                                                     (model == 'rfcn'))
  
  # Create the session
  sessions[model] = create_session_for_model(model)
  
# Postprocess scenario
ensembled_detections_tensor = postprocess_detections(built_model[model], model, {
    'class_predictions_with_background': class_predictions_tensor, 
    'refined_box_encodings': refined_box_encodings_tensor, 
    'proposal_boxes': proposal_boxes_tensor, 
    'num_proposals': num_proposals_tensor, 
    'image_shape': image_shape_tensor, 
  })

# Regular session for post processing
regular_session = tf.Session('', graph=tf.get_default_graph())




# Should come from client
image_path = '/home/pierre/projects/deep_learning/ensembleAPI/res/food1.jpg'
image = Image.open(image_path)
buffer = cStringIO.StringIO()
image.save(buffer, format="JPEG")
img_str = base64.b64encode(buffer.getvalue())






# The input image is the same for all the models in the ensemble
readable_image = get_image_for_odapi(img_str)

# Run the first stage
proposals = {}
for model in ensemble:
  proposals[model] = sessions[model].run(
    proposals_tensor[model], 
    feed_dict={image_tensor: readable_image}
  )

# For each image, merge the results of all the models
ensembled_proposals = merge_proposals(ensemble, proposals)

# Run the second stage
detections = {}
for model in ensemble:
  detections[model] = sessions[model].run(
    detections_tensor[model], 
    feed_dict={rpn_feat_tensor[model]: proposals[model]['prediction_dict']['rpn_features_to_crop'],
               image_shape_tensor:     proposals[model]['prediction_dict']['image_shape'],
               prop_boxes_tensor:      ensembled_proposals['proposal_boxes_normalized'],
               num_prop_tensor:        ensembled_proposals['num_proposals']}
  )

# For each image, merge the results of all the models
ensembled_detections = merge_detections(ensemble, detections)

# Postprocess
postprocessed_detections = regular_session.run(
    ensembled_detections_tensor, 
    feed_dict={class_predictions_tensor:     ensembled_detections['class_predictions_with_background'],
               refined_box_encodings_tensor: ensembled_detections['refined_box_encodings'],
               proposal_boxes_tensor:        ensembled_detections['proposal_boxes'],
               num_proposals_tensor:         ensembled_detections['num_proposals'], 
               image_shape_tensor:           ensembled_detections['image_shape']}
  )

# General post processing
label_id_offset = 1
num_detections = np.squeeze(postprocessed_detections['num_detections'], axis=0)
scores = np.squeeze(postprocessed_detections['detection_scores'], axis=0)
boxes = np.squeeze(postprocessed_detections['detection_boxes'], axis=0)
classes = np.squeeze(postprocessed_detections['detection_classes'], axis=0) + label_id_offset

        
# Parse the result into a json
doc = {}
for i in range(int(num_detections)):
    # ODAPI returns (ymin, xmin, ymax, xmax) format
    doc[str(i)] = [ { 'y1':     str(boxes[i][0]) },
                    { 'x1':     str(boxes[i][1]) },
                    { 'y2':     str(boxes[i][2]) },
                    { 'x2':     str(boxes[i][3]) },
                    { 'score':  str(scores[i])   },
                    { 'classe': str(classes[i])  }
        ]

# Create a JSON representation of the resource
print json.dumps(doc, ensure_ascii=False)

# Send back to client
raise SystemExit






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
  sessions[model].close()


print "Successfully done"

