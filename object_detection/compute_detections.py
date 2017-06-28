import numpy as np
import time
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import cv2
try:
  import cPickle as pickle
except ImportError:
  import pickle
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

start = time.time()

def ping (txts):
  txt = ' '.join([str(t) for t in txts])
  print "[Info    ]>>> " + txt + "..."
  global start
  start = time.time()
def pong ():
  t_spent = int(time.time() - start)
  print "[Timer   ]>>> Done in", t_spent, "secondes."
def displayProgress (index, nbMax, tempo, add_infos=""):
  if index % tempo != 0 and index < nbMax-1: return
  add_infos = " (" + add_infos + ")" if add_infos != "" else ""
  sys.stdout.write("\r[Process ]>>> " + str(int(index)) + " / " + str(int(nbMax)) + " done." + add_infos)
  sys.stdout.flush()
  if index >= nbMax-1: print

def py_cpu_nms(dets, thresh):
  if dets.shape[0] == 0:
    return []

  """Pure Python NMS baseline."""
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)

    inds = np.where(ovr <= thresh)[0]
    order = order[inds + 1]

  return keep

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



# Arguments
flags = tf.app.flags
flags.DEFINE_string('eval_dir', '',
                    'Directory where to save the detections.')
FLAGS = flags.FLAGS


# Detection values
THRESH = 0.05
NMS_OVERLAPS = 0.3
MAX_PER_IMAGE = 100


import getpass
user = getpass.getuser()
if user == "pierre":
  PATH_TO_CKPT = '/home/pierre/projects/deep_learning/setup_tf/models/object_detection/graph.pb'
  PATH_TO_DATASET = '/home/pierre/projects/datasets/Foodinc/'
elif user == "finc":
  PATH_TO_CKPT = '/mnt2/results/TMP_new_frcnn_res101_foodinc/graph.pb'
  PATH_TO_DATASET = '/mnt2/datasets/Foodinc/'
else:
  assert False, "unowkn user"

PATH_TO_LABELS = os.path.join('data', 'foodinc_label_map.pbtxt')
NUM_CLASSES = 67
NB_IMAGES = -1
IMG_EXT = 'png'
ANN_EXT = 'txt'
PATH_TO_TEST_IMAGES_DIR = PATH_TO_DATASET + 'Images'
PATH_TO_TEST_ANNOTATIONS_DIR = PATH_TO_DATASET + 'Annotations'
LIST_TEST_IMAGES = PATH_TO_DATASET + 'ImageSets/test.txt'


# Requirements
assert os.path.isfile(PATH_TO_CKPT)
assert os.path.isfile(PATH_TO_LABELS)
assert NUM_CLASSES > 0
assert os.path.exists(PATH_TO_TEST_IMAGES_DIR)
assert os.path.exists(PATH_TO_TEST_ANNOTATIONS_DIR)
assert os.path.isfile(LIST_TEST_IMAGES)
assert FLAGS.eval_dir
list_images_names = [line.rstrip('\n') for line in open(LIST_TEST_IMAGES)]
if NB_IMAGES > 0 and NB_IMAGES < len(list_images_names):
  list_images_names = list_images_names[:NB_IMAGES]
NB_IMAGES = len(list_images_names)
for name in list_images_names:
  assert os.path.isfile(os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.{}'.format(name, IMG_EXT)))
  assert os.path.isfile(os.path.join(PATH_TO_TEST_ANNOTATIONS_DIR, '{}.{}'.format(name, ANN_EXT)))

# Images and annotations
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 
                                  '{}.{}'.format(i, IMG_EXT)) for i in list_images_names ]
TEST_ANNOTATIONS_PATHS = [ os.path.join(PATH_TO_TEST_ANNOTATIONS_DIR, 
                                        '{}.{}'.format(i, ANN_EXT)) for i in list_images_names ]

# Categs infos
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Tensorflow init
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Detections
all_boxes = [[[] for _ in range(NB_IMAGES)]
                 for _ in range(NUM_CLASSES + 1)]

ping (['Start compute detections'])

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_idx, image_path in enumerate(TEST_IMAGE_PATHS):
      displayProgress (image_idx, len(TEST_IMAGE_PATHS), 1, image_path)

      image = Image.open(image_path)
      width, height = image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      # Squeeze the single dims
      boxes = np.squeeze(boxes)
      classes = np.squeeze(classes).astype(np.int32)
      scores = np.squeeze(scores)
      inds = np.where(scores[:] > THRESH)[0]
      boxes = boxes[inds]
      classes = classes[inds]
      scores = scores[inds]
      boxes_normalized = []
      for box in boxes:
        y1, x1, y2, x2 = box
        boxes_normalized.append([y1*height, x1*width, y2*height, x2*width])
      boxes = np.asarray(boxes_normalized)

      # For each category (except 0: background)
      for j in range(1, NUM_CLASSES):
        inds = np.where(classes[:] == j)[0]

        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        #print str(len(cls_boxes)) + " -> " + str(cls_boxes) + "\t\t" + str(len(cls_scores)) + " -> " + str(cls_scores)
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)

        keep = py_cpu_nms(cls_dets, THRESH)
        cls_dets = cls_dets[keep, :]
        all_boxes[j][image_idx] = cls_dets

      # Limit to max_per_image detections *over all classes*
      if MAX_PER_IMAGE > 0:
        image_scores = np.hstack([all_boxes[j][image_idx][:, -1]
                      for j in range(1, NUM_CLASSES)])
        if len(image_scores) > MAX_PER_IMAGE:
          image_thresh = np.sort(image_scores)[-MAX_PER_IMAGE]
          for j in range(1, NUM_CLASSES):
            keep = np.where(all_boxes[j][image_idx][:, -1] >= image_thresh)[0]
            all_boxes[j][image_idx] = all_boxes[j][image_idx][keep, :]
pong()

det_file = os.path.join(FLAGS.eval_dir, 'detections.pkl')
with open(det_file, 'wb') as f:
  pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
