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

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

start = time.time()
import re

def parse_rec(filename):
  """ Parse a Foodinc txt file """
  
  print('Reading annotation for file: {}'.format(filename))
  
  with open(filename) as f:
    data = f.read()
  
  # import re
  objs = re.findall('\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+', data)
  num_objs = len(objs)

  # Return objects
  objects = []

  # Load object bounding boxes into a data frame.
  for ix, obj in enumerate(objs):
    coor = re.findall('\d+', obj)
    # Make pixel indexes 0-based
    cls = int(coor[0])
    x1 = float(coor[1])
    y1 = float(coor[2])
    x2 = float(coor[3])
    y2 = float(coor[4])

    obj_struct = {}
    obj_struct['name'] = str(cls)
    obj_struct['bbox'] = [int(x1),
                          int(y1),
                          int(x2),
                          int(y2)]
    objects.append(obj_struct)

#  print(objects)

  return objects

def ping (txts):
  txt = ' '.join([str(t) for t in txts])
  print "[Info    ]>>> " + txt + "..."
  global start
  start = time.time()
def pong ():
  t_spent = int(time.time() - start)
  print "[Timer   ]>>> Done in", t_spent, "secondes."
def displayProgress (index, nbMax, tempo):
  if index % tempo != 0 and index < nbMax-1: return
  sys.stdout.write("\r[Process ]>>> " + str(int(index)) + " / " + str(int(nbMax)) + " done.")
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

def compareBoxes(BB_gt, BB_det):
  # compute overlaps
  # intersection
  ixmin = np.maximum(BB_gt[:, 0], BB_det[0])
  iymin = np.maximum(BB_gt[:, 1], BB_det[1])
  ixmax = np.minimum(BB_gt[:, 2], BB_det[2])
  iymax = np.minimum(BB_gt[:, 3], BB_det[3])
  iw = np.maximum(ixmax - ixmin + 1., 0.)
  ih = np.maximum(iymax - iymin + 1., 0.)
  inters = iw * ih

  # union
  uni = ((BB_det[2] - BB_det[0] + 1.) * (BB_det[3] - BB_det[1] + 1.) +
         (BB_gt[:, 2] - BB_gt[:, 0] + 1.) *
         (BB_gt[:, 3] - BB_gt[:, 1] + 1.) - inters)

  overlaps = inters / uni
  return np.max(overlaps), np.argmax(overlaps)



# Arguments
flags = tf.app.flags
flags.DEFINE_string('eval_dir', '',
                    'Directory where to find the detections.')
FLAGS = flags.FLAGS
assert FLAGS.eval_dir


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'foodinc_label_map.pbtxt')
NUM_CLASSES = 67
# Data
PATH_TO_TEST_IMAGES_DIR = '/mnt2/datasets/Foodinc/Images'
PATH_TO_TEST_ANNOTATIONS_DIR = '/mnt2/datasets/Annotations'
PATH_TO_TEST_ANNOTATIONS_DIR = '/mnt2/datasets/annotations_cache'
LIST_TEST_IMAGES = '/mnt2/datasets/Foodinc/ImageSets/test.txt'
NB_IMAGES = -1
IMG_EXT = 'png'
ANN_EXT = 'txt'

# Detections
DETECTIONS_PATH = os.path.join(FLAGS.eval_dir, "detections.pkl")


# Requirements
assert os.path.isfile(PATH_TO_LABELS)
assert NUM_CLASSES > 0
assert os.path.exists(PATH_TO_TEST_IMAGES_DIR)
assert os.path.exists(PATH_TO_TEST_ANNOTATIONS_DIR)
assert os.path.isfile(LIST_TEST_IMAGES)
assert os.path.isfile(DETECTIONS_PATH)
list_images_names = [line.rstrip('\n') for line in open(LIST_TEST_IMAGES)]
if NB_IMAGES > 0 and NB_IMAGES < len(list_images_names):
  list_images_names = list_images_names[:NB_IMAGES]
NB_IMAGES = len(list_images_names)
for name in list_images_names:
  assert os.path.isfile(os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.{}'.format(name, IMG_EXT)))
  assert os.path.isfile(os.path.join(PATH_TO_TEST_ANNOTATIONS_DIR, '{}.{}'.format(name, ANN_EXT)))

# Categs infos
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the detections
with open(DETECTIONS_PATH, 'rb') as f:
  all_boxes = pickle.load(f)


print('Evaluating detections')

for cls_ind, cls in enumerate(categories):
  if cls['name'] == 'none_of_the_above':
    continue
  print('Writing {} Foodinc results file, ID: {}'.format(cls['name'], cls_ind))
  filename = os.path.join(FLAGS.eval_dir, cls['name'] + ".txt")
  with open(filename, 'wt') as f:
    for im_ind, index in enumerate(list_images_names):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue
      for k in range(dets.shape[0]):
        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                format(index, dets[k, -1],
                       dets[k, 0], dets[k, 1],
                       dets[k, 2], dets[k, 3]))




def eval(detpath,
         images_dir,
         annotations_dir,
         list_images,
         classname,
         class_id,
         cachedir,
         ovthresh=0.5):
  """
  def foodinc_eval(detpath,
                 annopath,
                 imagesetfile,
                 classname,
                 class_id,
                 cachedir,
                 ovthresh=0.5,
                 reward_relatives=0.3,
                 confidence_metric=False):
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, 'annots.pkl')

  # read list of images
  imagenames = list_images

  # Get all the gt objects
  if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(os.path.join(annotations_dir, imagename + ".txt"))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
                                        i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'w') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'r') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')


  # Extract gt objects specific to this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == str(class_id)]
    bbox = np.array([x['bbox'] for x in R])
    # Keep track of counted detections, initialise as false
    det = [False] * len(R)
    class_recs[imagename] = { 'bbox': bbox, 'det': det }
  for rec in class_recs:
    npos += len(class_recs[rec]['det'])

  
  # Read the detection
  detections = [line.rstrip('\n').split(' ') for line in open(detpath)]
  image_ids = [x[0] for x in detections]
  confidence = np.array([float(x[1]) for x in detections])
  BB = np.array([[float(z) for z in x[2:]] for x in detections])

  # Go down detections and mark TPs and FPs (as much as detections)
  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  # If no detection for this class, returns nothing
  if BB.shape[0] == 0:
    return [], [], 0.0

  # Sort everything by confidence
  sorted_ind = np.argsort(-confidence)
  sorted_scores = np.sort(-confidence)
  BB = BB[sorted_ind, :]
  image_ids = [image_ids[x] for x in sorted_ind]

  # Go down dets and mark TPs and FPs
  for d in range(nd):
    # The detection box to compare
    _bb = BB[d, :].astype(float)
    bb = [_bb[1], _bb[0], _bb[3], _bb[2]]
    ovmax = -np.inf
    R = class_recs[image_ids[d]]
    BBGT = R['bbox'].astype(float)

    # Get the max overlap
    if BBGT.size > 0:
      ovmax, jmax = compareBoxes(BBGT, bb)

    # If overlap, this is a true positive
    if ovmax > ovthresh:
      if not R['det'][jmax]:
        tp[d] = 1.
        R['det'][jmax] = 1
      # If already detected (should not happen)
      else:
        fp[d] = 1.
    else:
      fp[d] = 1.

  # compute precision recall ap
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
  i = np.where(mrec[1:] != mrec[:-1])[0]
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

  return rec, prec, ap



aps = []

for i, cls in enumerate(categories):
  # if i not in mini_val:
  #   continue
  if cls['name'] == 'none_of_the_above':
    continue
  
  filename = os.path.join(FLAGS.eval_dir, cls['name'] + ".txt")
  rec, prec, ap = eval(filename, 
                       PATH_TO_TEST_IMAGES_DIR,
                       PATH_TO_TEST_ANNOTATIONS_DIR, 
                       list_images_names, 
                       cls['name'], 
                       i, 
                       PATH_TO_TEST_CACHE_DIR)

  aps += [ap]

  print('AP for {} = {:.4f}'.format(cls['name'], ap))
  with open(os.path.join(FLAGS.eval_dir, cls['name'] + '_pr.pkl'), 'w') as f:
    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

print(('Mean AP = {:.4f}'.format(np.mean(aps))))
print('~~~~~~~~')
print('Results:')
for ap in aps:
  print(('{:.3f}'.format(ap)))







