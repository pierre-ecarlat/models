import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import numpy as np

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2


# This is needed to display the images.
#matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


PATH_TO_FRCNN_CKPT = '/home/pierre/projects/deep_learning/foodDetectionAPI/models/Faster_RCNN_ResNet101_Foodinc_950k.pb'
PATH_TO_FRCNN_INCEPTION_CKPT = '/home/pierre/projects/deep_learning/foodDetectionAPI/models/Faster_RCNN_ResNet101_Inception_Foodinc_340k.pb'
PATH_TO_SSD_CKPT = '/home/pierre/projects/deep_learning/foodDetectionAPI/models/SSD_MobileNet_Foodinc_550k.pb'

# Arguments
flags = tf.app.flags
flags.DEFINE_string('graph', 'frcnn',
                    'Graph [frcnn, inception, ssd].')
FLAGS = flags.FLAGS
assert FLAGS.graph in ['frcnn', 'inception', 'ssd'], ''

if FLAGS.graph == 'frcnn':
  PATH_TO_CKPT = PATH_TO_FRCNN_CKPT
elif FLAGS.graph == 'inception':
  PATH_TO_CKPT = PATH_TO_FRCNN_INCEPTION_CKPT
elif FLAGS.graph == 'ssd':
  PATH_TO_CKPT = PATH_TO_SSD_CKPT



# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = '/home/pierre/projects/deep_learning/setup_tf/results/frcnnR101_coco/my_frozenInferenceGraph_optimized.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'foodinc_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 67
#NUM_CLASSES = 90

#PATH_TO_TEST_IMAGES_DIR = '/home/pierre/projects/datasets/VOC2007/Images'
#LIST_TEST_IMAGES = '/home/pierre/projects/datasets/VOC2007/ImageSets/Main/test.txt'
PATH_TO_TEST_IMAGES_DIR = '/home/pierre/projects/datasets/Foodinc/Images'
LIST_TEST_IMAGES = '/home/pierre/projects/datasets/Foodinc/ImageSets/test_bckp.txt'
#PATH_TO_TEST_IMAGES_DIR = '/home/pierre/Desktop/dumpImages'
#LIST_TEST_IMAGES = '/home/pierre/Desktop/dumpImages/list.txt'
NB_IMAGES = -1
#IMG_EXT = 'jpg'
IMG_EXT = 'png'


if not os.path.isfile(PATH_TO_CKPT):
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	    file_name = os.path.basename(file.name)
	    if 'frozen_inference_graph.pb' in file_name:
	        tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
 

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
list_images_names = [line.rstrip('\n') for line in open(LIST_TEST_IMAGES)]#[:NB_IMAGES]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.{}'.format(i, IMG_EXT)) for i in list_images_names ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


print 'start detection'

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
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
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      cv2_image = np.array(image_np)
      cv2_image = cv2_image[:, :, ::-1].copy()
      cv2.imshow(FLAGS.graph, cv2_image)
      cv2.waitKey(0)

