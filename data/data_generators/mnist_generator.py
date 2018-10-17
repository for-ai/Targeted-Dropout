import gzip
import os
import random
import urllib
import numpy as np
import tensorflow as tf

from .generator_utils import generate_files
from ...models.utils.model_utils import ModeKeys

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

_TRAIN_IMAGE_COUNT = 60000
_TRAIN_IMAGE_FILE = "train-images-idx3-ubyte.gz"
_TRAIN_LABEL_FILE = "train-labels-idx1-ubyte.gz"

_TEST_IMAGE_COUNT = 10000
_TEST_IMAGE_FILE = "t10k-images-idx3-ubyte.gz"
_TEST_LABEL_FILE = "t10k-labels-idx1-ubyte.gz"

_WORKING_DIR = "/tmp/tf_data"


def download_files(filenames):
  """Download files to tmp/data if file does not exist
  Args:
    filenames: list of string; list of filenames to check if exist
  """
  if not os.path.exists(_WORKING_DIR):
    os.makedirs(_WORKING_DIR)
  for filename in filenames:
    filepath = os.path.join(_WORKING_DIR, filename)
    url = "http://yann.lecun.com/exdb/mnist/" + filename
    if not os.path.isfile(filepath):
      print("Downloading %s" % (url + filename))
      try:
        urllib.urlretrieve(url, filepath)
      except AttributeError:
        urllib.request.urlretrieve(url, filepath)


def read_images(filepath, num_images):
  with gzip.open(filepath) as f:
    f.read(16)
    buf = f.read(28 * 28 * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, 28, 28, 1)
  return data


def read_labels(filepath, num_labels):
  with gzip.open(filepath) as f:
    f.read(8)
    buf = f.read(num_labels)
    data = np.frombuffer(buf, dtype=np.uint8)
  return data.astype(np.int64)


def mnist_generator(mode):
  num_images = _TRAIN_IMAGE_COUNT if mode != ModeKeys.TEST else _TEST_IMAGE_COUNT
  image_filepath = _TRAIN_IMAGE_FILE if mode != ModeKeys.TEST else _TEST_IMAGE_FILE
  label_filepath = _TRAIN_LABEL_FILE if mode != ModeKeys.TEST else _TEST_LABEL_FILE

  download_files([image_filepath, label_filepath])

  image_filepath = os.path.join(_WORKING_DIR, image_filepath)
  label_filepath = os.path.join(_WORKING_DIR, label_filepath)

  images = read_images(image_filepath, num_images)
  labels = read_labels(label_filepath, num_images)

  data = list(zip(images, labels))
  random.shuffle(data)
  
  if mode == ModeKeys.TRAIN:
    data = data[:5*num_images//6]
  elif mode == ModeKeys.EVAL:
    data = data[5*num_images//6:]

  image_ph = tf.placeholder(dtype=tf.uint8, shape=(28, 28, 1))
  encoded_ph = tf.image.encode_png(image_ph)

  sess = tf.Session()
  for image, label in data:
    encoded_im = sess.run(encoded_ph, feed_dict={image_ph: image})
    yield {
        "image/encoded": [encoded_im],
        "image/format": [b"png"],
        "image/class/label": [label],
        "image/height": [28],
        "image/width": [28]
    }


def generate(train_name, eval_name, test_name, hparams):
  generate_files(
      mnist_generator(mode=ModeKeys.TRAIN), train_name, hparams.data_dir, 1)
  generate_files(
      mnist_generator(mode=ModeKeys.EVAL), eval_name, hparams.data_dir, 1)
  generate_files(
      mnist_generator(mode=ModeKeys.TEST), test_name, hparams.data_dir, 1)
