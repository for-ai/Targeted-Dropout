try:
  import cPickle
except ImportError:
  import pickle as cPickle
import os
import random
import sys
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf

from .generator_utils import generate_files
from ...models.utils.model_utils import ModeKeys

FLAGS = tf.app.flags.FLAGS

_URL = "http://www.cs.toronto.edu/~kriz/"
_CIFAR10_TAR = "cifar-10-python.tar.gz"
_CIFAR10_DIR = "cifar-10-batches-py"
_CIFAR10_TRAIN = [
    "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
    "data_batch_5"
]
_CIFAR10_TEST = ["test_batch"]

_CIFAR100_TAR = "cifar-100-python.tar.gz"
_CIFAR100_DIR = "cifar-100-python"
_CIFAR100_TRAIN = ["train"]
_CIFAR100_TEST = ["test"]

_WORKING_DIR = "/tmp/tf_data"


def download(v100):
  archive = _CIFAR100_TAR if v100 else _CIFAR10_TAR
  filepath = os.path.join(_WORKING_DIR, archive)
  if not os.path.exists(_WORKING_DIR):
    os.makedirs(_WORKING_DIR)
  url = _URL + archive
  if not os.path.isfile(filepath):
    print("Downloading " + url)
    urllib.request.urlretrieve(url, filepath)
  print("Extracting " + filepath)
  tar = tarfile.open(filepath, "r:gz")
  tar.extractall(path=_WORKING_DIR)
  tar.close()


def maybe_download(files, v100):
  for file in files:
    filepath = os.path.join(_WORKING_DIR, _CIFAR100_DIR
                            if v100 else _CIFAR10_DIR, file)
    if not os.path.isfile(filepath):
      download(v100)
      break


def read_files(files, v100):
  images = None
  labels = None
  for file in files:
    filename = os.path.join(_WORKING_DIR, _CIFAR100_DIR
                            if v100 else _CIFAR10_DIR, file)
    data = None
    with tf.gfile.Open(filename, "rb") as f:
      if sys.version_info < (3,):
        data = cPickle.load(f)
      else:
        data = cPickle.load(f, encoding="bytes")

    info = np.transpose(data[b"data"].reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
    if images is None:
      images = info
    else:
      images = np.concatenate((images, info))

    info = data[b"fine_labels"] if v100 else data[b"labels"]
    if labels is None:
      labels = info
    else:
      labels = np.concatenate((labels, info))
  return images, labels


def cifar_generator(v100, mode):
  files = None
  if v100:
    files = _CIFAR100_TRAIN if mode != ModeKeys.TEST else _CIFAR100_TEST
  else:
    files = _CIFAR10_TRAIN if mode != ModeKeys.TEST else _CIFAR10_TEST
  maybe_download(files, v100)

  images, labels = read_files(files, v100)
  data = list(zip(images, labels))
  random.shuffle(data)
  
  samples = len(data)
  if mode == ModeKeys.TRAIN:
    data = data[:int(samples * 0.8)]
  elif mode == ModeKeys.EVAL:
    data = data[int(samples * 0.8):]

  image_ph = tf.placeholder(dtype=tf.uint8, shape=(32, 32, 3))
  encoded_ph = tf.image.encode_png(image_ph)

  sess = tf.Session()
  for image, label in data:
    encoded_im = sess.run(encoded_ph, feed_dict={image_ph: image})
    yield {
        "image/encoded": [encoded_im],
        "image/format": [b"png"],
        "image/class/label": [label],
        "image/height": [32],
        "image/width": [32],
        "image/channels": [3]
    }


def generate(train_name, eval_name, test_name, hparams):
  v100 = hparams.data in ["cifar100", "cifar100_tpu"]
  generate_files(
      cifar_generator(v100, mode=ModeKeys.TRAIN), train_name, hparams.data_dir,
      FLAGS.num_shards)
  generate_files(
      cifar_generator(v100, mode=ModeKeys.EVAL), eval_name, hparams.data_dir,
      FLAGS.num_shards)
  generate_files(
      cifar_generator(v100, mode=ModeKeys.TEST), test_name, hparams.data_dir,
      FLAGS.num_shards)
