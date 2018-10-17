import os

import tensorflow as tf

_INPUT_FNS = dict()
_GENERATORS = dict()


def register(name, generator):

  def add_to_dict(fn):
    global _INPUT_FNS
    global _GENERATORS
    _INPUT_FNS[name] = fn
    _GENERATORS[name] = generator
    return fn

  return add_to_dict


def get_input_fns(hparams, generate=True):
  train_path = os.path.join(hparams.data_dir, "train*")
  eval_path = os.path.join(hparams.data_dir, "eval*")
  test_path = os.path.join(hparams.data_dir, "test*")

  if generate:
    if not tf.gfile.Exists(hparams.data_dir):
      tf.gfile.MakeDirs(hparams.data_dir)

    # generate if train doesnt exist
    maybe_generate(train_path, hparams)
    maybe_generate(eval_path, hparams)
    maybe_generate(test_path, hparams)

  train_path = tf.gfile.Glob(train_path)
  eval_path = tf.gfile.Glob(eval_path)
  test_path = tf.gfile.Glob(test_path)

  input_fn = _INPUT_FNS[hparams.data]
  return input_fn(
      train_path, hparams, training=True), input_fn(
          eval_path, hparams, training=False), input_fn(
          test_path, hparams, training=False)


def get_dataset(hparams):
  train_path = os.path.join(hparams.data_dir, "train*")
  eval_path = os.path.join(hparams.data_dir, "eval*")
  test_path = os.path.join(hparams.data_dir, "test*")
  maybe_generate(train_path, hparams)
  maybe_generate(eval_path, hparams)
  maybe_generate(test_path, hparams)
  return train_path, eval_path, test_path


def maybe_generate(check_path, hparams):
  if not tf.gfile.Glob(check_path):
    _GENERATORS[hparams.data]("train", "eval", "test", hparams)
