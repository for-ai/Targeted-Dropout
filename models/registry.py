from ..training.lr_schemes import get_lr

import tensorflow as tf

_MODELS = dict()


def register(name):

  def add_to_dict(fn):
    global _MODELS
    _MODELS[name] = fn
    return fn

  return add_to_dict


def get_model(hparams):

  def model_fn(features, labels, mode, params=None):
    lr = tf.constant(0.0)
    if mode == tf.estimator.ModeKeys.TRAIN:
      lr = get_lr(hparams)
    return _MODELS[hparams.model](hparams, lr)(features, labels, mode, params)

  return model_fn
