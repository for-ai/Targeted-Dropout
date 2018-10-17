import tensorflow as tf

_OPTIMIZER = dict()


def register(name):

  def add_to_dict(fn):
    global _OPTIMIZER
    _OPTIMIZER[name] = fn
    return fn

  return add_to_dict


def get_optimizer(lr, params):
  optimizer = _OPTIMIZER[params.optimizer](lr, params)
  if params.use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
  return optimizer


@register("sgd")
def sgd(lr, params):
  return tf.train.GradientDescentOptimizer(lr)


@register("adam")
def adam(lr, params):
  return tf.train.AdamOptimizer(lr, beta1=params.beta1, beta2=params.beta2)


@register("adagrad")
def adagrad(lr, params):
  return tf.train.AdagradOptimizer(lr)


@register("momentum")
def momentum(lr, params):
  return tf.train.MomentumOptimizer(
      lr, momentum=params.momentum, use_nesterov=params.use_nesterov)
