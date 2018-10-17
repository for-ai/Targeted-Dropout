import tensorflow as tf

_ACTIVATION = dict()


def register(name):

  def add_to_dict(fn):
    global _ACTIVATION
    _ACTIVATION[name] = fn
    return fn

  return add_to_dict


def get_activation(params):
  return _ACTIVATION[params.activation](params)


@register("relu")
def relu(params):
  return tf.nn.relu


@register("brelu")
def brelu(params):

  def fn(a):
    idx = tf.range(a.shape[-1])
    idx = tf.mod(idx, 2)
    idx = tf.cast(idx, tf.bool)

    even = tf.nn.relu(a)
    odd = -tf.nn.relu(-a)

    return tf.where(idx, odd, even)

  return fn


@register("selu")
def selu(params):
  return tf.nn.selu


@register("elu")
def elu(params):
  return tf.nn.elu


@register("sigmoid")
def sigmoid(params):
  return tf.nn.sigmoid


@register("swish")
def swish(params):
  return lambda x: tf.nn.sigmoid(x) * x


@register("tanh")
def tanh(params):
  return tf.nn.tanh
