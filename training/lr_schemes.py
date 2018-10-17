import tensorflow as tf

_LR = dict()


def register(name):

  def add_to_dict(fn):
    global _LR
    _LR[name] = fn
    return fn

  return add_to_dict


def get_lr(params):
  return _LR[params.lr_scheme](params)


@register("constant")
def constant(params):
  return tf.constant(params.learning_rate)


@register("exp")
def exponential_decay(params, delay=0):
  gs = tf.train.get_global_step() - delay
  return tf.train.exponential_decay(
      params.learning_rate,
      gs,
      params.learning_rate_decay_interval,
      params.learning_rate_decay_rate,
      staircase=params.staircased)


@register("lin")
def linear_decay(params, delay=0):
  gs = tf.train.get_global_step() - delay
  return (params.learning_rate -
          (tf.to_float(gs) /
           (params.train_steps - delay)) * params.learning_rate)


@register("delay_exp")
def delayed_exponential_decay(params):
  gs = tf.train.get_global_step()
  d = params.delay
  return tf.cond(
      tf.greater(gs, d), lambda: exponential_decay(params, delay=d),
      lambda: params.learning_rate)


@register("delay_lin")
def delayed_linear_decay(params):
  gs = tf.train.get_global_step()
  d = params.delay
  return tf.cond(
      tf.greater(gs, d), lambda: linear_decay(params, delay=d),
      lambda: params.learning_rate)


@register("resnet")
def resnet(_):
  gs = tf.train.get_global_step()
  return tf.cond(
      tf.less(gs, 40000),
      lambda: 0.1,
      lambda: tf.cond(
          tf.less(gs, 60000),
          lambda: 0.01,
          lambda: tf.cond(
              tf.less(gs, 80000),
              lambda: 0.001,
              lambda: 0.0001)))


@register("lenet")
def lenet(_):
  gs = tf.train.get_global_step()
  return tf.cond(
      tf.less(gs, 80000), lambda: 0.05,
      lambda: tf.cond(tf.less(gs, 120000), lambda: 0.005, lambda: 0.0005))


@register("steps")
def stepped_lr(params):
  gs = tf.train.get_global_step()
  lr = params.lr_values[-1]
  for step, value in reversed(list(zip(params.lr_steps, params.lr_values))):
    lr = tf.cond(tf.greater(gs, step), lambda: lr, lambda: value)
  return lr


@register("warmup_linear_decay")
def warmup_linear_decay(params):
  gs = tf.train.get_global_step()
  d = params.delay
  warmup_steps = params.warmup_steps
  inv_base = tf.exp(tf.log(0.01) / warmup_steps)
  inv_decay = inv_base**(warmup_steps - tf.to_float(gs))

  return tf.cond(
      tf.greater(gs, warmup_steps), lambda: linear_decay(params, delay=d),
      lambda: inv_decay * params.learning_rate)


@register("warmup_constant")
def warmup_constant(params):
  gs = tf.train.get_global_step()
  warmup_steps = params.warmup_steps
  inv_base = tf.exp(tf.log(0.01) / warmup_steps)
  inv_decay = inv_base**(warmup_steps - tf.to_float(gs))

  return tf.cond(
      tf.greater(gs, warmup_steps), lambda: constant(params),
      lambda: inv_decay * params.learning_rate)


@register("warmup_exponential_decay")
def warmup_exponential_decay(params):
  gs = tf.train.get_global_step()
  d = params.delay
  warmup_steps = params.warmup_steps
  inv_base = tf.exp(tf.log(0.01) / warmup_steps)
  inv_decay = inv_base**(warmup_steps - tf.to_float(gs))

  return tf.cond(
      tf.greater(gs, warmup_steps), lambda: exponential_decay(params, delay=d),
      lambda: inv_decay * params.learning_rate)


@register("cosine")
def cosine_annealing(params):
  from numpy import pi

  gs = tf.train.get_global_step()
  gs = tf.minimum(gs, params.learning_rate_cosine_cycle_steps)
  cosine_decay = 0.5 * (1 + tf.cos(
      pi * tf.to_float(gs) / params.learning_rate_cosine_cycle_steps))
  decayed = (1 - params.cosine_alpha) * cosine_decay + params.cosine_alpha
  decayed_learning_rate = params.learning_rate * decayed
  return decayed_learning_rate
