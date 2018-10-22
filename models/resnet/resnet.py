import tensorflow as tf
import numpy as np
from ..utils.activations import get_activation
from ..utils.dropouts import get_dropout, smallify_dropout
from ..utils.initializations import get_init
from ..registry import register
from ..utils import model_utils
from ..utils.model_utils import ModeKeys
from ...training import tpu
import six

from tensorflow.contrib.tpu.python.tpu import tpu_estimator, tpu_optimizer


@register("resnet")
def get_resnet(hparams, lr):
  """Callable model function compatible with Experiment API.

          Args:
            params: a HParams object containing values for fields:
              use_bottleneck: bool to bottleneck the network
              relu_leakiness: leakiness factor for relu units
              num_residual_units: number of residual units
              num_classes: number of classes
              batch_size: batch size
              weight_decay_rate: weight decay rate
          """

  def resnet(features, labels, mode, params):
    if hparams.use_tpu and 'batch_size' in params.keys():
      hparams.batch_size = params['batch_size']

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    def _residual(x, out_filter, stride, projection=False):
      """Residual unit with 2 sub layers."""

      orig_x = x

      with tf.variable_scope('sub1'):
        x = model_utils.conv(
            orig_x,
            3,
            out_filter,
            hparams,
            is_training=is_training,
            strides=stride,
            name='conv1',
            schit_layer=hasattr(hparams, 'lipschitz_constant'))
        x = model_utils.batch_norm(x, is_training)
        x = tf.nn.relu(x)
      with tf.variable_scope('sub2'):
        x = model_utils.conv(
            x,
            3,
            out_filter,
            hparams,
            is_training=is_training,
            strides=[1, 1, 1, 1],
            name='conv2',
            schit_layer=hasattr(hparams, 'lipschitz_constant'))

      if projection:
        orig_x = model_utils.conv(
            orig_x,
            1,
            out_filter,
            hparams,
            is_training=is_training,
            strides=stride,
            name="shortcut")

      if hasattr(hparams, 'lipschitz_constant') and hasattr(
          hparams, 'multiply_identity') and hparams.multiply_identity:
        orig_x = hparams.lipschitz_constant * orig_x

      x += orig_x
      if hparams.dropout_type is not None and not "variational" in hparams.dropout_type:
        x = model_utils.batch_norm(x, is_training)
        x = tf.nn.relu(x)

      tf.logging.debug('image after unit %s', x.get_shape())
      return x

    def _l1():
      """L1 weight decay loss."""
      if hparams.l1_norm == 0:
        return 0

      costs = []
      for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
          costs.append(tf.reduce_mean(tf.abs(var)))

      return tf.multiply(hparams.l1_norm, tf.add_n(costs))

    def _fully_connected(x, out_dim):
      """FullyConnected layer for final output."""
      x = tf.reshape(x, [hparams.batch_size, 128])
      prev_dim = np.product(x.get_shape().as_list()[1:])
      w = tf.get_variable('DW', [prev_dim, out_dim])
      b = tf.get_variable(
          'biases', [out_dim], initializer=tf.constant_initializer())
      return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(x):
      assert x.get_shape().ndims == 4
      return tf.reduce_mean(x, [1, 2])

    def _stride_arr(stride):
      """Map a stride scalar to the stride array for tf.nn.conv2d."""
      return [1, stride, stride, 1]

    if mode == ModeKeys.PREDICT or mode == ModeKeys.ATTACK:
      if "labels" in features:
        labels = features["labels"]

    with tf.variable_scope("resnet", initializer=get_init(hparams)):
      hparams.mode = mode
      # 3 and 16 picked from example implementation
      with tf.variable_scope('init'):
        x = features["inputs"]
        x = model_utils.conv(
            x,
            3,
            16,
            hparams,
            strides=_stride_arr(1),
            dropout=False,
            name='init_conv')

      strides = [1, 2, 2]
      res_func = _residual
      filters = [16, 32, 64, 128]

      with tf.variable_scope('unit_1_0'):
        x = res_func(x, filters[1], _stride_arr(strides[0]), True)

      for i in six.moves.range(1, hparams.num_residual_units):
        with tf.variable_scope('unit_1_%d' % i):
          x = res_func(x, filters[1], _stride_arr(1), False)

      with tf.variable_scope('unit_2_0'):
        x = res_func(x, filters[2], _stride_arr(strides[1]), True)

      for i in six.moves.range(1, hparams.num_residual_units):
        with tf.variable_scope('unit_2_%d' % i):
          x = res_func(x, filters[2], _stride_arr(1), False)

      with tf.variable_scope('unit_3_0'):
        x = res_func(x, filters[3], _stride_arr(strides[2]), True)

      for i in six.moves.range(1, hparams.num_residual_units):
        with tf.variable_scope('unit_3_%d' % i):
          x = res_func(x, filters[3], _stride_arr(1), False)

      with tf.variable_scope('unit_last'):
        x = model_utils.batch_norm(x, is_training)
        x = tf.nn.relu(x)
        x = _global_avg_pool(x)

      with tf.variable_scope('logit'):
        logits = _fully_connected(x, hparams.num_classes)
        predictions = tf.nn.softmax(logits)

      if mode in [ModeKeys.PREDICT, ModeKeys.ATTACK]:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'classes': tf.argmax(predictions, axis=1),
                'logits': logits,
                'probabilities': predictions,
            })

      with tf.variable_scope('costs'):
        xent = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)
        cost = tf.reduce_mean(xent, name='xent')
        if is_training:
          if not hparams.dropout_type or "variational" not in hparams.dropout_type:
            cost += model_utils.weight_decay(hparams)

          tf.summary.scalar("avg_logit",
                            tf.reduce_mean(tf.reduce_max(tf.abs(logits), -1)))

          cost += _l1()

          if hparams.dropout_type is not None:
            if "louizos" in hparams.dropout_type:
              cost += hparams.louizos_cost * model_utils.louizos_complexity_cost(
                  hparams) / 50000

            if "variational" in hparams.dropout_type:
              # prior DKL part of the ELBO
              graph = tf.get_default_graph()
              node_defs = [
                  n for n in graph.as_graph_def().node if 'log_alpha' in n.name
              ]
              log_alphas = [
                  graph.get_tensor_by_name(n.name + ":0") for n in node_defs
              ]
              print([
                  n.name
                  for n in graph.as_graph_def().node
                  if 'log_alpha' in n.name
              ])
              print("found %i logalphas" % len(log_alphas))
              divergences = [model_utils.dkl_qp(la) for la in log_alphas]
              # combine to form the ELBO
              N = float(50000)
              dkl = tf.reduce_sum(tf.stack(divergences))

              warmup_steps = 50000
              dkl = (1. / N) * dkl * tf.minimum(
                  1.0,
                  tf.to_float(tf.train.get_global_step()) /
                  warmup_steps) * hparams.var_scale
              cost += dkl
              tf.summary.scalar("dkl", dkl)

          if hparams.ard_cost > 0.0:
            cost += model_utils.ard_cost() * hparams.ard_cost

          if hparams.smallify > 0.0:
            cost += model_utils.switch_loss() * hparams.smallify

    # Summaries
    # ========================
    tf.summary.scalar("louizos_cost", hparams.louizos_cost)
    tf.summary.scalar("total_nonzero", model_utils.nonzero_count())
    all_weights = tf.concat(
        [
            tf.reshape(v, [-1])
            for v in tf.trainable_variables()
            if "DW" in v.name
        ],
        axis=0)
    tf.summary.histogram("weights", all_weights)
    # ========================

    return model_utils.model_top(labels, predictions, cost, lr, mode, hparams)

  return resnet
