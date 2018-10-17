import shutil
import os
import sys
import subprocess
import random
import tensorflow as tf
import numpy as np
import time

from .hparams.registry import get_hparams
from .models.registry import get_model
from .data.registry import get_input_fns
from .training.lr_schemes import get_lr
from .training.envs import get_env
from .training import flags
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


def init_flags():
  tf.flags.DEFINE_string("model", None, "Which model to use.")
  tf.flags.DEFINE_string("data", None, "Which data to use.")
  tf.flags.DEFINE_string("env", "local", "Which environment to use.")
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_integer("train_steps", None,
                          "Number of training steps to perform.")
  tf.flags.DEFINE_integer("eval_steps", None,
                          "Number of evaluation steps to perform.")
  tf.flags.DEFINE_integer("eval_every", 1000,
                          "Number of steps between evaluations.")
  tf.flags.DEFINE_integer("copies", 1, "Number of copies of this run.")
  tf.flags.DEFINE_boolean("fresh", False, "Remove output_dir before running.")
  tf.flags.DEFINE_string("train_name", "data-train*",
                         "The train dataset file name.")
  tf.flags.DEFINE_string("test_name", "data-eval*",
                         "The test dataset file name.")

  # TPU flags
  tf.flags.DEFINE_string("tpu_name", "", "Name of TPU(s)")
  tf.flags.DEFINE_integer(
      "tpu_iterations_per_loop", 1000,
      "The number of training steps to run on TPU before"
      "returning control to CPU.")
  tf.flags.DEFINE_integer(
      "tpu_shards", 8, "The number of TPU shards in the system "
      "(a single Cloud TPU has 8 shards.")
  tf.flags.DEFINE_boolean(
      "tpu_summarize", False, "Save summaries for TensorBoard. "
      "Warning: this will slow down execution.")


def init_random_seeds():
  tf.set_random_seed(1234)
  random.seed(1234)
  np.random.seed(1234)


def init_model(FLAGS, i):
  flags.validate_flags(FLAGS)

  tf.reset_default_graph()

  hparams = get_hparams(FLAGS.hparams)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams)

  # set larger eval_every for TPUs to improve utilization
  if FLAGS.env == "tpu":
    FLAGS.eval_every = max(FLAGS.eval_every, 5000)
    hparams.tpu_summarize = FLAGS.tpu_summarize

  tf.logging.warn("\n-----------------------------------------\n"
                  "BEGINNING RUN:\n"
                  "\t hparams: %s\n"
                  "\t output_dir: %s\n"
                  "\t data_dir: %s\n"
                  "-----------------------------------------\n" %
                  (FLAGS.hparams, hparams.output_dir, hparams.data_dir))

  return hparams


def _run(FLAGS, i):
  """Run training, evaluation and inference."""
  hparams = init_model(FLAGS, i)
  hparams.output_dir = os.path.join(hparams.output_dir, str(i))
  if tf.gfile.Exists(hparams.output_dir) and FLAGS.fresh:
    tf.gfile.DeleteRecursively(hparams.output_dir)

  if not tf.gfile.Exists(hparams.output_dir):
    tf.gfile.MakeDirs(hparams.output_dir)
  model_fn = get_model(hparams)
  train_input_fn, eval_input_fn, test_input_fn = get_input_fns(hparams)

  gpu_config = tf.ConfigProto(allow_soft_placement=True)
  gpu_config.gpu_options.allow_growth = True

  if hparams.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu_name)
    master = tpu_cluster_resolver.get_master()
    config = tpu_config.RunConfig(
        master=master,
        evaluation_master=master,
        model_dir=hparams.output_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.tpu_iterations_per_loop,
            num_shards=FLAGS.tpu_shards),
        save_checkpoints_steps=FLAGS.eval_every)
    estimator = tpu_estimator.TPUEstimator(
        use_tpu=hparams.use_tpu,
        model_fn=model_fn,
        model_dir=hparams.output_dir,
        config=config,
        train_batch_size=hparams.batch_size,
        eval_batch_size=hparams.batch_size)
  else:
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=FLAGS.eval_every, session_config=gpu_config)

    estimator = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir=hparams.output_dir,
        config=run_config)

  # output metadata about the run
  with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                      'w') as hparams_file:
    hparams_file.write("{}\n".format(time.time()))
    hparams_file.write("{}\n".format(str(hparams)))

  k = 0
  while k <= int(hparams.train_steps / FLAGS.eval_every):
    estimator.train(train_input_fn, steps=FLAGS.eval_every)
    estimator.evaluate(eval_input_fn, steps=hparams.eval_steps, name="eval")
    estimator.evaluate(test_input_fn, steps=hparams.eval_steps, name="test")

    k = int(estimator.get_variable_value("global_step") / FLAGS.eval_every)
    k += 1


def init():
  init_random_seeds()
  init_flags()


def main(_):
  FLAGS = tf.app.flags.FLAGS

  for i in range(FLAGS.copies):
    _run(FLAGS, i)

    if FLAGS.env in ['gcp', 'tpu']:
      shut_down(FLAGS)


if __name__ == "__main__":
  init()
  tf.app.run()
