import tensorflow as tf
import os
import numpy as np

from ...hparams.registry import get_hparams
from ...models.registry import get_model
from ...data.registry import get_input_fns
from ...training import flags
from .prune import get_prune_fn, get_current_weights, get_louizos_masks, get_smallify_masks, prune_weights, is_prunable_weight


def init_flags():
  tf.flags.DEFINE_string("model", None, "Which model to use.")
  tf.flags.DEFINE_string("data", None, "Which data to use.")
  tf.flags.DEFINE_string("env", None, "Which environment to use.")
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_integer("train_steps", 10000,
                          "Number of training steps to perform.")
  tf.flags.DEFINE_integer("eval_steps", 100,
                          "Number of evaluation steps to perform.")
  tf.flags.DEFINE_integer("eval_every", 1000,
                          "Number of steps between evaluations.")
  tf.flags.DEFINE_integer("copy", 0, "Model copy to run on.")
  tf.flags.DEFINE_string(
      "post_weights_dir", "",
      "folder of the weights, if not set defaults to output_dir")
  tf.flags.DEFINE_string("prune_percent", "0.5",
                         "percent of weights to prune, comma separated")
  tf.flags.DEFINE_string("prune", "one_shot", "one_shot or fisher")
  tf.flags.DEFINE_boolean("variational", False, "use evaluate")
  tf.flags.DEFINE_string("eval_file", "eval_prune_results",
                         "file to put results")


def eval_model(FLAGS, hparam_name):
  hparams = get_hparams(hparam_name)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams)
  hparams.output_dir = os.path.join(FLAGS.output_dir, hparam_name,
                                    str(FLAGS.copy))

  model_fn = get_model(hparams)
  _, _, test_input_fn = get_input_fns(hparams, generate=False)

  run_config = tf.contrib.learn.RunConfig(
      save_checkpoints_steps=FLAGS.eval_every)

  features, labels = test_input_fn()
  sess = tf.Session()
  tf.train.create_global_step()
  model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
  saver = tf.train.Saver()
  print("Loading model from...", hparams.output_dir)
  saver.restore(sess, tf.train.latest_checkpoint(hparams.output_dir))

  evals = []
  prune_percents = [float(i) for i in FLAGS.prune_percent.split(",")]

  for prune_percent in prune_percents:
    post_weights = get_current_weights(sess)
    if "louizos" in hparam_name:
      louizos_masks = get_louizos_masks(sess, post_weights)
    else:
      louizos_masks = None

    if "smallify" in hparam_name:
      smallify_masks = get_smallify_masks(sess, post_weights)
    else:
      smallify_masks = None

    if prune_percent > 0.0:
      prune_fn = get_prune_fn(FLAGS.prune)(k=prune_percent)
      post_weights_pruned, weight_counts = prune_weights(
          prune_fn, post_weights, louizos_masks, smallify_masks, hparams)
      print("current weight counts at {}: {}".format(prune_percent,
                                                     weight_counts))

      print("there are ", len(tf.trainable_variables()), " weights")
      for v in tf.trainable_variables():
        if is_prunable_weight(v):
          assign_op = v.assign(
              np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
          sess.run(assign_op)

    saver.save(sess, os.path.join(hparams.output_dir, "tmp", "model"))
    estimator = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir=os.path.join(hparams.output_dir, "tmp"),
        config=run_config)
    print("Processing pruning {} of weights".format(prune_percent))
    acc = estimator.evaluate(test_input_fn, FLAGS.eval_steps)['acc']
    evals.append(acc)
  return evals


def _run(FLAGS):
  eval_file = open(FLAGS.eval_file, "w")

  hparams_list = FLAGS.hparams.split(",")
  total_evals = {}
  for hparam_name in hparams_list:
    evals = eval_model(FLAGS, hparam_name)

    print(hparam_name, ":", evals)
    eval_file.writelines("{}:{}\n".format(hparam_name, evals))
    total_evals[hparam_name] = evals
    tf.reset_default_graph()

  print("processed results: total_evals")
  eval_file.close()


if __name__ == "__main__":
  init_flags()
  FLAGS = tf.app.flags.FLAGS
  _run(FLAGS)
