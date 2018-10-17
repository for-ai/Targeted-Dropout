import operator
import os
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_boolean("v100", False,
                        "Download CIFAR-100 instead of CIFAR-10.")
tf.flags.DEFINE_integer("num_shards", 1,
                        "The number of output shards to write to.")


def to_example(dictionary):
  features = {}
  for k, v in dictionary.items():
    if len(v) == 0:
      raise Exception("Empty field: %s" % str((k, v)))
    if isinstance(v[0], (int, np.int8, np.int32, np.int64)):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], (float, np.float32)):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], (str, bytes)):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise Exception("Unsupported type: %s" % type(v[0]))
  return tf.train.Example(features=tf.train.Features(feature=features))


def generate_files(generator,
                   output_name,
                   output_dir,
                   num_shards,
                   max_cases=None):
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  writers = []
  for shard in range(num_shards):
    output_filename = "%s-%dof%d" % (output_name, shard + 1, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    writers.append(tf.python_io.TFRecordWriter(output_file))

  counter, shard = 0, 0
  for case in generator:
    if counter % 100 == 0:
      tf.logging.info("Processed %d examples..." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    sequence_example = to_example(case)
    writers[shard].write(sequence_example.SerializeToString())
    shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()
