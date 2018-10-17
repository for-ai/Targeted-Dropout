import tensorflow as tf

_AUGMENTATIONS = dict()


def register(fn):
  global _AUGMENTATIONS
  _AUGMENTATIONS[fn.__name__] = fn
  return fn


def get_augmentation(name, params, training):

  def fn(*args, **kwargs):
    return _AUGMENTATIONS[name](
        *args, **kwargs, training=training, params=params)

  return fn


@register
def image_augmentation(image, label, training, params):
  """Image augmentation suitable for CIFAR-10/100.
  As described in https://arxiv.org/pdf/1608.06993v3.pdf (page 5).
  Args:
    images: a Tensor.
  Returns:
    Tensor of the same shape as images.
  """
  if training:
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)

  image = tf.image.per_image_standardization(image)
  return image, label


@register
def load_images(example, training, params):
  data_fields_to_features = {
      "image/encoded": tf.FixedLenFeature((), tf.string),
      "image/format": tf.FixedLenFeature((), tf.string),
      "image/class/label": tf.FixedLenFeature((), tf.int64)
  }

  example = tf.parse_single_example(example, data_fields_to_features)
  image = example["image/encoded"]
  image = tf.image.decode_png(image, channels=params.channels, dtype=tf.uint8)
  image = tf.reshape(image, params.input_shape)
  image = tf.to_float(image) / 255.

  label = tf.to_int32(example["image/class/label"])
  label = tf.one_hot(label, params.output_shape[-1])

  return image, label
