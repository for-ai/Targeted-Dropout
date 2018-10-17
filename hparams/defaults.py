import tensorflow as tf

from .registry import register
from .utils import HParams


@register
def default():
  return HParams(
      model=None,
      data=None,
      shuffle_data=True,
      data_augmentations=None,
      train_steps=100000,
      eval_steps=100,
      type="image",
      batch_size=64,
      learning_rate=0.01,
      lr_scheme="constant",
      initializer="glorot_normal_initializer",
      delay=0,
      staircased=False,
      learning_rate_decay_interval=2000,
      learning_rate_decay_rate=0.1,
      clip_grad_norm=1.0,
      l2_loss=0.0,
      prune_val=0.8,
      label_smoothing=0.1,
      use_tpu=False,
      momentum=0.9,
      init_scheme="random",
      warmup_steps=10000,
      use_nesterov=False,
      louizos_cost=0.0,
      l1_norm=0.0,
      thresh=2.5,
      fixed=False,
      var_scale=1,
      klscale=1.0,
      ard_cost=0.0,
      logit_packing=0.0,
      logit_squeezing=0.0,
      clp=0.0,
      logit_bound=None,
      dropout_type=None,
      smallify=0.0,
      smallify_delay=1000,
      linear_drop_rate=False,
      weight_decay_and_noise=False,
      dropout_delay_steps=5000,
      grad_noise_scale=0.0,
      td_nines=0,
      targ_cost=1.0,
      aparams="",
      channels=1)


@register
def default_cifar10():
  hps = default()
  hps.data = "cifar10"
  hps.data_augmentations = ["image_augmentation"]

  hps.input_shape = [32, 32, 3]
  hps.output_shape = [10]
  hps.channels = 3
  hps.num_classes = 10

  return hps


@register
def default_cifar100():
  hps = default()
  hps.data = "cifar100"
  hps.data_augmentations = ["image_augmentation"]

  hps.input_shape = [32, 32, 3]
  hps.output_shape = [100]
  hps.num_classes = 100
  hps.channels = 3

  return hps
