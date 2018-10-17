import tensorflow as tf


class HParams(tf.contrib.training.HParams):
  """Override of TensorFlow's HParams.

  Replaces HParams.add_hparam(name, value) with simple attribute assignment.
    I.e. There is no need to explicitly add an hparam:
      Replace: `hparams.add_hparam("learning_rate", 0.1)`
      With:    `hparams.learning_rate = 0.1`
  """

  def __setattr__(self, name, value):
    """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    # Keys in kwargs are unique, but 'name' could the name of a pre-existing
    # attribute of this object.  In that case we refuse to use it as a
    # hyperparameter name.
    if name[0] == "_":
      object.__setattr__(self, name, value)
      return
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError(
            'Multi-valued hyperparameters cannot be empty: %s' % name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    object.__setattr__(self, name, value)
