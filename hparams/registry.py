import tensorflow as tf

_HPARAMS = dict()


def register(fn):
  global _HPARAMS
  _HPARAMS[fn.__name__] = fn()
  return fn


def get_hparams(hparams_list):
  """Fetches a merged group of hyperparameter sets (chronological priority)."""
  final = tf.contrib.training.HParams()
  for name in hparams_list.split("-"):
    curr = _HPARAMS[name]
    final_dict = final.values()
    for k, v in curr.values().items():
      if k not in final_dict:
        final.add_hparam(k, v)
      elif final_dict[k] is None:
        setattr(final, k, v)
  return final
