import numpy as np

from photonscore.python._native_s_ import ann_train as _ann_train
from photonscore.python._native_s_ import LossFunctionName


def _cb(summary):
  print(summary)


def weights_size(layers):
  res = 0
  ll = layers[:]
  ll.append(1)
  for i in range(0, len(ll) - 1):
    res += (ll[i] + 1) * ll[i + 1]
  return res


def train(
  layers,
  x0,
  inputs,
  target,
  threads = 4,
  iterations = 100,
  loss = None,
  log_callback = None
):
  if log_callback is None:
    log_callback = _cb

  if loss is None:
    loss = (LossFunctionName.NO_LOSS, 0.0)

  return _ann_train(
    np.array(layers, dtype = "intp")[:],
    np.array(x0, dtype = "f8")[:],
    np.array(inputs, dtype = "f8")[:],
    np.array(target, dtype = "f8")[:],
    loss[0],
    loss[1],
    threads,
    iterations,
    log_callback,
  )

