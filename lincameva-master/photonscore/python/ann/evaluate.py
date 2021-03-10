import numpy as np
from photonscore.python._native_s_ import ann_evaluate as _ann_evaluate

def evaluate(layers, weights, inputs):
  return _ann_evaluate(layers, np.array(weights), np.array(inputs))

