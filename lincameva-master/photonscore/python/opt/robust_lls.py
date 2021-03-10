import numpy as np

from photonscore.python._native_s_ import WeightFuntion
from photonscore.python._native_s_ import opt_robust_lls as _robust_lls


def robust_lls(A, b, weight = WeightFuntion.BISQUARE):
  if len(A) == 0:
    return np.array([]).reshape(A.shape[1], 0)
  return _robust_lls(A, b, weight)

