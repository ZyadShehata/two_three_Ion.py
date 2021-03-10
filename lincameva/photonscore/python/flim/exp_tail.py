import math
import collections
import numpy as np


def exp_tail(tau, channels):
  if not isinstance(tau, collections.Iterable):
    tau = np.array([tau])
  else:
    if not isinstance(tau, np.ndarray):
      tau = np.array(tau)
  res = np.zeros([len(tau), channels])
  res[:, 0] = np.exp(-1 / tau)

