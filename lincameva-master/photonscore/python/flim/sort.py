import math
import numpy as np

from photonscore.python._native_s_ import flim_sort as _sort
from photonscore.python._native_s_ import flim_median as _median
from photonscore.python._native_s_ import flim_mean as _mean


class SortResult:
  def __init__(self, counts, dt):
    self.counts = counts
    self.dt = dt

  def _mean_or_median(self, dt_range, f):
    if dt_range is None:
      dt_range = (0, 4096)
    return f(self.counts, self.dt, dt_range[0], dt_range[1])

  def mean(self, dt_range = None):
    return self._mean_or_median(dt_range, _mean)

  def median(self, dt_range = None):
    return self._mean_or_median(dt_range, _median)


def sort(x, x_min, x_max, x_bins, y, *args):
  if len(args) == 1:
    y_min = x_min
    y_max = x_max
    y_bins = x_bins
    dt = args[0]
  elif len(args) == 4:
    y_min = args[0]
    y_max = args[1]
    y_bins = args[2]
    dt = args[3]
  else:
    raise ValueError("Invalid number of arguments")

  (counts, t) = _sort(x, x_min, x_max, x_bins, y, y_min, y_max, y_bins, dt)
  return SortResult(counts, t)

