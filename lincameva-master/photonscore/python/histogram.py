import numpy as np
from photonscore.python._native_s_ import hist_1d
from photonscore.python._native_s_ import hist_2d


def histogram(
  x, x_min, x_max, x_bins, y = None, y_min = None, y_max = None, y_bins = None
):
  x = x if isinstance(x, np.ndarray) else np.array(x)
  if y is None:
    return hist_1d(x, x_min, x_max, x_bins)

  y = y if isinstance(y, np.ndarray) else np.array(y)
  y_min = x_min if y_min is None else y_min
  y_max = x_max if y_max is None else y_max
  y_bins = x_bins if y_bins is None else y_bins
  return hist_2d(x, x_min, x_max, x_bins, y, y_min, y_max, y_bins)

