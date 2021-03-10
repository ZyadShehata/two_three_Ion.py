import math
from photonscore.python._native_s_ import flim_gaussian_decay as _gaussian_decay


def gaussian_decay(mu, fwhm, tau, channels = 1000):
  return _gaussian_decay(channels, mu, fwhm, tau)

