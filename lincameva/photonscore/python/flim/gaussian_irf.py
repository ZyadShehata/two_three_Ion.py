import math

from photonscore.python._native_s_ import flim_gaussian_irf as _gaussian_irf


def gaussian_irf(mu, fwhm, channels = 1000):
  return _gaussian_irf(channels, mu, fwhm, channels)

