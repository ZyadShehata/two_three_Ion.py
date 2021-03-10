import math

from photonscore.python._native_s_ import flim_convolve as _convolve

def convolve(irf, irf_shift, tau, tau_ref = None, channels = None):
  tau_ref = 0.0 if tau_ref is None else math.fabs(tau_ref)
  channels = len(irf) if channels is None else channels
  return _convolve(irf, irf_shift, tau, tau_ref, channels)

