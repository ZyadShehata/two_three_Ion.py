import math

from photonscore.python.flim.sort import sort
from photonscore.python.histogram import histogram
from photonscore.python.vault import Vault


def _format_km(x):
  if x > 10 * 1000 * 1000:
    return "{0:0.2f} M".format(x / 1000000)
  if x > 10 * 1000:
    return "{0:0.2f} K".format(x / 1000)
  return "{}".format(x)


class ReadData:
  def __init__(self, x, y, dt):
    self.x = x
    self.y = y
    self.dt = dt

  def intensity(self, pixels = 512):
    return histogram(self.x, 0, 4096, pixels, self.y)

  def decay(self):
    return histogram(self.dt, 0, 4096, 4096)

  def sort(self, pixels = 512):
    return sort(self.x, 0, 4096, pixels, self.y, self.dt)

  @property
  def total_counts(self):
    return len(self.x)

  def __repr__(self):
    return "\n".join(
      [
        "total_counts: {}".format(_format_km(self.total_counts)),
        "x: {}".format(self.x.__repr__()),
        "y: {}".format(self.y.__repr__()),
        "dt: {}".format(self.dt.__repr__()),
      ]
    )

  def __getitem__(self, key):
    if isinstance(key, int):
      return self[key:key + 1]
    return ReadData(
      x = self.x[key],
      y = self.y[key],
      dt = self.dt[key],
    )


def read(path, seconds = None, events = None):
  v = Vault(path)
  n = min(
    v.data.photons.x.shape[0],
    v.data.photons.y.shape[0],
    v.data.photons.dt.shape[0],
  )
  a = b = 0
  if seconds is None and events is None:
    # Read all
    (a, b) = (0, n)
  elif not seconds is None:
    # Lookup ms index
    msi = v.data.photons.ms[:]
    a = int(math.floor(seconds[0] * 1000))
    b = int(math.floor(seconds[1] * 1000))
    a = min(len(msi) - 1, max(0, a))
    b = min(len(msi) - 1, max(0, b))
    (a, b) = (min(n, msi[a]), min(n, msi[b]))
  else:
    # Range
    a = events[0]
    b = events[1]
  return ReadData(
    x = v.data.photons.x[a:b],
    y = v.data.photons.y[a:b],
    dt = v.data.photons.dt[a:b],
  )

