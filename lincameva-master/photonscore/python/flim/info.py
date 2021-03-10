from photonscore.python.vault import Vault


class Info:
  def __init__(self, filename, duration, total_counts):
    self.filename = filename
    self.duration = duration
    self.total_counts = total_counts

  def __repr__(self):
    return "\n".join(
      [
        "filename = '{}'".format(self.filename),
        "duration = {} seconds".format(self.duration),
        "total_counts = {}".format(self.total_counts),
      ]
    )


def info(path):
  v = Vault(path)
  return Info(
    filename = path,
    total_counts = min(
      v.data.photons.x.shape[0],
      v.data.photons.y.shape[0],
      v.data.photons.dt.shape[0],
    ),
    duration = v.data.photons.ms.shape[0] / 1000
  )

