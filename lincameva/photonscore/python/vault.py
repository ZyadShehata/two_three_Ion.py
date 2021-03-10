from numpy import concatenate
from photonscore.python._native_s_ import Vault as _Vault


class DataGroup:
  def __init__(self):
    self._items = {}
    self._groups = {}

  def _add(self, path, item):
    pp = DataGroup._parse_path(path)
    group = self
    item_path = pp[:-1]
    item_name = pp[-1]
    for group_name in item_path:
      if group_name in group._items:
        raise "Item already exists"
      if group_name in group._groups:
        group = group._groups[group_name]
      else:
        sub_group = DataGroup()
        group.__setattr__(group_name, sub_group)
        group._groups[group_name] = sub_group
        group = sub_group
    group.__setattr__(item_name, item)
    group._items[item_name] = item

  def _parse_path(path):
    return [i for i in path.split('/') if len(i) > 0]


class Dataset:
  def __init__(self, vault, name):
    self._vault = vault
    self._name = name

  def __getitem__(self, key):
    if isinstance(key, slice):
      return self._read(key.start, key.stop, key.step)
    elif isinstance(key, int):
      return self[key:key + 1]
    elif isinstance(key, tuple):
      return concatenate([self[i] for i in key])
    else:
      raise "Unsupported slicing"

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._vault._impl.datasets(self._name)["dtype"]

  @property
  def shape(self):
    return self._vault._impl.datasets(self._name)["shape"]

  def _read(self, start, stop, step):
    shape = self.shape
    if start is None:
      start = 0
    if start < 0:
      start = shape[0] + start
    start = min(shape[0], start)
    if stop is None:
      stop = shape[0] - start
    if stop < 0:
      stop = shape[0] + stop
    count = abs(stop - start)
    data = self._vault.read(self._name, start, count)
    if step is None:
      step = 1
    return data[::step]


class Vault:
  def __init__(
    self,
    path,
    fail_if_exists = False,
    create_if_missing = False,
    read_only = True,
    backend = "",
    backend_options = ""
  ):
    self._impl = _Vault()
    self._path = path
    self._impl.open(
      path,
      fail_if_exists = fail_if_exists,
      create_if_missing = create_if_missing,
      read_only = read_only,
      backend = backend,
      backend_options = backend_options
    )
    self.data = DataGroup()
    dss = self._impl.datasets()
    self._datasets = {}
    for path in dss:
      ds = Dataset(self, path)
      self._datasets[path] = ds
      self.data._add(path, ds)

  def __getitem__(self, key):
    a = self.attr
    if key in a:
      return a[key]
    return self.data[key]

  def __repr__(self):
    return "\n".join(
      [
        "[{0}]".format(self._path),
        "Datasets:\n" + "\n".join(
          [
            "  {0:20} {2:8} {1}".format(
              i, self._datasets[i].shape, "{0}".format(self._datasets[i].dtype)
            ) for i in self._datasets.keys()
          ]
        ),
        "Attributes:\n" +
        "\n".join(["  {0:20} {1}".format(k, v) for k, v in self.attr.items()]),
      ]
    )

  @property
  def attr(self):
    return self._impl.attributes()

  def close(self):
    if self._impl is not None:
      self._impl.close()
      self._impl = None
      self.data = None

  def read(self, dataset, start, count):
    return self._impl.read(dataset, start, count)

