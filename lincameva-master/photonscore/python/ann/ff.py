from photonscore.python.ann.train import weights_size
import numpy as np


class FeedForward:
  def __init__(self, layers, weights = None):
    # Create the structure
    self.W = []
    self.b = []
    ll = [l for l in layers]
    ll.append(1)
    for i in range(len(ll) - 1):
      a = ll[i]
      b = ll[i + 1]
      self.W.append(np.zeros(shape = [a, b]))
      self.b.append(np.zeros(shape = b))
    if weights is not None:
      self.weights = weights

  @property
  def layers(self):
    return [w.shape[0] for w in self.W[:]]

  @property
  def weights(self):
    res = np.zeros(shape = weights_size(self.layers))
    offset = 0
    for i in range(len(self.W)):
      n = self.b[i].size
      res[offset:offset + n] = self.b[i].flatten()
      offset += n
      n = self.W[i].size
      res[offset:offset + n] = self.W[i].flatten()
      offset += n
    return res

  @weights.setter
  def weights(self, value):
    v = np.array(value)[:]
    offset = 0
    for i in range(len(self.W)):
      n = self.b[i].size
      self.b[i][:] = v[offset:offset + n].reshape(self.b[i].shape)
      offset += n
      n = self.W[i].size
      self.W[i][:] = v[offset:offset + n].reshape(self.W[i].shape)
      offset += n

  def evaluate(self, x):
    for i in range(len(self.W)):
      if i > 0:
        x = np.tanh(x)
      x = np.dot(self.W[i].T, x.T).T
      x += np.repeat(self.b[i], x.shape[0]).reshape(x.T.shape).T
    return x.flatten()

  def __call__(self, x):
    return self.evaluate(x)

