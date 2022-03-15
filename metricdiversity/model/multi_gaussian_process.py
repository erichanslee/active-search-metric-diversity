import numpy
from .gaussian_process import GaussianProcessSimple as GP

class MultiOutputGP:
  def __init__(self, X, Y):
    """
    X: n by d numpy array
    Y: n by m numpy array
    Y here is vector valued, not scalar valued
    """
    self.gaussian_processes = []
    self.n = X.shape[0]
    self.d = X.shape[1]
    self.m = Y.shape[1]
    for i in range(self.m):
      gaussian_process = GP(X, Y[:, i])
      gaussian_process.train()
      self.gaussian_processes.append(gaussian_process)

  def sample(self, n_samples, x):
    """
    Inputs
    x: 1 by d numpy array
    Y: n_samples by m array
    """
    Y = []
    for i in range(self.m):
      Y.append(self.gaussian_processes[i].sample(n_samples, x).flatten())
    return numpy.array(Y).T

  def get_historical_data(self):
    Y = []
    X = None
    for gp in self.gaussian_processes:
      X, y = gp.get_historical_data()
      Y.append(y)
    return (
      X,
      numpy.array(Y).T
    )

  def predict(self, X):
    Y = []
    for i in range(self.m):
      Y.append(self.gaussian_processes[i].predict(X).flatten())
    return numpy.array(Y).T