import numpy
from .gaussian_process import GaussianProcessSimple as GP

class MultiOutputGP:
  def __init__(self, X, Y, train=True):
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
      if train:
        gaussian_process.train()
      self.gaussian_processes.append(gaussian_process)

  def set_hypers(self, params_list):
    for params, gp in zip(params_list, self.gaussian_processes):
      gp.covariance.set_hyperparameters(params)
      gp.build_precomputed_data()

  def sample(self, n_samples, X):
    """
    Inputs
    x: n x d numpy array
    Y: (n * n_samples) x m array (block-wise ordering, n blocks of shape n_samples x m)
    """
    Y = []
    for i in range(self.m):
      Y.append(self.gaussian_processes[i].sample(n_samples, X).T.flatten())
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

  def variance(self, X):
    Y = []
    for i in range(self.m):
      Y.append(self.gaussian_processes[i].variance(X).flatten())
    return numpy.array(Y).T