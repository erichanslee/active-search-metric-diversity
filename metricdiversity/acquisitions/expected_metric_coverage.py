import numpy
import scipy

from ..model.multi_gaussian_process import MultiOutputGP
from ..model.domain import TensorProductDomain, ClosedInterval
NUM_MC_SAMPLES = 512

class ThresholdBox:
  def __init__(self, ub, lb):
    """
    Input
    ub, lb: array or list (shouldn't matter) of length m
    """
    self.ub = ub
    self.lb = lb

  def points_in_box(self, Y):
    """
    Input
    Y: n x m set of points
    Output
    Y_in: subset of points in Y that are within threshold values
    """
    idx_under_ub = numpy.all(Y <= self.ub, axis=1)
    Y = Y[idx_under_ub]
    idx_over_lb = numpy.all(Y >= self.lb, axis=1)
    Y = Y[idx_over_lb]
    return Y


class ExpectedMetricCoverage:

  def __init__(self, gaussian_process, ub, lb, punchout_radius, opt_domain=None):
    """
    Input
    gaussian_process: assumed to be an object of type MultiOutputGP
    ub: array of length m (num objectives), upper bounds
    lb: array of length m (num objectives), lower bounds
    punchout_radius: self explanatory
    opt_domain: assumed to be unit hypercube by default, of type TensorProductDomain

    """
    self.gaussian_process = gaussian_process
    self.threshold_box = ThresholdBox(ub, lb)
    self.punchout_radius = punchout_radius
    if opt_domain is None:
      self.opt_domain = TensorProductDomain([ClosedInterval(0, 1)] * self.gaussian_process.d)

  def compute_expected_utility(self, x):
    """
    Compute value via MC by generating sampling from y distribution at a point x, and then
    tossing points that are either outside the threshold box or too close to an existing metric value
    """
    x = numpy.atleast_2d(x)
    Y_samples = self.gaussian_process.sample(NUM_MC_SAMPLES, x)
    Y_in_box = self.threshold_box.points_in_box(Y_samples)
    _, Y_obs = self.gaussian_process.get_historical_data()
    dist_matrix = scipy.spatial.distance.cdist(Y_in_box, Y_obs)
    idx_outside_range = numpy.all(dist_matrix > self.punchout_radius, axis=1)
    return sum(idx_outside_range) / NUM_MC_SAMPLES

  def tiebreak_suggestions(self, X):
    """
    In the event that mutiple points have the same utility, we tiebreak by selecting the point
    whose mean y value is furthest away from the other observed metric values (in the same vein that ECI does it)
    """

    _, Y_obs = self.gaussian_process.get_historical_data()
    Y_preds = self.gaussian_process.predict(X)


    idx_under_ub = numpy.all(Y_preds <= self.threshold_box.ub, axis=1)
    idx_over_lb = numpy.all(Y_preds >= self.threshold_box.lb, axis=1)
    idxs = numpy.logical_and(idx_over_lb, idx_under_ub)

    Y_preds_in_box = Y_preds[idxs]
    X_preds_in_box = X[idxs]

    dist_matrix = scipy.spatial.distance.cdist(Y_preds_in_box, Y_obs)
    closest_distances = numpy.min(dist_matrix, axis=1)
    idx = numpy.argmax(closest_distances)
    return X_preds_in_box[idx, :]

  def get_suggestion(self):
    n_points = 512
    X = self.opt_domain.generate_quasi_random_points_in_domain(n_points)
    utility_values = [self.compute_expected_utility(x) for x in X]

    # If all utility values are low...
    if numpy.sum(utility_values) < 1e-6:
      return self.tiebreak_suggestions(X)
    else:
      idxs = numpy.argmax(utility_values) # need to check for multiple argmaxes
      return X[idxs, :]