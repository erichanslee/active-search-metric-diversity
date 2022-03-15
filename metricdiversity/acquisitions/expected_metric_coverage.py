import numpy
import scipy

from ..model.multi_gaussian_process import MultiOutputGP

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

  def __init__(self, gaussian_process, ub, lb, punchout_radius):
    self.gaussian_process = gaussian_process
    self.threshold_box = ThresholdBox(ub, lb)
    self.punchout_radius = punchout_radius

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