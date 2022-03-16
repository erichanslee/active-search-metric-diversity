import numpy
import qmcpy


def generate_sobol_points(num_points, domain_bounds):
  distribution = qmcpy.Sobol(dimension=len(domain_bounds))
  pts01 = distribution.gen_samples(n=2 ** numpy.ceil(numpy.log2(num_points)))[:num_points]
  pts_scale = numpy.diff(domain_bounds, axis=1).ravel()
  pts_min = domain_bounds[:, 0]
  return pts_min + pts_scale * pts01


class TensorProductDomain(object):
  def __init__(self, domain_bounds):
    """Construct a TensorProductDomain with the specified bounds defined using
    a list of list.
    """
    bounds_shape = numpy.asarray(domain_bounds).shape
    assert len(bounds_shape) == 2
    assert bounds_shape[1] == 2
    self.domain_bounds = numpy.copy(domain_bounds)
    assert numpy.all(numpy.diff(domain_bounds, axis=1) >= 0)

  def __repr__(self):
    return f'TensorProductDomain({self.domain_bounds})'

  @property
  def dim(self):
    return len(self.domain_bounds)

  def check_point_acceptable(self, point):
    assert len(point) == self.dim
    return numpy.all((point >= self.domain_bounds[:, 0]) & (point <= self.domain_bounds[:, 1]))

  def get_bounding_box(self):
    return numpy.copy(self.domain_bounds)

  def get_lower_upper_bounds(self):
    return self.domain_bounds.T

  def restrict_points_to_domain(self, points):
    lb, ub = self.get_lower_upper_bounds()
    restricted_points = numpy.clip(points, lb, ub)
    return restricted_points

  def generate_quasi_random_points_in_domain(self, num_points, log_sample=False):
    r"""Generate quasi-random points in the domain.

    :param num_points: max number of points to generate
    :type num_points: int >= 0
    :param log_sample: sample logarithmically spaced points
    :type log_sample: bool
    :return: uniform random sampling of points from the domain
    :rtype: array of float64 with shape (num_points, dim)

    """
    domain_bounds = self.domain_bounds
    if log_sample:
      domain_bounds = numpy.log(self.domain_bounds)

    points = generate_sobol_points(num_points, domain_bounds)
    return numpy.exp(points) if log_sample else points
