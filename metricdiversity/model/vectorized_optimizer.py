from collections import namedtuple

import numpy

from ..model.domain import TensorProductDomain


DEFAULT_VECOPT_MAXITER = 100
DEParameters = namedtuple('DEParameters', [
  'crossover_probability',
  'mutation',
  'strategy',
  'tolerance',
])
DEFAULT_DE_PARAMETERS = DEParameters(
  crossover_probability=0.7,
  mutation=0.8,
  strategy='best1bin',
  tolerance=1e-10,
)

class VectorizedOptimizer(object):
  optimizer_parameters_type = NotImplemented

  def __init__(self, domain, acquisition_function, num_multistarts, optimizer_parameters, maxiter):
    """
    This is the base class for vectorized _maximization_.
    """
    assert isinstance(domain, TensorProductDomain)
    self.domain = domain
    assert self.dim == acquisition_function.dim
    self.af = acquisition_function

    self.num_multistarts = num_multistarts
    self.maxiter = maxiter if maxiter is not None else DEFAULT_VECOPT_MAXITER
    self.optimizer_parameters = optimizer_parameters

    # This is information for monitoring progress during the optimization
    # It is private to make sure that it is only updated during evaluate_and_monitor
    self._best_location = None
    self._best_value = None

  def __repr__(self):
    return (
      f'{self.__class__.__name__}'
      f'(optimizer_parameters={self.optimizer_parameters}, '
      f'num_multistarts={self.num_multistarts}, '
      f'maxiter={self.maxiter})'
    )

  @property
  def best_location(self):
    return self._best_location

  @property
  def best_value(self):
    return self._best_value

  @property
  def dim(self):
    return self.domain.dim

  def restrict_points_to_domain(self, points):
    return self.domain.restrict_points_to_domain(points)

  # NOTE(Mike) - If we want to extract out the AF, we need to build reshaping into the vectorized evaluation
  def evaluate_and_monitor(self, points):
    values = self.af.evaluate_at_point_list(points)

    best_index_now = numpy.nanargmax(values)
    best_value_now = values[best_index_now]
    if self.best_value is None or best_value_now > self.best_value:
      self._best_location = points[best_index_now].flatten()
      self._best_value = best_value_now

    return values

  def optimize(self, selected_starts=None):
    if selected_starts is None:
      starting_points = self.domain.generate_quasi_random_points_in_domain(self.num_multistarts)
    else:
      num_extra_starts = self.num_multistarts - len(selected_starts)
      if num_extra_starts <= 0:
        starting_points = numpy.copy(selected_starts)
      else:
        extra_starts = self.domain.generate_quasi_random_points_in_domain(num_extra_starts)
        starting_points = numpy.concatenate((selected_starts, extra_starts), axis=0)

    # Restrict points makes a copy of starting points and guarantees they are all valid
    restricted_starting_points = self.restrict_points_to_domain(starting_points)
    ending_points = self._optimize(restricted_starting_points)  # restricted_starting_points may change
    values = self.evaluate_and_monitor(ending_points)

    # TODO(Mike) - Should consider either cutting this or making it a dedicated struct
    all_results = {
      'starting_points': starting_points,
      'ending_points': ending_points,
      'function_values': values,
    }
    return self.best_location, all_results

  def _optimize(self, points):
    raise NotImplementedError()


class DEOptimizer(VectorizedOptimizer):
  """
  Implementation of Differential Evolution optimizer, references:
    - Storn and Price, Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over
      Continuous Spaces
  """
  def __init__(self, domain, acquisition_function, num_multistarts, optimizer_parameters=None, maxiter=None):
    if optimizer_parameters is None:
      optimizer_parameters = DEFAULT_DE_PARAMETERS
    super().__init__(domain, acquisition_function, num_multistarts, optimizer_parameters, maxiter)
    self.strategy = self.optimizer_parameters.strategy
    self.tolerance = self.optimizer_parameters.tolerance
    self.mutation = self.optimizer_parameters.mutation
    self.crossover_probability = self.optimizer_parameters.crossover_probability

    # NOTE(Harvey) This command creates a n x n-1 matrix, where n = num_multistarts, of the following form
    # [[1, 2, 3, ... , n],
    #  [0, 2, 3, ... , n],
    #  [0, 1, 3, ...., n],
    #  ....
    #  [0, 1, 2, ...., n-1]]
    self.index_matrix = numpy.triu(
      numpy.tile(numpy.arange(1, self.num_multistarts, dtype=int), (self.num_multistarts, 1))
    ) + numpy.tril(
      numpy.tile(numpy.arange(0, self.num_multistarts - 1, dtype=int), (self.num_multistarts, 1)), -1
    )
    if self.strategy == 'rand1bin':
      self.mutation_strat = self._rand1
    if self.strategy == 'best1bin':
      self.mutation_strat = self._best1

  def _optimize(self, points):
    self.evaluate_and_monitor(points)
    for _ in range(self.maxiter):
      selection_indices = numpy.random.randint(0, self.num_multistarts, (self.num_multistarts, 3))
      mutants = self.mutation_strat(points, selection_indices)
      cross_points = numpy.random.random((self.num_multistarts, self.dim)) < self.crossover_probability
      trials = numpy.where(cross_points, mutants, points)

      # makes sure we always evaluate_and_monitor acceptable points
      trials = self.domain.restrict_points_to_domain(trials)
      values_from_trials = self.evaluate_and_monitor(trials)

      trials_which_improved = values_from_trials >= self.best_value
      points[trials_which_improved] = trials[trials_which_improved]
    return points

  def _best1(self, candidates, selection_indices):
    return (
        self.best_location +
        self.mutation * (candidates[selection_indices[:, 0]] - candidates[selection_indices[:, 1]])
    )

  def _rand1(self, candidates, selection_indices):
    return candidates[selection_indices[:, 0]] + self.mutation * (
      candidates[selection_indices[:, 1]] - candidates[selection_indices[:, 2]]
    )
