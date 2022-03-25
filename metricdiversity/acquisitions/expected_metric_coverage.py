import numpy
import scipy

from ..model.multi_gaussian_process import MultiOutputGP
from ..model.domain import TensorProductDomain
from ..model.vectorized_optimizer import DEOptimizer
NUM_MC_SAMPLES = 512

class ThresholdBox:
  def __init__(self, lb):
    """
    Input
    ub, lb: array or list (shouldn't matter) of length m
    """
    self.lb = numpy.array(lb)

  def points_in_box(self, Y):
    """
    Input
    Y: n x m set of points
    Output
    Y_in: subset of points in Y that are within threshold values
    """
    idx_over_lb = numpy.all(Y >= self.lb, axis=1)
    Y = Y[idx_over_lb]
    return Y


class ExpectedMetricCoverage:

  def __init__(
    self,
    gaussian_process,
    lb,
    punchout_radius,
    opt_domain=None,
    num_mc_samples=None,
    normalize_y=False,
  ):
    """
    Input
    gaussian_process: assumed to be an object of type MultiOutputGP
    ub: array of length m (num objectives), upper bounds
    lb: array of length m (num objectives), lower bounds
    punchout_radius: self explanatory
    opt_domain: assumed to be unit hypercube by default, of type TensorProductDomain

    """
    self.gaussian_process = gaussian_process
    self.lb = lb
    self.threshold_box = ThresholdBox(lb)
    self.punchout_radius = punchout_radius
    self.num_mc_samples = num_mc_samples or NUM_MC_SAMPLES
    if opt_domain is None:
      self.opt_domain = TensorProductDomain([[0, 1]] * self.gaussian_process.d)
    self.normalize_y = normalize_y

  @property
  def dim(self):
    return self.gaussian_process.dim


  def compute_expected_utility(self, X):
    """
    Compute value via MC by generating sampling from y distribution at a point x, and then
    tossing points that are either outside the threshold box or too close to an existing metric value

    Note: if normalization, we normalize Y based hypercube whose bottom edges are the thresholds and whose top edges
    are determined by the maximum y value along that axis.
    """
    Y_samples = self.gaussian_process.sample(self.num_mc_samples, X)
    idx_in_threshold = numpy.all(Y_samples >= self.threshold_box.lb, axis=1)

    _, Y_obs = self.gaussian_process.get_historical_data()
    Y_obs = self.threshold_box.points_in_box(Y_obs)
    if len(Y_obs) == 0:
      idx_outside_range = numpy.ones_like(idx_in_threshold)
    else:
      if self.normalize_y and len(Y_obs) > 1:
        ub = numpy.max(Y_obs, axis=0)
        Y_samples = (Y_samples - self.lb) / (ub - self.lb)
        Y_obs = (Y_obs - self.lb) / (ub - self.lb)
      dist_matrix = scipy.spatial.distance.cdist(Y_samples, Y_obs)
      idx_outside_range = numpy.all(dist_matrix > self.punchout_radius, axis=1)

    feasible_idx = numpy.logical_and(idx_in_threshold, idx_outside_range)
    feasible_idx = numpy.reshape(feasible_idx, (X.shape[0], self.num_mc_samples))
    expected_utility = numpy.sum(feasible_idx, axis=1)
    return expected_utility / self.num_mc_samples

  def tiebreak_suggestions(self, X):
    """
    In the event that mutiple points have the same utility, we tiebreak by selecting the point
    whose mean y value is furthest away from the other observed metric values (in the same vein that ECI does it)
    """

    _, Y_obs = self.gaussian_process.get_historical_data()
    Y_preds = self.gaussian_process.predict(X)


    idxs = numpy.all(Y_preds >= self.threshold_box.lb, axis=1)

    Y_preds_in_box = Y_preds[idxs]
    X_preds_in_box = X[idxs]

    dist_matrix = scipy.spatial.distance.cdist(Y_preds_in_box, Y_obs)
    closest_distances = numpy.min(dist_matrix, axis=1)
    idx = numpy.argmax(closest_distances)
    return X_preds_in_box[idx, :]

  def get_suggestion(self):
    n_points = 512
    X = self.opt_domain.generate_quasi_random_points_in_domain(n_points)
    utility_values = self.compute_expected_utility(X)

    # If all utility values are low...
    if numpy.sum(utility_values) < 1e-6:
      return self.tiebreak_suggestions(X)
    else:
      idxs = numpy.argmax(utility_values) # need to check for multiple argmaxes
      if isinstance(idxs, list):
        return X[numpy.random.choice(idxs), :]
      return X[idxs, :]

  def evaluate_at_point_list(self, points):
    assert len(points.shape) == 2
    return self.compute_expected_utility(points)


class EMCEngine(object):
  def __init__(self, domain, thresholds, punchout_radius, hyperparameters=None, normalize_y=False, verbose=True):
    self.domain = domain
    self.thresholds = thresholds
    self.punchout_radius = punchout_radius
    self.train_gp = True if hyperparameters is None else False
    self.hyperparameters = hyperparameters
    if self.hyperparameters is not None:
      assert self.hyperparameters.shape[0] == len(self.thresholds)
      assert self.hyperparameters.shape[1] == domain.dim + 1
    self.normalize_y = normalize_y
    self.verbose = verbose

  def update_models(self, X, Y):
    mgp = MultiOutputGP(X, Y, train=self.train_gp)
    if not self.train_gp:
      assert self.hyperparameters is not None
      mgp.set_hypers(self.hyperparameters)
    return mgp

  def get_next(self, model):
    emc = ExpectedMetricCoverage(model, self.thresholds, self.punchout_radius, normalize_y=self.normalize_y)
    de = DEOptimizer(self.domain, emc, 10 * self.domain.dim)
    next_point, all_results = de.optimize()
    best_emc = numpy.max(all_results["function_values"])
    if self.verbose:
      print(" >>> x_next", next_point, "emc", best_emc)
    return next_point



class ExpectedMetricCoverageService(object):
  def __init__(
    self,
    parameters,
    constraints,
    punchout_radius,
    num_init_points=5,
    hyperparameters=None,
    normalize_y=False,
    verbose=True,
  ):
    self.num_suggestions = 0
    self.num_observations = 0
    self.parameters = parameters
    self.parameters_names = []
    self.domain = self.parse_parameters_to_domain()
    # assuming maximizing everything
    self.thresholds = [c[1] for c in constraints]
    self.punchout_radius = punchout_radius
    self.engine = EMCEngine(
      self.domain,
      self.thresholds,
      self.punchout_radius,
      hyperparameters,
      normalize_y,
      verbose,
    )
    self.list_suggestions = []
    self.served_suggestions = {}
    self.X = None
    self.Y = None
    self.add_random_suggestions(num_init_points)

  @property
  def dim(self):
    return self.domain.dim

  def parse_parameters_to_domain(self):
    domain_comp = []
    for i, param in enumerate(self.parameters):
      assert param["type"] in ("double", "int")
      domain_comp.append([param["bounds"]["min"], param["bounds"]["max"]])
      self.parameters_names.append(param["name"])
    return TensorProductDomain(domain_comp)

  def add_random_suggestions(self, num):
    X_init = self.domain.generate_quasi_random_points_in_domain(num)
    self.form_suggestions(X_init)

  def add_engine_suggestion(self):
    assert self.X.shape[0] == self.Y.shape[0]
    model = self.engine.update_models(self.X, self.Y)
    x_next = self.engine.get_next(model)
    self.form_suggestions(numpy.atleast_2d(x_next))

  def vector_to_param_dict(self, point):
    param_value_dict = {}
    for p, name in zip(point, self.parameters_names):
      param_value_dict[name] = p.item()
    return param_value_dict

  def param_dict_to_vector(self, param_dict):
    vector_value = numpy.zeros(len(self.parameters_names))
    for i, p in enumerate(self.parameters_names):
      vector_value[i] = param_dict[p]
    return vector_value

  def form_suggestions(self, X):
    for x in X:
      self.num_suggestions += 1
      suggestion = {'id': self.num_suggestions}
      suggestion.update(self.vector_to_param_dict(x))
      self.list_suggestions.append(suggestion)

  def create_suggestion(self):
    if len(self.list_suggestions) == 0:
        self.add_engine_suggestion()
    suggestion = self.list_suggestions.pop()
    self.served_suggestions[suggestion['id']] = (suggestion, False)
    return suggestion

  def get_values_from_observation(self, observation):
    assert 'values' in observation
    values = observation['values']
    if not isinstance(values, list):
      values = [values]
    return numpy.array(values)

  def create_observation(self, observation):
    assert 'suggestion' in observation
    suggestion, used_suggestion = self.served_suggestions[observation['suggestion']]
    assert used_suggestion is False
    suggestion_vector = self.param_dict_to_vector(suggestion)
    x = suggestion_vector
    y = self.get_values_from_observation(observation)
    if self.num_observations == 0:
      self.X = x
      self.Y = y
    else:
      self.X = numpy.concatenate((numpy.atleast_2d(self.X), x[None, :]))
      self.Y = numpy.concatenate((numpy.atleast_2d(self.Y), y[None, :]))
    self.num_observations += 1
    self.served_suggestions[observation['suggestion']] = (suggestion, True)

  def dump_manual_data(self, points, values):
    # This will destroy your current data so make sure is None
    assert self.X is None
    assert self.Y is None
    assert len(points) == len(points)
    self.X = points
    self.Y = values
    self.num_observations = len(self.Y)

  def get_points(self):
    return self.X

  def get_values(self):
    return self.Y

  def get_feasible_indices(self):
    return numpy.all(self.Y >= self.thresholds, axis=1)
