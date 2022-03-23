import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import sample_hypersphere
from botorch.utils.transforms import t_batch_mode_transform, normalize, unnormalize
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from itertools import combinations


def smooth_mask(x, a, eps=2e-3):
    """Return 0ish for x < a and 1ish for x > a"""
    return torch.nn.Sigmoid()((x - a) / eps)

def smooth_box_mask(x, a, b, eps=2e-3):
    """Return 1ish for a < x < b and 0ish otherwise"""
    return smooth_mask(x, a, eps) - smooth_mask(x, b, eps)

def identify_samples_which_satisfy_constraints(X, constraints):
    """
    Take in values (a1, ..., ak, o) and returns (a1, ..., ak, o)

    True/False values, where o is the number of outputs.
    """
    successful = torch.ones(X.shape).to(X)
    for model_index in range(X.shape[-1]):
        these_X = X[..., model_index]
        direction, value = constraints[model_index]
        successful[..., model_index] = (
            these_X < value if direction == "lt" else these_X > value
        )
    return successful

def get_and_fit_gp(X, Y):
  """
    Simple method for creating a GP with one output dimension.

    X is assumed to be in [0, 1]^d.
  """
  assert Y.ndim == 2 and Y.shape[-1] == 1
  likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-3))  # Noise-free
  octf = Standardize(m=Y.shape[-1])
  gp = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=octf)
  mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
  fit_gpytorch_model(mll)
  return gp


class ExpectedCoverageImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        constraints,
        punchout_radius,
        bounds,
        num_samples=512,
        base_points=None,
        **kwargs,
    ):
        """Expected Coverage Improvement (q=1 required, analytic)

        Right now, we assume that all the models in the ModelListGP have
        the same training inputs.

        Args:
            model: A ModelListGP object containing models matching the corresponding constraints.
                All models are assumed to have the same training data.
            constraints: List containing 2-tuples with (direction, value), e.g.,
                [('gt', 3), ('lt', 4)]. It is necessary that
                len(constraints) == model.num_outputs.
            punchout_radius: Positive value defining the desired minimum distance between points
            bounds: torch.tensor whose first row is the lower bounds and second row is the upper bounds
            num_samples: Number of samples for MC integration
        """
        super().__init__(model=model, objective=IdentityMCObjective(), **kwargs)
        assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        assert punchout_radius > 0
        self.constraints = constraints
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.base_points = base_points if base_points is not None else self._identify_base_points_from_train_inputs()
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples,
            radius=punchout_radius,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)
        assert (
            all(ub > lb for lb, ub in self.bounds.T) and len(self.bounds.T) == self.dim
        )

    @property
    def num_outputs(self):
        return self.model.num_outputs

    @property
    def dim(self):
        return self.train_inputs.shape[-1]

    @property
    def train_inputs(self):
        return self.model.models[0].train_inputs[0]

    def _identify_base_points_from_train_inputs(self):
        return self.train_inputs

    def _generate_ball_of_points(
        self, num_samples, radius, device=None, dtype=torch.double
    ):
        """Create a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, **tkwargs)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z

    def _identify_samples_which_satisfy_constraints(self, X):
        return identify_samples_which_satisfy_constraints(X, self.constraints)

    def _get_base_point_mask(self, X):
        distance_matrix = self.model.models[0].covar_module.base_kernel.covar_dist(
            X, self.base_points
        )
        return smooth_mask(distance_matrix, self.punchout_radius)

    def _estimate_probabilities_of_satisfaction_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        posterior = self.model.posterior(X=points)
        mus, sigma2s = posterior.mean, posterior.variance
        dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
        norm_cdf = dist.cdf(self._thresholds)
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (
                norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
            )
        return probs

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate Expected Improvement on the candidate set X."""
        ball_around_X = self.ball_of_points + X
        domain_mask = smooth_box_mask(
            ball_around_X, self.bounds[0, :], self.bounds[1, :]
        ).prod(dim=-1)
        num_points_in_integral = domain_mask.sum(dim=-1)
        base_point_mask = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        prob = self._estimate_probabilities_of_satisfaction_at_points(ball_around_X)
        masked_prob = prob * domain_mask * base_point_mask
        y = masked_prob.sum(dim=-1) / num_points_in_integral
        return y


class Domain(object):
  def __init__(self, parameters):
    self.parameters_name = None
    self.parameters = self._validate_and_copy_parameters(parameters)
    self.dim = len(self.parameters)
    self.bounds, self.param_mapping = self.form_domain()

  def _validate_and_copy_parameters(self, parameters):
    parameters_name = []
    for param in parameters:
      assert 'name' in param
      parameters_name.append(param['name'])
      assert 'type' in param
      assert param['type'] in ('double', 'int')
      assert 'bounds' in param
      assert 'min' in param['bounds'] and 'max' in param['bounds']
    sorted_parameters = []
    self.parameters_name = []
    for name, p in sorted(zip(parameters_name, parameters)):
        self.parameters_name.append(name)
        sorted_parameters.append(p)
    return sorted_parameters

  def form_domain(self):
    param_mapping = []
    lower_bounds, upper_bounds = [], []
    for i, param in enumerate(self.parameters):
        param_mapping.append({'type': param['type'], 'input_ind': i})
        lower_bounds.append(param['bounds']['min'])
        upper_bounds.append(param['bounds']['max'])
    lower_bounds = torch.tensor(lower_bounds)
    upper_bounds = torch.tensor(upper_bounds)
    bounds = torch.stack([lower_bounds, upper_bounds])
    assert bounds.shape[0] == 2
    assert bounds.shape[1] == self.dim
    return bounds, param_mapping

  def map_to_hypercube(self, points):
    return normalize(points, self.bounds)

  def map_from_hypercube(self, hypercube_points):
    points = unnormalize(hypercube_points, self.bounds)
    return self.round_points_to_integer_values(points)

  def round_points_to_integer_values(self, points):
    snapped_points = torch.clone(points)
    for param_map in self.param_mapping:
      if param_map['type'] == 'int':
        index = param_map['input_ind']
        snapped_points[:, index] = torch.round(snapped_points[:, index])
    return snapped_points

  def generate_quasi_random_points_in_domain(self, num_points):
    hypercube_points = SobolEngine(self.dim, scramble=True).draw(num_points)
    return self.map_from_hypercube(hypercube_points)

  def vector_to_param_dict(self, point):
    param_value_dict = {}
    for p, name in zip(point, self.parameters_name):
      param_value_dict[name] = p.item()
    return param_value_dict

  def param_dict_to_vector(self, param_dict):
    list_values = []
    for p in self.parameters_name:
      list_values.append(param_dict[p])
    return torch.tensor(list_values)

class ConstraintActiveSearch(object):
  def __init__(self, domain, constraints, punchout_radius, black_box_function, verbose=True, tkwargs=None):
    self.domain = domain
    self.constraints = constraints
    self.punchout_radius = punchout_radius  # Should be the distances in the hypercube!
    self.black_box_function = black_box_function
    self.verbose = verbose
    if tkwargs is None:
      tkwargs = {
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          "dtype": torch.double,
      }
    self.tkwargs = tkwargs
    self.hypercube_bounds = torch.tensor([[0] * self.dim, [1] * self.dim], **self.tkwargs)

  @property
  def dim(self):
    return self.domain.dim

  def evaluate_objective(self, X):
    assert len(X.shape) == 2
    assert X.shape[1] == self.dim
    x_unnormalized = self.domain.map_from_hypercube(X)
    output = []
    for x in x_unnormalized:
      black_box_output = self.black_box_function(x)
      if not isinstance(black_box_output, list):
        black_box_output = [black_box_output]
      output.append(black_box_output)
    torch_output = torch.FloatTensor(output)
    assert torch_output.shape[0] == x_unnormalized.shape[0]
    assert torch_output.shape[1] == len(self.constraints)
    return torch_output

  def update_models(self, X, Y):
    gp_models = [get_and_fit_gp(X, Y[:, i : i + 1]) for i in range(Y.shape[-1])]
    return ModelListGP(*gp_models)

  def get_next(self, model, num_restarts=10, raw_samples=512, q=1):
    eci = ExpectedCoverageImprovement(
      model=model,
      constraints=self.constraints,
      punchout_radius=self.punchout_radius,
      bounds=self.hypercube_bounds,
      num_samples=512,
    )
    x_next, best_eci = optimize_acqf(
      acq_function=eci,
      bounds=self.hypercube_bounds,
      q=q,
      num_restarts=num_restarts,
      raw_samples=raw_samples,
    )
    if self.verbose:
      print(' >>> x_next', x_next, 'eci', best_eci)
    return x_next

  def run_optimization(self, num_total_points, num_init_points=5):
    lb, ub = self.hypercube_bounds
    X = lb + (ub - lb) * SobolEngine(self.dim, scramble=True).draw(num_init_points).to(**self.tkwargs)
    Y = self.evaluate_objective(X)
    while len(X) < num_total_points:
      model = self.update_models(X, Y)
      x_next = self.get_next(model)
      y_next = self.evaluate_objective(x_next)
      X = torch.cat((X, x_next))
      Y = torch.cat((Y, y_next))
    X_unnormalized = self.domain.map_from_hypercube(X)
    return X_unnormalized, Y

class ConstraintActiveSearchService(object):
  def __init__(self, parameters, constraints, punchout_radius, num_init_points=5, verbose=True):
    self.num_suggestions = 0
    self.num_observations = 0
    self.engine = ConstraintActiveSearch(
      domain=Domain(parameters),
      constraints=constraints,
      punchout_radius=punchout_radius,
      black_box_function=None,
      verbose=verbose,
    )
    self.list_suggestions = []
    self.served_suggestions = {}
    self.X = None
    self.Y = None
    self.add_random_suggestions(num_init_points)

  @property
  def dim(self):
    return self.engine.dim

  @property
  def domain(self):
    return self.engine.domain

  def add_random_suggestions(self, num):
    lb, ub = self.engine.hypercube_bounds
    X = lb + (ub - lb) * SobolEngine(self.dim, scramble=True).draw(num).to(**self.engine.tkwargs)
    X_unnormalized = self.domain.map_from_hypercube(X)
    self.form_suggestions(X_unnormalized)

  def add_engine_suggestion(self):
    assert self.X.shape[0] == self.Y.shape[0]
    model = self.engine.update_models(self.X, self.Y)
    x_next = self.engine.get_next(model)
    x_unnormalized = self.domain.map_from_hypercube(x_next)
    self.form_suggestions(x_unnormalized)

  def form_suggestions(self, X):
    for x in X:
      self.num_suggestions += 1
      suggestion = {'id': self.num_suggestions}
      suggestion.update(self.domain.vector_to_param_dict(x))
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
    torch_output = torch.FloatTensor(values)
    return torch_output

  def create_observation(self, observation):
    assert 'suggestion' in observation
    suggestion, used_suggestion = self.served_suggestions[observation['suggestion']]
    assert used_suggestion is False
    suggestion_vector = self.domain.param_dict_to_vector(suggestion)
    x = self.domain.map_to_hypercube(suggestion_vector)
    y = self.get_values_from_observation(observation)
    x = torch.atleast_2d(x).to(**self.engine.tkwargs)
    y = torch.atleast_2d(y).to(**self.engine.tkwargs)
    if self.num_observations == 0:
      self.X = x
      self.Y = y
    else:
      self.X = torch.cat((self.X, x))
      self.Y = torch.cat((self.Y, y))
    self.num_observations += 1
    self.served_suggestions[observation['suggestion']] = (suggestion, True)

  def dump_manual_data(self, points, values):
    # This will destroy your current data so make sure is None
    assert self.X is None
    assert self.Y is None
    assert len(points) == len(points)
    self.X = self.domain.map_to_hypercube(torch.tensor(points))
    self.Y = torch.tensor(values)
    self.num_observations = len(self.Y)

  def get_points(self):
    return self.domain.map_from_hypercube(self.X)

  def get_values(self):
    return self.Y

  def get_feasible_indices(self):
    return (
      identify_samples_which_satisfy_constraints(self.get_values(), self.engine.constraints)
      .prod(dim=-1)
      .to(torch.bool)
    )

  def get_best_from_eci(self, k):
    feasible_inds = self.get_feasible_indices()
    valid_points = self.X[feasible_inds, :]
    num_valid_observations = len(valid_points)
    if k >= num_valid_observations:  # no sub-selection needed, return all valid points
      return valid_points

    num_observations = self.num_observations
    assert num_observations == len(self.X)
    assert k < num_observations
    list_of_best_indices = []

    # get best offline value
    best_index = torch.argmax(self.Y)
    list_of_best_indices.append(best_index)

    model = self.engine.update_models(self.X, self.Y)
    for i in range(k - 1):
      eci = ExpectedCoverageImprovement(
        model=model,
        constraints=self.engine.constraints,
        punchout_radius=self.engine.punchout_radius,
        bounds=self.engine.hypercube_bounds,
        num_samples=512,
        base_points=torch.atleast_2d(self.X[list_of_best_indices, :])
      )
      eci_list = [eci(torch.atleast_2d(self.X[j, :])) for j in range(num_observations)]
      best_index = torch.argmax(torch.tensor(eci_list))
      list_of_best_indices.append(best_index)

    points = self.get_points()
    return torch.atleast_2d(points[list_of_best_indices, :])

  def get_best_from_dpp(self, k):
    feasible_inds = self.get_feasible_indices()
    valid_points = self.X[feasible_inds, :]
    valid_values = self.Y[feasible_inds]
    num_valid_observations = len(valid_points)
    if k >= num_valid_observations:  # no sub-selection needed, return all valid points
      return self.domain.map_from_hypercube(valid_points), valid_values

    # get best offline value
    best_index = torch.argmax(self.Y)
    best_point = self.X[best_index, None]
    best_value = self.Y[best_index, None]

    model = self.engine.update_models(
      best_point, self.Y[best_index, None]
    )

    best_determinant = None
    best_indices = None
    for indices_to_include in combinations(torch.arange(len(valid_points)), k - 1):
      points_to_consider = valid_points[indices_to_include, :]
      L = model.posterior(points_to_consider).mvn.lazy_covariance_matrix.root_decomposition().root.evaluate()
      det = torch.prod(torch.diag(L))
      if best_determinant is None or det > best_determinant:
        best_determinant = det
        best_indices = indices_to_include

    det_points = valid_points[best_indices, :]
    det_values = valid_values[best_indices, :]
    det_values = torch.cat((best_value, det_values))

    return self.domain.map_from_hypercube(torch.cat((best_point, det_points))), det_values
