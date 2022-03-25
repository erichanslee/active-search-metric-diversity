import torch
from botorch.acquisition.multi_objective.monte_carlo import (
  qNoisyExpectedHypervolumeImprovement,
  qExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.transforms import unnormalize, normalize
from torch.quasirandom import SobolEngine

from ..acquisitions.cas import (
  identify_samples_which_satisfy_constraints, get_and_fit_gp, Domain
)


class QEHVI(object):
  def __init__(self, domain, thresholds, verbose=True, tkwargs=None):
    self.domain = domain
    self.verbose = verbose
    if tkwargs is None:
      tkwargs = {
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          "dtype": torch.double,
      }
    self.thresholds = torch.tensor(thresholds, **tkwargs)
    self.tkwargs = tkwargs
    self.hypercube_bounds = torch.tensor([[0] * self.dim, [1] * self.dim], **self.tkwargs)

  @property
  def dim(self):
    return self.domain.dim

  def update_models(self, X, Y):
    gp_models = [get_and_fit_gp(X, Y[:, i : i + 1]) for i in range(Y.shape[-1])]
    return ModelListGP(*gp_models)

  def get_next(self, model, X, num_restarts=10, raw_samples=512, q=1):
    with torch.no_grad():
      pred = model.posterior(X).mean
    partitioning = FastNondominatedPartitioning(
      ref_point=self.thresholds,
      Y=pred,
    )
    qehvi = qExpectedHypervolumeImprovement(
      model=model,
      ref_point=self.thresholds,
      partitioning=partitioning,
      sampler=SobolQMCNormalSampler(num_samples=128),
    )
#    qehvi = qNoisyExpectedHypervolumeImprovement(
#      model=model,
#      ref_point=self.thresholds,
#      X_baseline=X,
#      prune_baseline=True,
#      sampler=SobolQMCNormalSampler(num_samples=128),
#    )
    x_next, _ = optimize_acqf(
      acq_function=qehvi,
      bounds=self.hypercube_bounds,
      q=1,
      num_restarts=num_restarts,
      raw_samples=raw_samples,
      options={"batch_limit": 5, "maxiter": 200},
      sequential=True,
    )
    if self.verbose:
      print(' >>> x_next', x_next)
    return x_next


class MOBOService(object):
  def __init__(self, parameters, constraints, num_init_points=5, verbose=True):
    self.num_suggestions = 0
    self.num_observations = 0
    self.constraints = constraints
    self.thresholds = [c[1] for c in constraints]
    self.engine = QEHVI(
      domain=Domain(parameters),
      verbose=verbose,
      thresholds=self.thresholds
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
    x_next = self.engine.get_next(model, self.X)
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
      identify_samples_which_satisfy_constraints(self.get_values(), self.constraints)
      .prod(dim=-1)
      .to(torch.bool)
    )
