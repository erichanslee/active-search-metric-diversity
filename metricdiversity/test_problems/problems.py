import numpy

class TestProblem(object):
  parameters = []
  parameter_names = []
  metric_names = []

  def __init__(self, thresholds, punchout_radius_param, punchout_radius_metric):
    assert len(thresholds) == len(self.metric_names)
    self.thresholds = thresholds
    self.constraints = [("gt", t) for t in self.thresholds]
    assert 0 < punchout_radius_param <= 1
    assert 0 < punchout_radius_metric
    self.punchout_radius_param = punchout_radius_param
    self.punchout_radius_metric = punchout_radius_metric

  def evaluate(self, suggestion):
    assert isinstance(suggestion, dict)
    assert all(pn in suggestion.keys() for pn in self.parameter_names)
    values = self._evaluate(suggestion)
    assert len(values) == len(self.metric_names)
    return values

  def _evaluate(self, suggestion):
    raise NotImplementedError


class TwoHumps(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 0, 'max': 1}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': 0, 'max': 1}}
  ]
  parameter_names = ["x0", "x1"]
  metric_names = ["f1", "f2"]
  fixed_lengthscales = numpy.array([
    [0.01, 0.3, 0.3],
    [0.01, 0.3, 0.3],
  ])

  def __init__(self, thresholds=None, punchout_radius_param=None, punchout_radius_metric=None):
    if thresholds is None:
      thresholds = [0.85, 0.85]
    if punchout_radius_param is None:
      punchout_radius_param = 0.1
    if punchout_radius_metric is None:
      punchout_radius_metric = 0.03
    super().__init__(thresholds, punchout_radius_param, punchout_radius_metric)

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values

  def objective_functions(self, x):
    assert len(x) == len(self.parameter_names)
    scale_x = 1
    scale_y = 1
    f1 = numpy.exp(-0.5* ((x[0]-0.2)**2 + scale_x * (x[1]-0.5)**2))
    f2 = numpy.exp(-0.5* ((x[0]-0.8)**2 + scale_y *(x[1]-0.5)**2))

    return [f1, f2]


class EMI(TestProblem):
  parameters=[
    dict(name="silvert", bounds=dict(min=1, max=50), type="double"),
    dict(name="bott", bounds=dict(min=40, max=100), type="double"),
    dict(name="pitch", bounds=dict(min=10, max=400), type="double"),
    dict(name="cbotr", bounds=dict(min=20, max=200), type="double"),
    dict(name="ctopr", bounds=dict(min=10, max=200), type="double"),
  ]
  metric_names = [
    "EMI_SE",
    "Transmission",
    "0.5 pitch - cbotr",
    "cbotr - ctopr",
  ]
  parameter_names = [
    "silvert",
    "bott",
    "pitch",
    "cbotr",
    "ctopr",
  ]
  fixed_lengthscales = numpy.array([
    [20, 5, 100, 400, 200, 200],
    [0.4, 5, 10, 100, 200, 200],
    [20, 50, 100, 100, 50, 200],
    [20, 50, 100, 400, 50,  50],
  ])

  def __init__(self, thresholds=None, punchout_radius_param=None, punchout_radius_metric=None):
    if thresholds is None:
      thresholds = [40, 0.6, 0, 0]
    if punchout_radius_param is None:
      punchout_radius_param = 0.1 * numpy.sqrt(len(self.parameters))
    if punchout_radius_metric is None:
      punchout_radius_metric = 10
    super().__init__(thresholds, punchout_radius_param, punchout_radius_metric)

  def _evaluate(self, suggestion):
    s = suggestion['silvert']
    b = suggestion['bott']
    p = suggestion['pitch']
    cbotr = suggestion['cbotr']
    ctopr = suggestion['ctopr']

    emi_se = 7.5 * numpy.log(s) + 22
    transmission = .91 / (.65 - .22) * (
        (.4 + .5 * numpy.exp(-.001 * (s - 5) ** 2)) *
        (.6 + .25 * numpy.exp(-.0005 * (b - 40) ** 2)) *
        (.6 + .25 * numpy.exp(-.000005 * (p - 100) ** 2)) - .22
    )
    values = [emi_se, transmission, 0.5 * p - cbotr, cbotr - ctopr]
    return values
