import numpy

class TestProblem(object):
  parameters = []
  parameter_names = []
  metric_names = []

  def __init__(self, thresholds):
    assert len(thresholds) == len(self.metric_names)
    self.thresholds = thresholds
    self.constraints = [("gt", t) for t in self.thresholds]

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

  def __init__(self, thresholds=None):
    if thresholds is None:
      thresholds = [0.85, 0.85]
    super().__init__(thresholds)

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

  def __init__(self, thresholds=None):
    if thresholds is None:
      thresholds = [40, 0.6, 0, 0]
    super().__init__(thresholds)

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
