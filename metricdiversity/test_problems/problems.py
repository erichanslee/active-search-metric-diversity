import numpy


RE21_THRESHOLDS = [2100, 0.025] # About 3 percent
RE22_THRESHOLDS = [400, 2]  # About 10 percent
RE23_THRESHOLDS = [2.5e5, -1]  # About 6 percent
RE24_THRESHOLDS = [100, -7]  # About 8 percent
RE25_THRESHOLDS = [20, 2] # About 6 percent
RE31_THRESHOLDS = [80, 120, 150]  # About 4 percent
RE33_THRESHOLDS = [2, 3, 4]  # About 4 percent

def find_nearest_value(array, value):
  array = numpy.asarray(array)
  idx = numpy.argmin(numpy.abs(array - value))
  return array[idx]


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


class RE21(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 1, 'max': 3}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': numpy.sqrt(2), 'max': 3}},
    {'name': 'x2', 'type': 'double', 'bounds': {'min': numpy.sqrt(2), 'max': 3}},
    {'name': 'x3', 'type': 'double', 'bounds': {'min': 1, 'max': 3}},
  ]

  def __init__(self, thresholds=RE21_THRESHOLDS, punchout_radius_param=None, punchout_radius_metric=None):
    pass

  def objective_functions(self, x):
    x1, x2, x3, x4 = x
    F = 10
    E = 2e5
    L = 200

    f1 =  L * (2 * x1 + numpy.sqrt(2) * x2 + numpy.sqrt(x3) + x4)
    f2 = 2 * F * L / E * (
      1 / x1 + numpy.sqrt(2) / x2 - numpy.sqrt(2) / x3 + 1 / x4
    )
    return [f1, f2]

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values


class RE22(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 0.2, 'max': 15}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': 0, 'max': 20}},
    {'name': 'x2', 'type': 'double', 'bounds': {'min': 0, 'max': 40}},
  ]

  def __init__(self, thresholds=RE22_THRESHOLDS, punchout_radius_param=None, punchout_radius_metric=None):
    pass

  def objective_functions(self, x):
    FEASIBLE_VALUES_RE22 = [
      0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93,
      1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86,
      2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80,
      3.0, 3.08, 3.10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96,
      4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84,
      5.0, 5.28, 5.40, 5.53, 5.72,
      6.0, 6.16, 6.32, 6.60,
      7.11, 7.20, 7.80, 7.90,
      8.0, 8.40, 8.69,
      9.0, 9.48,
      10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0
    ]

    x1, x2, x3 = x
    x1 = find_nearest_value(FEASIBLE_VALUES_RE22, x1)

    # First original objective function
    f1 = (29.4 * x1) + (0.6 * x2 * x3)

    # Original constraint functions
    g = numpy.zeros(2)
    g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
    g[1] = 4.0 - (x3 / x2)
    # Calculate the constratint violation values
    g[g >= 0] = 0
    g[g < 0] = -g[g < 0]
    f2 = numpy.sum(g)
    return [f1, numpy.log10(f2 + 1e-6)]

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values


class RE23(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 1, 'max': 100}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': 1, 'max': 100}},
    {'name': 'x2', 'type': 'double', 'bounds': {'min': 10, 'max': 200}},
    {'name': 'x3', 'type': 'double', 'bounds': {'min': 10, 'max': 240}},
  ]

  def __init__(self, thresholds=RE23_THRESHOLDS, punchout_radius_param=None, punchout_radius_metric=None):
    pass

  def objective_functions(self, x):
    x1, x2, x3, x4 = x
    x1 = 0.0625 * numpy.round(x1)
    x2 = 0.0625 * numpy.round(x2)

    # First original objective function
    f1 = (
      0.6224 * x1 * x3 * x4 +
      1.7781 * x2 * x3 ** 2 +
      3.1661 * x1 ** 2 * x4 +
      19.84 * x1 ** 2 * x3
      )

    # Original constraint functions
    g = numpy.zeros(3)
    g[0] = x1 - (0.0193 * x3)
    g[1] = x2 - (0.00954 * x3)
    g[2] = (numpy.pi * x3 ** 2 * x4) + (4.0 / 3.0 * numpy.pi * x3 ** 3) - 1296000

    # Calculate the constratint violation values
    g[g >= 0] = 0
    g[g < 0] = -g[g < 0]
    f2 = numpy.sum(g)
    return [f1, numpy.log10(f2 + 1e-6)]

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values


class RE24(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 0.5, 'max': 4}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': 0.5, 'max': 50}},
  ]

  def __init__(self, thresholds=RE24_THRESHOLDS, punchout_radius_param=None, punchout_radius_metric=None):
    pass

  def objective_functions(self, x):
    x1, x2 = x

    f1 = x1 + 120 * x2

    E = 700000
    sigmaBMax = 700
    tauMax = 450
    deltaMax = 1.5
    sigmaK = E * x1 ** 2 / 100
    sigmaB = 4500 / (x1 * x2)
    tau = 1800 / x2
    delta = 562000 / (E * x1 * x2 ** 2)

    g = numpy.zeros(4)
    g[0] = 1 - (sigmaB / sigmaBMax)
    g[1] = 1 - (tau / tauMax)
    g[2] = 1 - (delta / deltaMax)
    g[3] = 1 - (sigmaB / sigmaK)

    g[g >= 0] = 0
    g[g < 0] = -g[g < 0]
    f2 = numpy.sum(g)
    return [f1, numpy.log(f2 + 1e-6)]

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values


class RE25(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 1, 'max': 70}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': 0.6, 'max': 3}},
    {'name': 'x2', 'type': 'double', 'bounds': {'min': 0.09, 'max': 0.5}},
  ]

  def __init__(self, thresholds=RE25_THRESHOLDS, punchout_radius_param=None, punchout_radius_metric=None):
    pass

  def objective_functions(self, x):
    FEASIBLE_VALUES_RE25 = [
      0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162,
      0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041,
      0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135,
      0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283,
      0.307, 0.331, 0.362, 0.394, 0.4375, 0.5,
    ]

    x1, x2, x3 = x
    x1 = numpy.round(x1)
    x3 = find_nearest_value(FEASIBLE_VALUES_RE25, x3)

    f1 = numpy.pi ** 2 * x2 * x3 ** 2 * (x1 + 2) / 4.0

    # constraint functions
    Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
    Fmax = 1000.0
    S = 189000.0
    G = 11.5 * 1e6
    K = (G * x3 ** 4) / (8 * x1 * x2 ** 3)
    lmax = 14.0
    lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
    dmin = 0.2
    Dmax = 3
    Fp = 300.0
    sigmaP = Fp / K
    sigmaPM = 6
    sigmaW = 1.25

    g = numpy.zeros(6)
    g[0] = -((8 * Cf * Fmax * x2) / (numpy.pi * x3 ** 3)) + S
    g[1] = -lf + lmax
    g[2] = -3 + (x2 / x3)
    g[3] = -sigmaP + sigmaPM
    g[4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
    g[5] = sigmaW - ((Fmax - Fp) / K)

    g[g >= 0] = 0
    g[g < 0] = -g[g < 0]
    f2 = numpy.sum(g)
    return [f1, numpy.log10(f2 + 1e-6)]

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values


class RE31(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 0.00001, 'max': 100}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': 0.00001, 'max': 100}},
    {'name': 'x2', 'type': 'double', 'bounds': {'min': 1, 'max': 3}},
  ]

  def __init__(self):
    self.thresholds = RE31_THRESHOLDS
    self.is_minimization = True

  def objective_functions(self, x):
    x1, x2, x3 = x

    metric_0 = x1 * numpy.sqrt(16.0 + x3 ** 2) + x2 * numpy.sqrt(1.0 + x3 ** 2)

    metric_1 = 20.0 * numpy.sqrt(16.0 + x3 ** 2)
    metric_1 = metric_1 / (x3 * x1)

    # Constraint functions
    g = numpy.zeros(3)
    g[0] = 0.1 - metric_0
    g[1] = 100000.0 - metric_1
    g[2] = 100000.0 - ((80.0 * numpy.sqrt(1.0 + x3 ** 2)) / (x3 * x2))

    # Calculate the constratint violation values
    g[g >= 0] = 0
    g[g < 0] = -g[g < 0]
    metric_2 = numpy.sum(g)

    return [metric_0, metric_1, metric_2]

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values


class RE33(TestProblem):
  parameters = [
    {'name': 'x0', 'type': 'double', 'bounds': {'min': 55, 'max': 80}},
    {'name': 'x1', 'type': 'double', 'bounds': {'min': 75, 'max': 110}},
    {'name': 'x2', 'type': 'double', 'bounds': {'min': 1000, 'max': 3000}},
    {'name': 'x3', 'type': 'double', 'bounds': {'min': 11, 'max': 20}},
  ]

  def __init__(self):
    self.thresholds = RE33_THRESHOLDS
    self.is_minimization = True

  def objective_functions(self, x):
    x1, x2, x3, x4 = x

    metric_0 = 4.9 * 1e-5 * (x2 ** 2 - x1 ** 2) * (x4 - 1.0)

    metric_1 = ((9.82 * 1e6) * (x2 ** 2 - x1 ** 2)) / (x3 * x4 * (x2**3 - x1**3))

    # Constraint functions
    g = numpy.zeros(4)
    g[0] = (x2 - x1) - 20
    g[1] = 0.4 - x3 / (3.14 * (x2 ** 2 - x1 ** 2))
    g[2] = 1 - (2.22 * 1e-3 * x3 * (x2 ** 3 - x1 ** 3)) / ((x2 ** 2 - x1 ** 2) ** 2)
    g[3] = 2.66 * 1e-2 * x3 * x4 * (x2 ** 3 - x1 ** 3) / (x2 ** 2 - x1 ** 2) - 900

    # Calculate the constratint violation values
    g[g >= 0] = 0
    g[g < 0] = -g[g < 0]
    metric_2 = numpy.sum(g)

    return [metric_0, metric_1, metric_2]

  def _evaluate(self, suggestion):
    pt = numpy.array([suggestion[p] for p in self.parameter_names])
    values = self.objective_functions(pt)
    return values


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
