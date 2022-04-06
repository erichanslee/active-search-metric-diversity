import argparse
import copy
import pickle

import numpy

from metricdiversity.acquisitions.cas import ConstraintActiveSearchService
from metricdiversity.acquisitions.expected_metric_coverage import ExpectedMetricCoverageService
from metricdiversity.acquisitions.mobo import MOBOService
from metricdiversity.test_problems.problems import *


def run_mobo(test_problem, budget=None, reps=1):
  problem = test_problem()
  if budget is None:
    assert problem.budget > 0
    budget = problem.budget
  all_results = []
  problem_name = problem.__class__.__name__
  print(f"Running problem {problem_name} for MOBO with budget={budget}.")
  for i in range(reps):
    print(f"replication {i + 1}/{reps}")
    mobo = MOBOService(
      parameters=problem.parameters,
      constraints=problem.constraints,
      num_init_points=len(problem.metric_names) * len(problem.parameter_names),
      verbose=False,
    )
    for _ in range(budget):
      suggestion = mobo.create_suggestion()
      values = problem.evaluate(suggestion)
      observation = {'suggestion': suggestion['id'], 'values': values}
      mobo.create_observation(observation)

    mobo_points = numpy.array(mobo.get_points())
    mobo_values = numpy.array(mobo.get_values())
    mobo_feasible_inds = numpy.array(mobo.get_feasible_indices())

    all_results.append({
      'points': mobo_points,
      'values': mobo_values,
      'satisfied': mobo_feasible_inds,
    })

  file_name = f"./results/{problem_name}_MOBO.pkl"

  save_results(all_results, file_name)
  return

def run_cas(test_problem, budget=None, reps=1):
  problem = test_problem()
  if budget is None:
    assert problem.budget > 0
    budget = problem.budget
  all_results = []
  problem_name = problem.__class__.__name__
  print(f"Running problem {problem_name} for CAS with budget={budget}.")
  for i in range(reps):
    print(f"replication {i + 1}/{reps}")
    cas = ConstraintActiveSearchService(
      parameters=problem.parameters,
      constraints=problem.constraints,
      punchout_radius=problem.punchout_radius_param,
      num_init_points=len(problem.metric_names) * len(problem.parameter_names),
      verbose=False,
    )
    for _ in range(budget):
      suggestion = cas.create_suggestion()
      values = problem.evaluate(suggestion)
      observation = {'suggestion': suggestion['id'], 'values': values}
      cas.create_observation(observation)

    cas_points = numpy.array(cas.get_points())
    cas_values = numpy.array(cas.get_values())
    cas_feasible_inds = numpy.array(cas.get_feasible_indices())

    all_results.append({
      'points': cas_points,
      'values': cas_values,
      'satisfied': cas_feasible_inds,
    })

  file_name = f"./results/{problem_name}_CAS.pkl"

  save_results(all_results, file_name)
  return

def run_emc(test_problem, budget=None, reps=1):
  problem = test_problem()
  if budget is None:
    assert problem.budget > 0
    budget = problem.budget
  all_results = []
  problem_name = problem.__class__.__name__
  print(f"Running problem {problem_name} for EMC with budget={budget}.")
  for i in range(reps):
    print(f"replication {i + 1}/{reps}")
    emc = ExpectedMetricCoverageService(
      parameters=problem.parameters,
      constraints=problem.constraints,
      punchout_radius=problem.punchout_radius_metric_normalized,
      num_init_points=len(problem.metric_names) * len(problem.parameter_names),
      normalize_y=True,
      verbose=False,
    )
    for _ in range(budget):
      suggestion = emc.create_suggestion()
      values = problem.evaluate(suggestion)
      observation = {'suggestion': suggestion['id'], 'values': values}
      emc.create_observation(observation)

    emc_points = numpy.array(emc.get_points())
    emc_values = numpy.array(emc.get_values())
    emc_feasible_inds = numpy.array(emc.get_feasible_indices())

    all_results.append({
      'points': emc_points,
      'values': emc_values,
      'satisfied': emc_feasible_inds,
    })

  file_name = f"./results/{problem_name}_EMC.pkl"

  save_results(all_results, file_name)
  return

def save_results(results, filename):
  with open(filename, 'wb') as file:
    pickle.dump(results, file)
  print(f"Saved result to {filename}")


def run_problems(args):
  budget = args.budget
  reps = args.reps
  problems = [TwoHumps, EMI]
  for problem in problems:
    run_mobo(problem, budget, reps)
    run_cas(problem, budget, reps)
    run_emc(problem, budget, reps)

  return

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--budget', type=int, required=False, default=None)
  parser.add_argument('--reps', type=int, required=False, default=1)
  return parser.parse_args()

def main():
  args = parse_args()
  run_problems(args)

if __name__ == '__main__':
  main()
