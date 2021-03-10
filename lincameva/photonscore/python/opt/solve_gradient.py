import numpy as np
from photonscore.python._native_s_ import GradientSolverOptions
from photonscore.python._native_s_ import opt_solve_gradient as _solve_gradient


class SolveGradientResult:
  def __init__(self, x, y, report):
    self.x = x
    self.y = y
    self.report = report


def solve_gradient(f, x0, opts = GradientSolverOptions()):
  x = np.array(x0, dtype = "double")
  report = _solve_gradient(opts, f, x)
  return SolveGradientResult(x, f(x)[0], report)

