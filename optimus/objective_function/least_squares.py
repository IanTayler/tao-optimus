"""Ordinary least squares and related problems."""
from typing import Optional

import numpy as np

from optimus.objective_function import ObjectiveFunction


class LeastSquares(ObjectiveFunction):
    """Ordinary least squares of the form min ||Ax - b||^2."""

    is_c2 = True

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        """Initialize a least squares problem.

        `inputs` is `A` (i.e. the matrix of features), and targets is
        `b` (i.e. the vector of observed values)."""
        self.inputs = inputs
        self.targets = targets
        self._cached_hessian: Optional[np.ndarray] = None

    def __call__(self, parameters: np.ndarray) -> float:
        difference_vector = self.inputs.dot(parameters) - self.targets
        return np.dot(difference_vector, difference_vector)

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        return 2 * (self.inputs.dot(parameters) - self.targets).T.dot(self.inputs)

    def partial_second_derivative(
        self, parameters: np.ndarray, first_variable: int, second_variable: int
    ) -> float:
        # This is reasonable because the hessian is constant.
        return self.hessian(parameters)[first_variable, second_variable]

    def hessian(self, parameters: np.ndarray) -> np.ndarray:
        if self._cached_hessian is None:
            self._cached_hessian = self.inputs.T.dot(self.inputs)
        return self._cached_hessian
