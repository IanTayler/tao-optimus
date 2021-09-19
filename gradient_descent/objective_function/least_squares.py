import numpy as np

from gradient_descent.objective_function import ObjectiveFunction


class LeastSquares(ObjectiveFunction):
    """Ordinary least squares of the form min ||Ax - b||."""

    is_c2 = True

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __call__(self, parameters: np.ndarray) -> float:
        difference_vector = self.inputs.dot(parameters) - self.targets
        return np.dot(difference_vector, difference_vector)

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        return 2 * (self.inputs.dot(parameters) - self.targets).T.dot(self.inputs)
