"""Methods that use the gradient in a very direct way."""
import numpy as np

from optimus.direction_method import DirectionMethod
from optimus.objective_function import ObjectiveFunction


class Gradient(DirectionMethod):
    """Just using the direction is the gradient itself."""

    def __call__(self, parameters: np.ndarray, objective_function: ObjectiveFunction):
        gradient = objective_function.gradient(parameters)
        return gradient
