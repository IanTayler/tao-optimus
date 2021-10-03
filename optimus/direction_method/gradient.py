"""Methods that use the gradient in a very direct way."""
import numpy as np

from optimus.types import DirectionMethod
from optimus.types import OptimusFunction


class Gradient(DirectionMethod):
    """Just using the direction is the gradient itself."""

    def __call__(self, parameters: np.ndarray, objective_function: OptimusFunction):
        gradient = objective_function.gradient(parameters)
        return gradient
