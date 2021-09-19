import numpy as np

from gradient_descent.direction_method import DirectionMethod
from gradient_descent.objective_function import ObjectiveFunction


class Gradient(DirectionMethod):
    def __call__(self, parameters: np.ndarray, objective_function: ObjectiveFunction):
        gradient = objective_function.gradient(parameters)
        return gradient
