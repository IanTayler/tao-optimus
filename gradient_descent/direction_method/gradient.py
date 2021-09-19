import numpy as np
from objective_function.objective_function import ObjectiveFunction

from gradient_descent.direction_method import DirectionMethod


class Gradient(DirectionMethod):
    def __call__(self, parameters: np.ndarray, objective_function: ObjectiveFunction):
        gradient = objective_function.gradient(parameters)
        return -gradient
