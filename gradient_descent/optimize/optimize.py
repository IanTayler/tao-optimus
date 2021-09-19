import numpy as np

from gradient_descent.direction_method import DirectionMethod
from gradient_descent.lr_method import LRMethod
from gradient_descent.objective_function import ObjectiveFunction


def optimize(
    initial_parameters: np.ndarray,
    function: ObjectiveFunction,
    direction_method: DirectionMethod,
    lr_method: LRMethod,
    max_steps: int,
) -> np.ndarray:
    step = 0
    parameters = initial_parameters
    while max_steps < max_steps:
        function_value = function(parameters)
        gradient = function.gradient(parameters)
        direction = direction_method(parameters, function)
        lr = lr_method(parameters, function_value, gradient, direction, step, function)
        parameters -= lr * direction
        step += 1
    return parameters
