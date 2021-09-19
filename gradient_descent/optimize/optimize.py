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
    minimum_gradient_norm: float = 0.1,
    minimum_step_norm: float = 1e-3,
    verbose: bool = False,
) -> np.ndarray:
    step = 0
    parameters = initial_parameters
    while step < max_steps:
        if verbose:
            print(f"Step {step}.")
        function_value = function(parameters)
        gradient = function.gradient(parameters)
        gradient_norm = np.sqrt((gradient ** 2).sum())
        if verbose:
            print(f"Gradient norm: {gradient_norm}.")
        if gradient_norm < minimum_gradient_norm:
            print("Stopping due to small gradient norm.")
            return parameters
        direction = direction_method(parameters, function)
        lr = lr_method(parameters, function_value, gradient, direction, step, function)
        step_vector = lr * direction
        step_norm = np.sqrt((step_vector ** 2).sum())
        if verbose:
            print(f"Step norm: {step_norm}.")
        if step_norm < minimum_step_norm:
            print("Stopping due to small step norm.")
            return parameters
        parameters -= step_vector
        step += 1
    print("Stopping due to maximum iterations reached.")
    return parameters
