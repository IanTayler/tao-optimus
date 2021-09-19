import numpy as np

from gradient_descent.direction_method import DirectionMethod
from gradient_descent.objective_function import ObjectiveFunction


class Newton(DirectionMethod):
    def __call__(
        self, parameters: np.ndarray, objective_function: ObjectiveFunction
    ) -> np.ndarray:
        return np.linalg.inv(objective_function.hessian(parameters)).dot(
            objective_function.gradient(parameters)
        )