import numpy as np

from gradient_descent.direction_method import DirectionMethod
from gradient_descent.objective_function import ObjectiveFunction


class Diagonal(DirectionMethod):
    def __call__(
        self, parameters: np.ndarray, objective_function: ObjectiveFunction
    ) -> np.ndarray:
        gradient = objective_function.gradient(parameters)
        size = gradient.size
        self_second_partial_derivatives = np.array(
            [
                objective_function.partial_second_derivative(parameters, i, i)
                for i in range(size)
            ]
        )
        return gradient / self_second_partial_derivatives
