"""Methods based on multiplying the gradient by a diagonal definite-positive matrix"""
import numpy as np

from optimus.direction_method import DirectionMethod
from optimus.objective_function import ObjectiveFunction


class DiagonalQuasiHessian(DirectionMethod):
    """Direction is gradient multiplied by an inverse diagonal-only Hessian."""

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
