"""Methods based on Newton's method."""
import numpy as np

from optimus.types import DirectionMethod
from optimus.types import Function


class Newton(DirectionMethod):
    """Classic Netwon's method. Direction is the inverse hessian times gradient."""

    def __call__(
        self, parameters: np.ndarray, objective_function: Function
    ) -> np.ndarray:
        return np.linalg.inv(objective_function.hessian(parameters)).dot(
            objective_function.gradient(parameters)
        )
