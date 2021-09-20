"""Base class implementations for objective functions."""
from abc import ABC, abstractmethod

import numpy as np


class ObjectiveFunction(ABC):
    """Base class for functions to be optimized using gradient descent methods.

    Example:

    ```python
    # This is an example for how to build your own ObjectiveFunction.
    class Identity(ObjectiveFunction):

        is_c2 = True

        def __call__(self, parameters):
            return parameters

        def gradient(self, parameters):
            return np.ones_like(parameters)

        def second_partial_derivative(self, parameters, first_variable, second_variable):
            return 0.0

        def hessian(self, parameters):
            size = parameters.size
            return np.zeros([size, size])
    ```
    """

    is_c2: bool = False
    """Whether the function can be assumed to be twice differentiable with a continuous
    second differential. If this is set, it makes computing the hessian from the partial
    derivatives faster."""

    @abstractmethod
    def __call__(self, parameters: np.ndarray) -> float:
        """Implementation of the function itself."""

    @abstractmethod
    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Implementation of the gradient for the function."""

    def partial_second_derivative(
        self,
        parameters: np.ndarray,
        first_variable: int,
        second_variable: int,
    ) -> float:
        """Implementation of a single partial second derivative.

        This function should be interpreted as (d^2 f) / (dx[first_variable] dx[second_variable]).

        In other words, this function is return how much the gradient at [second_variable] varies
        when we change the value at [first_value] a small amount, and not the other way around.

        Not that if the function is C2, this is a moot point as both are identical.

        Note: this is optional. By default, raises NotImplementedError."""
        raise NotImplementedError(
            f"ObjectiveFunction {self.__class__.__qualname__} does not"
            " implement partial_second_derivative."
        )

    def hessian(self, parameters: np.ndarray) -> np.ndarray:
        """Implementation of the hessian for the function.

        Note: this is optional. By default, raises NotImplementedError."""
        size = parameters.size
        returned_hessian = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                if self.is_c2 and i > j:
                    returned_hessian[i, j] = returned_hessian[j, i]
                else:
                    try:
                        returned_hessian[i, j] = self.partial_second_derivative(
                            parameters, i, j
                        )
                    except NotImplementedError:
                        raise NotImplementedError(
                            f"ObjectiveFunction {self.__class__.__qualname__} does not"
                            " implement hessian nor partial_second_derivative."
                        )
        return returned_hessian
