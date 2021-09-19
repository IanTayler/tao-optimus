from abc import ABC, abstractmethod

import numpy as np

from gradient_descent.objective_function import ObjectiveFunction


class DirectionMethod(ABC):
    """Base class for direction methods.

    Used for findig in which direction to take a step."""

    @abstractmethod
    def __call__(
        self, parameters: np.ndarray, objective_function: ObjectiveFunction
    ) -> np.ndarray:
        pass
