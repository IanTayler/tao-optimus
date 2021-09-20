"""Base for building learning rate methods."""
from abc import ABC, abstractmethod

import numpy as np

from gradient_descent.objective_function import ObjectiveFunction


class LRMethod(ABC):
    """Base Learning Rate class."""

    @abstractmethod
    def __call__(
        self,
        parameters: np.ndarray,
        function_value: float,
        gradient: np.ndarray,
        direction: np.ndarray,
        step: int,
        objective_function: ObjectiveFunction,
    ) -> float:
        pass
