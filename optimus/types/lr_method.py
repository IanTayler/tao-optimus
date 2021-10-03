"""Base for building learning rate methods."""
from abc import ABC, abstractmethod

import numpy as np

from .function import Function


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
        objective_function: Function,
    ) -> float:
        pass
