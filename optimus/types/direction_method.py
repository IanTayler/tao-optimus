"""Base classes for direction methods."""
from abc import ABC, abstractmethod

import numpy as np

from .function import Function


class DirectionMethod(ABC):
    """Base class for direction methods.

    Used for findig in which direction to take a step."""

    @abstractmethod
    def __call__(
        self, parameters: np.ndarray, objective_function: Function
    ) -> np.ndarray:
        pass
