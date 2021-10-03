"""Base classes for direction methods."""
from abc import ABC, abstractmethod

import numpy as np

from .optimus_function import OptimusFunction


class DirectionMethod(ABC):
    """Base class for direction methods.

    Used for findig in which direction to take a step."""

    @abstractmethod
    def __call__(
        self, parameters: np.ndarray, objective_function: OptimusFunction
    ) -> np.ndarray:
        pass
