"""Base classes for constraints on optimization problems."""
from abc import ABC, abstractmethod

import numpy as np


class Constraint(ABC):
    """Base class for contraints on the feasible set of optimization problems."""

    @abstractmethod
    def project(self, point: np.ndarray) -> np.ndarray:
        """Projects a point into the set of feasible points given this Constraint."""
