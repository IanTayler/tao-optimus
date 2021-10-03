"""Base classes for constraints on optimization problems."""
from abc import ABC, abstractmethod


class Constraint(ABC):
    """Base class for contraints on the feasible set of optimization problems."""

    @abstractmethod
    def project(self, point):
        """Projects a point into the set of feasible points given this Constraint."""
