"""Constraints by balls (i.e. center and radius)."""
import numpy as np

from optimus.types import Constraint


class Ball(Constraint):
    """Constraint based on a ball."""

    def __init__(self, center: np.ndarray, radius: float):
        """Create consrtaint from the defining vector and threshold."""
        self.center = center
        self.radius = radius

    def project(self, point: np.ndarray) -> np.ndarray:
        difference_vector = point - self.center
        distance = np.linalg.norm(difference_vector)
        if distance <= self.radius:
            return point
        return point - (distance - self.radius) * difference_vector
