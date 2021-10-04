"""Constraints by half-spaces."""
import numpy as np

from optimus.types import Constraint


class HalfSpace(Constraint):
    """Constraint based on a linear inequality."""

    def __init__(self, vector: np.ndarray, threshold: float):
        """Create consrtaint from the defining vector and threshold.

        This defines the inequality vector.dot(point) <= threshold."""
        self.vector = vector
        self.threshold = threshold

    def project(self, point: np.ndarray) -> np.ndarray:
        dot_product_value = self.vector.dot(point)
        if dot_product_value <= self.threshold:
            return point
        return point - self.vector * (
            dot_product_value - self.threshold
        ) / np.linalg.norm(self.vector)
