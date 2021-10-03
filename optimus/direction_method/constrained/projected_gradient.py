"""Move inside the feasible set by taking a GD step and then projecting."""
import numpy as np

from optimus.types import Constraint, DirectionMethod, LRMethod, OptimusFunction


class Projected(DirectionMethod):
    """Direction is some other method + projection to constraints."""

    def __init__(
        self,
        internal_direction_method: DirectionMethod,
        internal_lr_method: LRMethod,
        constraint: Constraint,
    ):
        self.internal_direction_method = internal_direction_method
        self.internal_lr_method = internal_lr_method
        self.constraint = constraint

    def __call__(
        self, parameters: np.ndarray, objective_function: OptimusFunction
    ) -> np.ndarray:
        internal_direction = self.internal_direction_method(
            parameters, objective_function
        )
        internal_lr = self.internal_lr_method(
            parameters,
            objective_function(parameters),
            objective_function.gradient(parameters),
            internal_direction,
            step=1,
            objective_function=objective_function,
        )
        unprojected_new_point = parameters - internal_lr * internal_direction
        return self.constraint.project(unprojected_new_point)
