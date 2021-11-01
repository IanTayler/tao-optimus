"""Main optimization functions."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from optimus.types import DirectionMethod, Function, LRMethod


class StopReason(Enum):
    SMALL_GRADIENT = "Gradient norm too small"
    SMALL_STEP = "Movement step norm too small"


@dataclass
class OptimizeInfo:
    function_value: float
    gradient: np.ndarray
    direction: np.ndarray
    lr: float
    step_vector: np.ndarray
    parameters: np.ndarray
    step: int
    stop_reason: Optional[StopReason]


def _should_stop(
    gradient,
    step_vector,
    minimum_gradient_norm,
    minimum_step_norm,
) -> Optional[StopReason]:
    # Stop if gradient is too small:
    # This means we're close (enough) to a critical point.
    gradient_norm = np.linalg.norm(gradient)
    if gradient_norm < minimum_gradient_norm:
        return StopReason.SMALL_GRADIENT
    # Also stop if the step is too small.
    step_norm = np.linalg.norm(step_vector)
    if step_norm < minimum_step_norm:
        return StopReason.SMALL_STEP
    return None


def optimize(
    initial_parameters: np.ndarray,
    function: Function,
    direction_method: DirectionMethod,
    lr_method: LRMethod,
    max_steps: int,
    minimum_gradient_norm: float = 0.1,
    minimum_step_norm: float = 1e-3,
) -> np.ndarray:
    """Run a gradient-descent style algorithm until it converges or `max_steps` is reached."""
    step = 0
    parameters = initial_parameters
    stop_reason = None
    while step < max_steps and stop_reason is None:
        function_value = function(parameters)
        gradient = function.gradient(parameters)
        direction = direction_method(parameters, function)
        lr = lr_method(parameters, function_value, gradient, direction, step, function)
        step_vector = lr * direction
        parameters -= step_vector
        step += 1
        yield OptimizeInfo(
            function_value,
            gradient,
            direction,
            lr,
            step_vector,
            parameters,
            step,
            stop_reason,
        )
