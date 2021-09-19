import numpy as np
from gradient_descent.objective_function import ObjectiveFunction


def optimize(
    parameters: np.ndarray,
    function: ObjectiveFunction,
    gradient_transform: GradientTransform,
    lr_method: LRMethod,
    stop_criterion: StopCriterion,
    max_steps: Optional[int],
):
    function()
