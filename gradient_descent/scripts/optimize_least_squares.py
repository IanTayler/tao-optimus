import numpy as np

from gradient_descent import optimize
from gradient_descent.direction_method import gradient
from gradient_descent.lr_method import armijo
from gradient_descent.objective_function import least_squares


def _main():
    with open("A.asc") as inputf:
        inputs = np.loadtxt(inputf)
    with open("b.asc") as targetf:
        targets = np.loadtxt(targetf)
    direction_method = gradient.Gradient()
    lr_method = armijo.Armijo(1e-3, 0.1, 0.8, max_iters=10)
    objective_function = least_squares.LeastSquares(inputs, targets)
    initial_params = np.array([0.1, 0.1])
    initial_value = objective_function(initial_params)
    params = optimize.optimize(
        np.array(initial_params),
        objective_function,
        direction_method,
        lr_method,
        max_steps=1000,
        verbose=True,
    )
    print(
        f"Initial value: {initial_value}. Final value: {objective_function(params)}."
        f"\nFinal params: {params}"
    )


if __name__ == "__main__":
    _main()
