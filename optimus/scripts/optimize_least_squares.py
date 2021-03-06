"""Quick script for running a least squares regression."""
import click
import numpy as np

from optimus import optimize
from optimus.data import load, save
from optimus.direction_method import diagonal, gradient, newton
from optimus.lr_method import armijo, constant
from optimus.function import least_squares

direction_methods = {
    "newton": newton.Newton,
    "gradient": gradient.Gradient,
    "quasihessian": diagonal.DiagonalQuasiHessian,
}
lr_methods = {
    "armijo": lambda lr: armijo.Armijo(
        lr, tolerance=0.1, decrease_factor=0.5, max_iters=20
    ),
    "constant": constant.Constant,
}


@click.command(name="optimize")
@click.option(
    "direction_method_name",
    "--direction-method",
    type=click.Choice(list(direction_methods.keys())),
    default="gradient",
)
@click.option(
    "lr_method_name",
    "--lr-method",
    type=click.Choice(list(lr_methods.keys())),
    default="armijo",
)
@click.option("--lr", type=float, default=1e-3)
@click.option(
    "inputs_path", "--inputs", "-i", type=click.Path(exists=True), default="A.asc"
)
@click.option(
    "targets_path", "--targets", "-t", type=click.Path(exists=True), default="b.asc"
)
@click.option(
    "output_path", "--output", "-o", type=click.Path(exists=False), default="best.npy"
)
def _main(
    direction_method_name: str,
    lr_method_name: str,
    lr: float,
    inputs_path: str,
    targets_path: str,
    output_path: str,
):
    # Load data
    inputs = load.load_numpy(inputs_path)
    targets = load.load_numpy(targets_path)
    # Build classes describing the details of the optimization problem and
    # method.
    direction_method = direction_methods[direction_method_name]()
    lr_method = lr_methods[lr_method_name](lr)
    objective_function = least_squares.LeastSquares(inputs, targets)
    # Initialize state.
    initial_params = np.random.normal(0.0, 1.0, inputs.shape[1])
    initial_value = objective_function(initial_params)
    params = initial_params
    for step_info in optimize.optimize(
        initial_params,
        objective_function,
        direction_method,
        lr_method,
        max_steps=1000,
    ):
        params = step_info.parameters
        print(f"Step {step_info.step}.")
        print(f"Function value {step_info.function_value}")
        print(f"Gradient norm: {np.linalg.norm(step_info.gradient)}.")
    print(
        f"Initial value: {initial_value}. Final value: {objective_function(params)}."
        f"\nFinal params: {params}"
    )
    save.save_numpy(output_path, params)


if __name__ == "__main__":
    _main()
