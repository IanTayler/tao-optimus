# Optimus

Several variations on iterative methods for optimization, implemented from scratch using numpy. This is only meant for learning purposes. If you want fast, robust and efficient implementations of optimization algorithms, you should look elsewhere.

This module is organized assuming the optimization step is conceptualized as `x := x - lr * d`. So, taking a step amounts to choosing a direction in which to move, and an amount to move.

The methods for choosing a direction are implemented in `optimus.direction_method`. Methods for finding how much to move are in `optimus.lr_method`. Objective functions (i.e. the functions that can be optimized) are implemented under `optimus.optimus_function`. These functions are also used to define constraints, and elsewhere where a differentiable function might be needed. Finally, the optimization function that puts everything together is in `optimus.optimize`.

## Scripts

For quickly running some of the optimization methods implemented, check `scripts/optimize_least_squares.py`.

You should first download suitable data for inputs (A) and targets (b) and store them as separate files in `.asc` or `.npy` formats. Then you can check supported methods by running `poetry run python scripts/optimize_least_squares.py --help` from this directory.
