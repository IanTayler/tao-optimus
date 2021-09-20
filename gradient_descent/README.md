# Gradient descent

Several variations on gradient descent methods for optimization, implemented from scratch using numpy. This is only meant for learning purposes. If you want fast, robust and efficient implementations of optimization algorithms, you should look elsewhere.

This module is organized assuming the gradient descent step is conceptualized as `x := x - lr * d`. So, taking a step amounts to choosing a direction in which to move, and an amount to move.

The methods for choosing a direction are implemented in `gradient_descent.direction_method`. Methods for finding how much to move are in `gradient_descent.lr_method`. Objective functions (i.e. the functions that can be optimized) are implemented under `gradient_descent.objective_function`. Finally, the optimization function that puts everything together is in `gradient_descent.optimize`.
