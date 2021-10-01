"""Constant learning rate."""
from optimus.lr_method import LRMethod


class Constant(LRMethod):
    """Used when a single learning rate will be used for all steps."""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, *args, **kwargs):
        return self.learning_rate
