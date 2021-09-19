from gradient_descent.lr_method import LRMethod


class Constant(LRMethod):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, *args, **kwargs):
        return self.learning_rate
