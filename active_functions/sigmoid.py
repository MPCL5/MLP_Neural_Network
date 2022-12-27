import numpy as np
from active_func import ActiveFunc


class Sigmoid(ActiveFunc):
    def calculate(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_derivative(self, x):
        return x * (1.0 - x)
