import numpy as np
from active_func import ActiveFunc


class ReLU(ActiveFunc):
    def calculate(self, x):
        return np.max([0, x])

    def calculate_derivative(self, x):
        return 1 if x > 0 else 0
