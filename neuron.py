import numpy as np
from active_func import ActiveFunc
from active_functions.sigmoid import Sigmoid


class Neuron:
    alpha = .0
    bias = np.random.random() * .25
    weights = []

    def __init__(self, len_inputs, alpha, active_func) -> None:
        if not isinstance(active_func, ActiveFunc):
            raise Exception('Wrong active function')

        self.weights = [np.random.random() * .5 for _ in range(len_inputs)]
        self.alpha = alpha
        self.active_func: ActiveFunc = active_func

    def __calculate_vanilla(self, inputs) -> float:
        if (len(inputs) != len(self.weights)):
            raise Exception("inputs length is not matching weights length")

        result = self.bias
        for i in range(len(self.weights)):
            result += self.weights[i] * inputs[i]

        return result

    def __calculate_delta(self, inputs, error) -> None:
        deriviative_y = self.active_func.calculate_derivative(
            self.__calculate_vanilla(inputs)
        )

        return error * deriviative_y

    def forward(self, inputs) -> float:
        y = self.__calculate_vanilla(inputs)

        return self.active_func.calculate(y)

    def backward(self, inputs, error) -> float:
        delta = self.__calculate_delta(inputs, error)

        for i in range(len(inputs)):
            self.weights[i] += self.alpha * delta * inputs[i]

        self.bias += self.alpha * delta
        
        return delta


if __name__ == '__main__':
    neuron = Neuron(2, 1, Sigmoid())
    print(neuron.forward([5, 1]))
