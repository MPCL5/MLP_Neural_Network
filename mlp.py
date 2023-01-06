from __future__ import annotations

from active_functions.ReLU import ReLU
from neuron import Neuron
from active_functions.sigmoid import Sigmoid
from typing import List
import numpy as np
import pandas as pd


class MLP:
    epochs = 0
    len_hidden = 0
    hidden_layer: List[Neuron] = []
    class_layer: List[Neuron] = []
    alpha = None

    def __init__(self, epochs, len_hidden, alpha) -> None:
        self.epochs = epochs
        self.len_hidden = len_hidden
        self.alpha = alpha

    def __init_hidden_layer(self, input_len):
        for _ in range(self.len_hidden):
            neuron = Neuron(input_len, self.alpha, ReLU())
            self.hidden_layer.append(neuron)

    def __init_class_layer(self, class_count):
        for _ in range(class_count):
            neuron = Neuron(len(self.hidden_layer), self.alpha, ReLU())
            self.class_layer.append(neuron)

    def __predict_one(self, input) -> List[float]:
        results = []
        for neuron in self.hidden_layer:
            results.append(neuron.forward(input))

        predicted = []
        for i in range(len(self.class_layer)):
            # forward in last layer
            predicted.append(self.class_layer[i].forward(results))

        return predicted

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        input_count = x_train.shape[1]  # get input count
        unique_values_count = len(y_train.unique())

        self.__init_hidden_layer(input_count)  # vanilla hidden layer neurons
        self.__init_class_layer(unique_values_count)  # vanilla class neurons

        for _ in range(self.epochs):
            for row_index, row in x_train.iterrows():
                results = []
                # forward in the hidden layer.
                for i in range(len(self.hidden_layer)):
                    results.append(self.hidden_layer[i].forward(row))

                deltas = np.zeros(len(self.hidden_layer))
                for i in range(len(self.class_layer)):
                    # forward in the class layer
                    predicted = self.class_layer[i].forward(results)
                    expected = 1 if y_train[row_index] == i + 1 else 0
                    error = expected - predicted

                    # backward in the last layer
                    new_delta = self.class_layer[i].backward(results, error)
                    deltas = np.sum([deltas, new_delta], axis=0)

                # backward in the hidden layer
                for i in range(len(self.hidden_layer)):
                    self.hidden_layer[i].backward(row, deltas[i])

    def predict(self, input_vector):
        results = []
        # forward in the hidden layer.
        for i in range(len(self.hidden_layer)):
            results.append(self.hidden_layer[i].forward(input_vector))

        predicted = np.zeros(len(self.class_layer))
        for i in range(len(self.class_layer)):
            # forward in the class layer
            predicted[i] = self.class_layer[i].forward(results)

        return predicted

    def save_weights(self, save_path) -> None:
        pass

    @classmethod
    def load_weights(cls) -> MLP:
        pass
