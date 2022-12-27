from __future__ import annotations
from neuron import Neuron
from active_functions.sigmoid import Sigmoid
from typing import List
import numpy as np
import pandas as pd


class MLP:
    epoches = 0
    len_hidden = 0
    hidden_layer: List[Neuron] = []
    class_layer: List[Neuron] = []
    alpha = None

    def __init__(self, epoches, len_hidden, alpha) -> None:
        self.epoches = epoches
        self.len_hidden = len_hidden
        self.alpha = alpha

    def __init_hidden_layer(self, input_len):
        for _ in range(self.len_hidden):
            neuron = Neuron(input_len, self.alpha, Sigmoid())
            self.hidden_layer.append(neuron)
            
    def __init_class_layer(self, class_count):
        for _ in range(class_count):
            neuron = Neuron(len(self.hidden_layer), self.alpha, Sigmoid())
            self.class_layer.append(neuron)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        shape = X_train.shape
        unique_values_count = len(y_train.unique())

        self.__init_hidden_layer(shape[1])
        self.__init_class_layer(unique_values_count)

        for row_index, row in X_train.iterrows():
            results = []
            for neuron in self.hidden_layer:
                results.append(neuron.forward(row))

            deltas = []
            for i in range(len(self.class_layer)):
                predicted = self.class_layer[i].forward(results)
                error = y_train[row_index] - predicted
                new_delta = self.class_layer[i].backward(results, error)
                deltas = np.sum([deltas, new_delta], axis=0)

            for i in range(len(self.hidden_layer)):
                self.hidden_layer[i].backward(row, deltas[i])

    def save_weights(self, save_path) -> None:
        pass

    @classmethod
    def load_weights(cls) -> MLP:
        pass
