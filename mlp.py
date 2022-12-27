from __future__ import annotations
from neuron import Neuron
from active_functions.sigmoid import Sigmoid
import pandas as pd


class MLP:
    epoches = 0
    len_hidden = 0
    hidden_layer = []
    class_layer = []
    alpha = None

    def __init__(self, epoches, len_hidden, alpha) -> None:
        self.epoches = epoches
        self.len_hidden = len_hidden
        self.alpha = alpha

    def __init_hidden_layer(self, input_len):
        for _ in range(input_len):
            neuron = Neuron(input_len, self.alpha, Sigmoid())
            self.hidden_layer.append(neuron)
            
    def __init_class_layer(self, class_count):
        for i in range(class_count):
            neuron = Neuron(len(self.hidden_layer), self.alpha, Sigmoid())

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        shape = X_train.shape
        self.__init_hidden_layer(shape[0])

        for i, row in X_train.iterrows():
            print(i)
            print(y_train.loc[i])

    def save_weights(self, save_path) -> None:
        pass

    @classmethod
    def load_weights(cls) -> MLP:
        pass
