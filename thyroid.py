from mlp import MLP
import pandas as pd
import numpy as np


def train_test_split(df, frac=0.2):

    # get random sample
    test = df.sample(frac=frac, axis=0)

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test


data = pd.read_csv('./data/hypothyroid.csv')
# print(data.describe())
# print(data._get_numeric_data().columns)
train, test = train_test_split(data, 0.3)

y_train = train['binaryClass']
X_train = train.drop('binaryClass', axis=1)

mlp = MLP(1, 5, 0.1)
mlp.fit(X_train, y_train)

