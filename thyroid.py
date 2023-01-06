from mlp import MLP
import pandas as pd
import numpy as np


def train_test_split(df, frac=0.2):
    # get random sample
    test = df.sample(frac=frac, axis=0)

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test


df = pd.read_csv('./data/data.csv')
# for col in list(df.columns):
#     mapping = {label: idx for idx, label in enumerate(np.unique(df[col]))}  # make your mapping dict
#     df[col] = df[col].map(mapping)  # map your class

print(df.describe())
train, test = train_test_split(df, 0.3)
#
y_train = train['target']
X_train = train.drop('target', axis=1)
y_test = test['target']
X_test = test.drop('target', axis=1)
# print(y_train.iloc[0])
# print(X_train.iloc[0])

mlp = MLP(10, 5, 0.1)
mlp.fit(X_train, y_train)
wrong = 0
for row_index, row in X_test.iterrows():
    if np.argmax(mlp.predict(row)) + 1 != int(y_test.loc[row_index]):
        wrong += 1
        print(mlp.predict(row))
        print(y_test.loc[row_index])

print(wrong/X_test.shape[0])
