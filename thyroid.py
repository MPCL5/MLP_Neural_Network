from mlp import MLP
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_test_split(df, frac=0.2):
    # get random sample
    test = df.sample(frac=frac, axis=0, weights='target')

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test


if __name__ == "__main__":
    df = pd.read_csv('./data/data.csv')
    values_counts = df["target"].value_counts()

    # print(df.describe())
    train, test = train_test_split(df, 0.3)

    y_train = train['target']
    X_train = train.drop('target', axis=1)
    y_test = test['target'].tolist()
    X_test = test.drop('target', axis=1)

    mlp = MLP(8, 10, 0.1)
    mlp.fit(X_train, y_train)
    mlp.save_weights('./test.json')

    # mlp = MLP.load_weights('./test.json')
    y_pred = mlp.predict(X_test)
    print(f'Accuracy is: {accuracy_score(y_test, y_pred)}')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

