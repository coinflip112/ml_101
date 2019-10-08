#!/usr/bin/python
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.metrics import mean_squared_error


def rms(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", default=50, type=int, help="Test size to use")
    args = parser.parse_args()

    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()
    print(dataset.DESCR)

    # The input data are in dataset.data, targets are in dataset.target.

    # TODO: Pad a value of `1` to every instance in dataset.data
    # (np.pad or np.concatenate might be useful).
    X = dataset.data
    y = dataset.target
    # TODO: Split data so that the last `args.test_size` data are the test
    # set and the rest is the training set.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=args.test_size
    )
    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using np.linalg.inv).
    ols_weights = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    # TODO: Predict target values on the test set.
    y_test_pred = X_test @ ols_weights
    # TODO: Compute root mean square error on the test set predictions.
    rmse = rms(y_test, y_test_pred)

    with open("linear_regression_manual.out", "w") as output_file:
        print("{:.2f}".format(rmse), file=output_file)
