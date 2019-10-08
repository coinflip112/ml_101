#!/usr/bin/python
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
from sklearn.metrics import mean_squared_error

def rms(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_val_set_rmse(X_train, X_test, y_train, y_test, alpha):
    ridge = sklearn.linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    rmse = rms(y_test, y_pred)
    return [alpha, rmse]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", default=False, action="store_true", help="Plot the results"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=50, type=int, help="Test size to use")
    args = parser.parse_args()

    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()
    X = dataset.data
    y = dataset.target

    # TODO: Split the dataset randomly to train and test using
    # `sklearn.model_selection.train_test_split`, with
    # `test_size=args.test_size` and `random_state=args.seed`.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    # TODO: Using sklearn.linear_model.Ridge, fit the train set using
    # L2 regularization, employing lambdas from 0 to 100 with a step size 0.1.
    # For every model, compute the root mean squared error
    # (sklearn.metrics.mean_squared_error may come handy) and print out
    # the lambda producing lowest test error.
    alpha_space = np.linspace(start = 0.0, stop = 100.0, num = 1001)
    results = np.array([get_val_set_rmse(X_train, X_test, y_train, y_test, alpha) for alpha in alpha_space])
    best_alpha_id = np.argmin(results[:,1])
    best_lambda = results[best_alpha_id,0]
    best_rmse = results[best_alpha_id,1]

    with open("linear_regression_l2.out", "w") as output_file:
        print("{:.1f}, {:.2f}".format(best_lambda, best_rmse), file=output_file)

    if args.plot:
        # TODO: This block is not part of ReCodEx submission, so you
        # will get points even without it. However, it is useful to
        # learn to visualize the results.

        # If you collect used lambdas to `lambdas` and their respective
        # results to `rmses`, the following lines will plot the result
        # if you add `--plot` argument.
        import matplotlib.pyplot as plt

        plt.plot(lambdas, rmses)
        plt.show()

