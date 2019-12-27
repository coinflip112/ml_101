#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from itertools import combinations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C", default=1, type=float, help="Inverse regularization strenth"
    )
    parser.add_argument("--classes", default=5, type=int, help="Number of classes")
    parser.add_argument(
        "--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]"
    )
    parser.add_argument(
        "--kernel_degree", default=5, type=int, help="Degree for poly kernel"
    )
    parser.add_argument(
        "--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel"
    )
    parser.add_argument(
        "--num_passes",
        default=10,
        type=int,
        help="Number of passes without changes to stop after",
    )
    parser.add_argument(
        "--plot", default=False, action="store_true", help="Plot progress"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=701, type=int, help="Test set size")
    parser.add_argument(
        "--tolerance",
        default=1e-4,
        type=float,
        help="Default tolerance for KKT conditions",
    )
    args = parser.parse_args()

    def kernel(x, y):
        if args.kernel == "linear":
            return x @ y
        if args.kernel == "poly":
            return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * ((x - y) @ (x - y)))

    def clip(a, H, L):
        if a > H:
            return H
        if L > a:
            return L
        return a

    def predict(a, b, train_data, train_target, x):
        return (
            sum(
                a[i] * train_target[i] * kernel(train_data[i], x) for i in range(len(a))
            )
            + b
        )

    def smo(train_data, train_target, test_data, args):
        a, b = np.zeros(len(train_data)), 0
        j_generator = np.random.RandomState(args.seed)

        passes = 0

        while passes < args.num_passes:
            a_changed = 0
            for i in range(len(a)):
                pred_i = predict(a, b, train_data, train_target, train_data[i, :])
                Ei = pred_i - train_target[i]
                cond_1 = (a[i] < args.C) and (train_target[i] * Ei < -args.tolerance)
                cond_2 = (a[i] > 0) and (train_target[i] * Ei > args.tolerance)
                if cond_1 or cond_2:
                    j = j_generator.randint(len(a) - 1)
                    j = j + (j >= i)
                    pred_j = predict(a, b, train_data, train_target, train_data[j, :])
                    Ej = pred_j - train_target[j]

                    second_derivative_j = (
                        2 * kernel(train_data[i,], train_data[j,])
                        - kernel(train_data[i,], train_data[i,])
                        - kernel(train_data[j,], train_data[j,])
                    )

                    a_j_new = a[j] - train_target[j] * (
                        (Ei - Ej) / (second_derivative_j)
                    )

                    if second_derivative_j >= -args.tolerance:
                        continue

                    if train_target[i] == train_target[j]:
                        L = np.maximum(0, a[i] + a[j] - args.C)
                        H = np.minimum(args.C, a[i] + a[j])
                    else:
                        L = np.maximum(0, a[j] - a[i])
                        H = np.minimum(args.C, args.C + a[j] - a[i])

                    if (H - L) < args.tolerance:
                        continue
                        # nothing

                    a_j_new = clip(a_j_new, H, L)

                    if abs(a_j_new - a[j]) <= args.tolerance:
                        continue

                    a_i_new = a[i] - train_target[i] * train_target[j] * (
                        a_j_new - a[j]
                    )

                    b_j = (
                        b
                        - Ej
                        - train_target[i]
                        * (a_i_new - a[i])
                        * kernel(train_data[i,], train_data[j,])
                        - train_target[j]
                        * (a_j_new - a[j])
                        * kernel(train_data[j,], train_data[j,])
                    )
                    b_i = (
                        b
                        - Ei
                        - train_target[i]
                        * (a_i_new - a[i])
                        * kernel(train_data[i,], train_data[i,])
                        - train_target[j]
                        * (a_j_new - a[j])
                        * kernel(train_data[j,], train_data[i,])
                    )
                    a[j] = a_j_new
                    a[i] = a_i_new

                    # - increase a_changed
                    if 0 < a[i] < args.C:
                        b = b_i.copy()
                    elif 0 < a[j] < args.C:
                        b = b_j.copy()
                    else:
                        b = (b_i + b_j) / 2
                    a_changed = a_changed + 1
                passes = 0 if a_changed else passes + 1
            pred_test = np.sign(
                [
                    predict(a, b, train_data, train_target, test_data[o, :])
                    for o in range(test_data.shape[0])
                ]
            )
        return a, b, pred_test

    # Set random seed
    np.random.seed(args.seed)

    # Load the digits dataset with specified number of classes, and normalize it.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data /= np.max(data)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed
    )

    unique_classes = np.unique(target)

    classifiers = {}
    predictions = {}
    for class_a, class_b in combinations(unique_classes, 2):
        relevant_classes = pd.Series(train_target).isin((class_a, class_b))

        train_data_classed = train_data[relevant_classes, :]

        train_target_classed = train_target[relevant_classes]
        train_target_classed = (train_target_classed == class_b).astype(np.int8)
        train_target_classed = 2 * train_target_classed - 1

        a_classed, b_classed, test_pred_classed = smo(
            train_data_classed, train_target_classed, test_data, args
        )

        classifiers[(class_a, class_b)] = (a_classed, b_classed)

        test_pred_classed = np.where(
            np.array(test_pred_classed) == -1, class_a, class_b
        )
        predictions[(class_a, class_b)] = test_pred_classed

    predictions_test_set = pd.DataFrame(predictions)
    predictions_test_set = (
        predictions_test_set.apply(lambda row: row.mode().iloc[0], axis=1)
        .squeeze()
        .values
    )

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes.

    # Then, classify the test set by majority voting, using the lowest class
    # index in case of ties. Finally compute `test accuracy`.

    print(
        "{:.2f}".format(
            100 * (np.array(predictions_test_set) == np.array(test_target)).mean()
        )
    )

