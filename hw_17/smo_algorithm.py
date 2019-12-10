#!/usr/bin/env python3
# dcfac0e3-1ade-11e8-9de3-00505601122b
# 7d179d73-3e93-11e9-b0fd-00505601122b
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C", default=1, type=float, help="Inverse regularization strenth"
    )
    parser.add_argument("--examples", default=200, type=int, help="Number of examples")
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
    parser.add_argument(
        "--test_ratio", default=0.5, type=float, help="Test set size ratio"
    )
    parser.add_argument(
        "--tolerance",
        default=1e-4,
        type=float,
        help="Default tolerance for KKT conditions",
    )
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.examples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=args.seed,
    )
    target = 2 * target - 1

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_ratio, random_state=args.seed
    )

    # We consider the following kernels:
    # - linear: K(x, y) = x^T y
    # - poly: K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - rbf: K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    def kernel(x, y):
        if args.kernel == "linear":
            return x @ y
        if args.kernel == "poly":
            return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * ((x - y) @ (x - y)))

    def calc_b(X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

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

    # Create initial weights
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

                a_j_new = a[j] - train_target[j] * ((Ei - Ej) / (second_derivative_j))

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
                if np.abs(a_j_new - a[j]) < args.tolerance:
                    continue

                a_i_new = a[i] - train_target[i] * train_target[j] * (a_j_new - a[j])

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
                    b = b_i
                elif 0 < a[j] < args.C:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2
                a_changed = a_changed + 1
            passes = 0 if a_changed else passes + 1

        pred_train = np.sign(
            [
                predict(a, b, train_data, train_target, train_data[o, :])
                for o in range(train_data.shape[0])
            ]
        )
        pred_test = np.sign(
            [
                predict(a, b, train_data, train_target, test_data[o, :])
                for o in range(test_data.shape[0])
            ]
        )
        train_accuracy = np.mean(pred_train == train_target)
        test_accuracy = np.mean(pred_test == test_target)
        # TODO: After each iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        print(
            "Train acc {:.1f}%, test acc {:.1f}%".format(
                100 * train_accuracy, 100 * test_accuracy
            )
        )

    if args.plot:

        def predict_simple(x):
            return (
                sum(
                    a[i] * train_target[i] * kernel(train_data[i], x)
                    for i in range(len(a))
                )
                + b
            )

        xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
        ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
        predictions = [[predict_simple(np.array([x, y])) for x in xs] for y in ys]
        plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
        plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
        plt.scatter(
            train_data[:, 0],
            train_data[:, 1],
            c=train_target,
            marker="o",
            label="Train",
            cmap=plt.cm.RdBu,
            zorder=2,
        )
        plt.scatter(
            train_data[a > args.tolerance, 0],
            train_data[a > args.tolerance, 1],
            marker="o",
            s=90,
            label="Support Vectors",
            c="#00dd00",
        )
        plt.scatter(
            test_data[:, 0],
            test_data[:, 1],
            c=test_target,
            marker="*",
            label="Test",
            cmap=plt.cm.RdBu,
            zorder=2,
        )
        plt.legend(loc="upper center", ncol=3)
        plt.show()
