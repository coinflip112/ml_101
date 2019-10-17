#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default=50, type=int, help="Number of examples")
    parser.add_argument(
        "--plot", default=False, action="store_true", help="Plot progress"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate a binary classification data with labels [-1, 1]
    data, target = sklearn.datasets.make_classification(
        n_samples=args.examples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0,
        class_sep=2,
        random_state=args.seed,
    )
    target = 2 * target - 1

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = np.column_stack([data, np.ones(shape=(args.examples))])

    # Generate initial perceptron weights
    weights = np.random.uniform(size=data.shape[1])

    done = False
    while not done:
        

        for i in range(args.examples):
            y_pred = weights @ data[i, :]
            if target[i] * y_pred < 0:
                weights += target[i] * data[i, :]
        done = np.all(target * (data @ weights) > 0, axis=0)
        # TODO: Implement the perceptron algorithm, notably one iteration
        # over the training data in sequential order. (In practise we might
        # consider also processing the examples in a randomized order.)
        # For incorrectly examples perform the required update to the `weights`.
        # If all training instances are correctly separated, set `done=True`,
        # otherwise set `done=False`.
    if args.plot:
        plt.scatter(data[:, 0], data[:, 1], c=target)
        xs = np.linspace(*plt.gca().get_xbound() + (20,))
        ys = np.linspace(*plt.gca().get_ybound() + (20,))
        plt.contour(
            xs, ys, [[[x, y, 1] @ weights for x in xs] for y in ys], levels=[0]
        )
        plt.show()
    print(" ".join("{:.2f}".format(weight) for weight in weights))
