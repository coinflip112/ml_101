#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from scipy.special import softmax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument(
        "--classes", default=10, type=int, help="Number of classes to use"
    )
    parser.add_argument(
        "--iterations", default=50, type=int, help="Number of iterations over the data"
    )
    parser.add_argument(
        "--learning_rate", default=0.01, type=float, help="Learning rate"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=797, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed
    )

    # Generate initial model weights
    weights = np.random.uniform(size=[train_data.shape[1], args.classes])

    for iteration in range(args.iterations):
        permutation = np.random.permutation(train_data.shape[0])
        permuted_x_train, permuted_y_train = (
            train_data[permutation],
            train_target[permutation],
        )
        batch_count = int(train_data.shape[0] / args.batch_size)
        for batch_x, batch_y in zip(
            np.split(permuted_x_train, batch_count),
            np.split(permuted_y_train, batch_count),
        ):
            logits = batch_x @ weights
            probs = softmax(logits, axis=1)
            batch_y = np.eye(args.classes)[batch_y]
            grads =  np.dot(batch_x.T, probs - batch_y)/args.batch_size
            weights -= args.learning_rate * grads
        
        
        
        predictions_train = np.argmax(softmax(train_data @ weights, axis=1), axis=1)
        predictions_test = np.argmax(softmax(test_data @ weights, axis=1), axis=1)
        print(
            "After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
                iteration + 1,
                100
                * sklearn.metrics.accuracy_score(
                    train_target, predictions_train
                ),  # Training accuracy,
                100
                * sklearn.metrics.accuracy_score(
                    test_target, predictions_test
                ),  # Test accuracy,
            )
        )
