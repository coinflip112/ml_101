#!/usr/bin/env python3
# dcfac0e3-1ade-11e8-9de3-00505601122b
# dce9cf60-42b6-11e9-b0fd-00505601122b
# 7d179d73-3e93-11e9-b0fd-00505601122b

import argparse

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


class DecisionTree:
    class Node:
        def __init__(self, instances, prediction):
            self.is_leaf = True
            self.instances = instances
            self.prediction = prediction

        def split(self, feature, value, left, right):
            self.is_leaf = False
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right

    def __init__(self, max_depth, min_to_split, generator, feature_subsampling):
        self._max_depth = max_depth
        self._min_to_split = min_to_split
        self._generator = generator
        self._feature_subsampling = feature_subsampling

    def fit(self, data, targets):
        self._data = data
        self._targets = targets

        self._root = self._create_leaf(np.arange(len(self._data)))
        self._split_recursively(self._root, 0)

        return self

    def predict(self, data):
        results = np.zeros(len(data), dtype=np.int32)
        for i in range(len(data)):
            node = self._root
            while not node.is_leaf:
                node = node.left if data[i][node.feature] <= node.value else node.right
            results[i] = node.prediction

        return results

    def _split_recursively(self, node, depth):
        if not self._can_split(node, depth):
            return

        feature, value, left, right = self._best_split(node)
        node.split(feature, value, self._create_leaf(left), self._create_leaf(right))
        self._split_recursively(node.left, depth + 1)
        self._split_recursively(node.right, depth + 1)

    def _can_split(self, node, depth):
        return (
            (self._max_depth is None or depth < self._max_depth)
            and len(node.instances) >= self._min_to_split
            and not np.array_equiv(self._targets[node.instances], node.prediction)
            and not np.array_equiv(
                self._data[node.instances], self._data[node.instances[0]]
            )
        )

    def _best_split(self, node):
        best_criterion = None
        mask = (
            self._generator.uniform(size=self._data.shape[1])
            <= self._feature_subsampling
        )
        for feature_usable, feature in zip(mask, range(self._data.shape[1])):
            if not feature_usable:
                continue
            node_features = self._data[node.instances, feature]
            separators = np.unique(node_features)
            for i in range(len(separators) - 1):
                value = (separators[i] + separators[i + 1]) / 2
                left, right = (
                    node.instances[node_features <= value],
                    node.instances[node_features > value],
                )
                criterion = self._criterion(left) + self._criterion(right)
                if best_criterion is None or criterion < best_criterion:
                    best_criterion, best_feature, best_value, best_left, best_right = (
                        criterion,
                        feature,
                        value,
                        left,
                        right,
                    )

        return best_feature, best_value, best_left, best_right

    def _criterion(self, instances):
        # We use Gini index
        bins = np.bincount(self._targets[instances])
        return np.sum(bins * (1 - bins / len(instances)))

    def _create_leaf(self, instances):
        # Create a new leaf, together with its prediction (the most frequent class)
        return self.Node(instances, np.argmax(np.bincount(self._targets[instances])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrapping",
        default=False,
        action="store_true",
        help="Perform data bootstrapping",
    )
    parser.add_argument(
        "--feature_subsampling",
        default=1,
        type=float,
        help="What fraction of features to subsample",
    )
    parser.add_argument(
        "--max_depth", default=None, type=int, help="Maximum decision tree depth"
    )
    parser.add_argument(
        "--min_to_split", default=2, type=int, help="Minimum examples required to split"
    )
    parser.add_argument(
        "--n_estimators", default=1, type=int, help="Number of trees in the forest"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=42, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed
    )

    # TODO: Create a random forest on the trainining data.
    #
    # For determinism, create a generator
    #   generator = np.random.RandomState(args.seed)
    # at the beginning and then use it for all random number generation.
    #
    generator = np.random.RandomState(args.seed)
    forest = {}
    for i in range(args.n_estimators):

        tree = DecisionTree(
            args.max_depth, args.min_to_split, generator, args.feature_subsampling
        )
        if args.bootstrapping:
            indices = generator.randint(len(train_data), size=len(train_data))
            tree.fit(train_data[indices, :], train_target[indices])
        else:
            tree.fit(train_data, train_target)
        forest[i] = tree

    predictions_train = {key: tree.predict(train_data) for key, tree in forest.items()}
    predictions_test = {key: tree.predict(test_data) for key, tree in forest.items()}

    predictions_train = pd.DataFrame(predictions_train)
    predictions_test = pd.DataFrame(predictions_test)

    predictions_train = (
        predictions_train.apply(lambda row: row.mode().iloc[0], axis=1).squeeze().values
    )
    predictions_test = (
        predictions_test.apply(lambda row: row.mode().iloc[0], axis=1).squeeze().values
    )
    # You can use the given implementation of DecisionTree, which you need to
    # modify to support feature subsampling. When searching for the best split
    # (in the order in which the given implementation creates nodes), try only
    # a subset of features. Notably generate for every node a feature mask
    # using
    #   generator.uniform(size=number_of_features) <= feature_subsampling
    # which gives a boolean value for every feature, with `True` meaning the
    # feature is used during best split search, and `False` it is not.
    # (When feature_subsampling == 1, all features are used.)
    #
    # Finally train a random forest consisting of `args.n_estimators` decision trees.
    # If `args.bootstrapping` is set, right before training every decision tree
    # create a bootstrap sample of the training data using the indices
    #   indices = self._random_generator.randint(len(train_data), size=len(train_data))
    # and if `args.bootstrapping` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for given
    # input, choosing the smaller index number in case of a tie.

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = np.mean(predictions_train == train_target)
    test_accuracy = np.mean(predictions_test == test_target)
    print("Train acc: {:.1f}%".format(100 * train_accuracy))
    print("Test acc: {:.1f}%".format(100 * test_accuracy))
