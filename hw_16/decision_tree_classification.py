#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


class DecisionTree:
    def __init__(
        self,
        max_depth=None,
        min_to_split=2,
        max_leaves=None,
        criterion="gini",
        depth=0,
        leaves=0,
    ):
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.max_leaves = max_leaves
        self.criterion = criterion
        self.leaves = leaves
        self.depth = depth

    def _entropy(self, targets):
        count = targets.shape[0]
        value_counts = pd.Series(targets).value_counts(normalize=True)
        return -count * np.sum(value_counts.values * np.log(value_counts.values))

    def _gini(self, targets):
        count = targets.shape[0]
        value_counts = pd.Series(targets).value_counts(normalize=True)
        return count * np.sum(value_counts.values * (1 - value_counts.values))

    def _evaluate(self, targets):
        if self.criterion == "gini":
            return self._gini(targets)
        if self.criterion == "entropy":
            return self._entropy(targets)
        return None

    def get_split_points(self, feature):
        feature = np.sort(np.unique(feature))
        if feature.shape[0] <= 1:
            return None
        split_points = np.mean(list(zip(feature[:-1], feature[1:])), axis=1)
        return split_points

    def get_best_split_point(self, feature, targets):
        split_points = self.get_split_points(feature)
        if split_points is None:
            return None, -np.inf
        split_scores = []
        for split_point in split_points:
            left_targets = targets[feature <= split_point]
            right_targets = targets[feature >= split_point]
            score_left = self._evaluate(left_targets)
            score_right = self._evaluate(right_targets)
            score_split = score_left + score_right - self.score
            split_scores.append(score_split)

        best_split_index = np.argmin(split_scores)
        return split_points[best_split_index], split_scores[best_split_index]

    def split(self, features, targets):
        self.score = self._evaluate(targets)
        if (not self.min_to_split is None) and (targets.shape[0] < self.min_to_split):
            self.is_leaf = True
            self.class_proba = pd.Series(targets).value_counts()
            return None
        if (not self.max_depth is None) and (self.max_depth <= self.depth):
            self.is_leaf = True
            self.class_proba = pd.Series(targets).value_counts()
            return None
        if np.unique(targets).shape[0] <= 1:
            self.is_leaf = True
            self.class_proba = pd.Series(targets).value_counts()
            return None
        feature_split_points = []
        feature_split_scores = []

        for feature_index in range(features.shape[1]):
            feature = features[:, feature_index]
            split_point, split_score = self.get_best_split_point(feature, targets)
            feature_split_points.append(split_point)
            feature_split_scores.append(split_score)

        best_feature_index = np.argmin(feature_split_scores)
        best_feature_split_point = feature_split_points[best_feature_index]

        selected_feature = features[:, best_feature_index]

        self.selected_feature = best_feature_index
        self.best_feature_split_point = best_feature_split_point

        left_features = features[selected_feature <= best_feature_split_point, :]
        left_targets = targets[selected_feature <= best_feature_split_point]
        right_features = features[selected_feature > best_feature_split_point, :]
        right_targets = targets[selected_feature > best_feature_split_point]

        self.left_tree = DecisionTree(
            max_depth=self.max_depth,
            min_to_split=self.min_to_split,
            max_leaves=self.max_leaves,
            criterion=self.criterion,
            depth=self.depth + 1,
        )
        self.right_tree = DecisionTree(
            max_depth=self.max_depth,
            min_to_split=self.min_to_split,
            max_leaves=self.max_leaves,
            criterion=self.criterion,
            depth=self.depth + 1,
        )
        self.left_tree.split(features=left_features, targets=left_targets)
        self.right_tree.split(features=right_features, targets=right_targets)
        self.is_leaf = False
        return self

    def predict(self, observation):
        if self.is_leaf:
            prediction = self.class_proba.idxmax()
            return prediction
        else:
            relevant_feature = observation[self.selected_feature]
            if relevant_feature <= self.best_feature_split_point:
                return self.left_tree.predict(observation)
            else:
                return self.right_tree.predict(observation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--criterion",
        default="gini",
        type=str,
        help="Criterion to use; either `gini` or `entropy`",
    )
    parser.add_argument(
        "--max_depth", default=None, type=int, help="Maximum decision tree depth"
    )
    parser.add_argument(
        "--min_to_split", default=2, type=int, help="Minimum examples required to split"
    )
    parser.add_argument(
        "--max_leaves", default=None, type=int, help="Maximum number of leaf nodes"
    )
    parser.add_argument(
        "--plot", default=False, action="store_true", help="Plot progress"
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
    tree = DecisionTree(
        max_depth=args.max_depth,
        min_to_split=args.min_to_split,
        max_leaves=args.max_leaves,
        criterion=args.criterion,
    )
    tree.split(train_data, train_target)
    predictions_train = [
        tree.predict(train_data[i, :]) for i in range(train_data.shape[0])
    ]
    predictions_test = [
        tree.predict(test_data[i, :]) for i in range(test_data.shape[0])
    ]

    # TODO: Create a decision tree on the trainining data.
    #
    # - For each node, predict the most frequent class (and the one with
    # smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split descreasing the criterion
    #   the most. Each split point is an average of two nearest feature values
    #   of the instances corresponding to the given node (i.e., for three instances
    #   with values 1, 7, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be at most `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    print(
        "Train acc: {:.1f}%".format(
            sklearn.metrics.accuracy_score(train_target, predictions_train) * 100
        )
    )
    print(
        "Test acc: {:.1f}%".format(
            sklearn.metrics.accuracy_score(test_target, predictions_test) * 100
        )
    )

