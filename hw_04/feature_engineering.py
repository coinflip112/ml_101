#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd

import sklearn.datasets
import sklearn.model_selection

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
    PolynomialFeatures,
)


def get_feature_types(dataset):
    dataset = pd.DataFrame(dataset)
    dataset = dataset.apply(
        lambda x: pd.to_numeric(x, downcast="integer"), axis="index"
    )
    dtypes = dataset.dtypes.astype(np.str_)
    numerics = [
        index for index in range(dtypes.shape[0]) if "float" in dtypes.iloc[index]
    ]
    categoricals = [
        index for index in range(dtypes.shape[0]) if "int" in dtypes.iloc[index]
    ]
    return numerics, categoricals


preprocessing = lambda features_mapping_dict: ColumnTransformer(
    transformers=[
        (
            "categorical_features",
            OneHotEncoder(handle_unknown="ignore", categories="auto", sparse=False),
            features_mapping_dict["categoricals"],
        ),
        ("numericals", StandardScaler(), features_mapping_dict["numerics"]),
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="boston", type=str, help="Standard sklearn dataset to load"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--test_ratio", default=0.5, type=float, help="Test set size ratio"
    )
    args = parser.parse_args()

    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO(linear_regression_l2): Split the dataset randomly to train
    # and test using `sklearn.model_selection.train_test_split`, with
    # `test_size=args.test_ratio` and `random_state=args.seed`.
    X = dataset.data
    y = dataset.target

    numerics, categoricals = get_feature_types(X)
    features_mapping_dict = {"numerics": numerics, "categoricals": categoricals}

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=args.test_ratio, random_state=args.seed
    )

    preprocessing_pipeline = preprocessing(features_mapping_dict)
    feat_gen_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing_pipeline),
            ("poly_feat_gen", PolynomialFeatures(2, include_bias=False)),
        ]
    )

    X_train = feat_gen_pipeline.fit_transform(X_train)
    X_test = feat_gen_pipeline.transform(X_test)

    with open("feature_engineering.out", "w") as output_file:
        for data in [X_train, X_test]:
            for line in range(5):
                print(
                    " ".join(
                        "{:.6g}".format(data[line, column])
                        for column in range(data.shape[1])
                    ),
                    file=output_file,
                )
