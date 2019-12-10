#!/usr/bin/env python3
import argparse
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet

preprocessing = lambda features_mapping_dict: ColumnTransformer(
    transformers=[
        ("numericals", StandardScaler(), features_mapping_dict["numerics"]),
        (
            "binary_features",
            FunctionTransformer(validate=False),
            features_mapping_dict["binary"],
        ),
        (
            "categorical_features",
            OneHotEncoder(handle_unknown="ignore", categories="auto", sparse=False),
            features_mapping_dict["categoricals"],
        ),
    ]
)

binary_features = [1, 4]
categorical_features = [0, 2, 3, 5, 6, 7]
numerical_features = [8, 9, 10, 11]

# create variable mapping dictionary
features_mapping_dict = {
    "numerics": numerical_features,
    "binary": binary_features,
    "categoricals": categorical_features,
}

best_params = {
    "elastic_net": {"alpha": 0.013805212784019607, "l1_ratio": 0.9135450867220719},
    "gb": {
        "learning_rate": 0.051561333292395484,
        "max_depth": 3,
        "n_estimators": 996,
        "subsample": 0.709693194180173,
    },
    "svr": {
        "C": 14746.443911967926,
        "degree": 4,
        "gamma": 0.024061171791590563,
        "kernel": "rbf",
    },
    "rf": {
        "bootstrap": False,
        "max_depth": 31,
        "max_features": "log2",
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "n_estimators": 98,
    },
}
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="linear_regression_competition.model",
    type=str,
    help="Model path",
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")


def fit_best_model(
    feature_mapping=features_mapping_dict, best_params=best_params, save_to_disk=True
):
    # load data
    data = np.load("linear_regression_competition.train.npz")
    features, targets = data["data"], data["target"]

    # define models
    estimator_svr = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", SVR(**best_params["svr"])),
        ]
    )
    estimator_rf = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", RandomForestRegressor(**best_params["rf"])),
        ]
    )
    estimator_gb = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", GradientBoostingRegressor(**best_params["gb"])),
        ]
    )
    estimator_elastic_net = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", ElasticNet(**best_params["elastic_net"])),
        ]
    )

    voter = VotingRegressor(
        estimators=[
            ("gb", estimator_gb),
            ("rf", estimator_rf),
            ("lr", estimator_elastic_net),
            ("svr", estimator_svr),
        ]
    )

    voter.fit(features, targets)

    with open("linear_regression_competition.model", "wb") as model_file:
        pickle.dump(voter, model_file)


def recodex_predict(data):
    # The `data` is a Numpy array containt test se

    with open("linear_regression_competition.model", "rb") as model_file:
        model = pickle.load(model_file)

    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)

