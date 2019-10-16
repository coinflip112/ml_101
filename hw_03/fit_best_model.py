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

def fit_best_model(feature_mapping, best_params,save_to_disk = True)
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
    
    with open(args.model_path, "wb") as model_file:
        pickle.dump(voter, model_file)

    
