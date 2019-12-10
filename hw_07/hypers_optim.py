import argparse
import json
import lzma
import os
import pickle
import sys
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
    PolynomialFeatures,
    FunctionTransformer,
)
from target_encoder import TargetEncoder
from search_spaces import (
    svc_linear_search_space,
    svc_polynomial_search_space,
    svc_rbf_search_space,
    rf_search_space,
    elastic_net_search_space,
    gaussian_process_search_space,
    knn_search_space,
    naive_bayes_search_space,
    extra_tree_search_space,
)
from skopt import BayesSearchCV

from skopt.space import Categorical, Integer, Real


class Dataset:
    def __init__(
        self,
        name="binary_classification_competition.train.csv.xz",
        url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/",
    ):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `train_target` (column Target)
        # and `train_data` (all other columns).
        dataset = pd.read_csv(name)
        self.data, self.target = dataset.drop("Target", axis=1), dataset["Target"]


if __name__ == "__main__":
    train = Dataset()
    features = train.data
    targets = train.target
    categoricals = [
        "Workclass",
        "Education",
        "Marital-status",
        "Occupation",
        "Relationship",
        "Race",
        "Native-country",
        "Sex",
    ]
    numerics = [
        "Age",
        "Fnlwgt",
        "Education-num",
        "Capital-gain",
        "Capital-loss",
        "Hours-per-week",
    ]
    features_mapping_dict = {"categoricals": categoricals, "numerics": numerics}

    numerics_pipeline = lambda: Pipeline(
        steps=[
            ("poly_features", PolynomialFeatures(include_bias=False)),
            ("scaler", FunctionTransformer(validate=False)),
        ]
    )

    preprocessing = lambda features_mapping_dict: ColumnTransformer(
        transformers=[
            ("numerics", numerics_pipeline(), features_mapping_dict["numerics"]),
            ("categoricals", TargetEncoder(), features_mapping_dict["categoricals"]),
        ]
    )

    estimator_svc_linear = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            (
                "estimator",
                SGDClassifier(loss="hinge", penalty="elasticnet", max_iter=30000),
            ),
        ]
    )
    estimator_extra_trees = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", ExtraTreesClassifier()),
        ]
    )
    estimator_rf = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", RandomForestClassifier()),
        ]
    )
    estimator_elastic_net = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            (
                "estimator",
                SGDClassifier(max_iter=30000, penalty="elasticnet", loss="log"),
            ),
        ]
    )

    estimator_naive_bayes = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", GaussianNB()),
        ]
    )
    estimator_knn = Pipeline(
        steps=[
            ("preprocessing", preprocessing(features_mapping_dict)),
            ("estimator", KNeighborsClassifier()),
        ]
    )

    naive_bayes_opt = BayesSearchCV(
        cv=4,
        estimator=estimator_naive_bayes,
        search_spaces=naive_bayes_search_space,
        n_iter=100,
        n_jobs=-1,
        refit=False,
        verbose=3,
    )
    knn_opt = BayesSearchCV(
        cv=4,
        estimator=estimator_knn,
        search_spaces=knn_search_space,
        n_iter=60,
        n_jobs=-1,
        refit=False,
        verbose=3,
    )

    svc_linear_opt = BayesSearchCV(
        cv=4,
        estimator=estimator_svc_linear,
        search_spaces=svc_linear_search_space,
        n_iter=100,
        n_jobs=-1,
        refit=False,
        verbose=3,
    )
    extra_tree_opt = BayesSearchCV(
        cv=4,
        estimator=estimator_extra_trees,
        search_spaces=extra_tree_search_space,
        n_iter=100,
        n_jobs=-1,
        refit=False,
        verbose=3,
    )

    rf_opt = BayesSearchCV(
        cv=4,
        estimator=estimator_rf,
        search_spaces=rf_search_space,
        n_iter=100,
        n_jobs=-1,
        refit=False,
        verbose=3,
    )

    elastic_net_opt = BayesSearchCV(
        cv=4,
        estimator=estimator_elastic_net,
        search_spaces=elastic_net_search_space,
        n_iter=80,
        n_jobs=-1,
        refit=False,
        verbose=3,
    )

    naive_bayes_opt.fit(features, targets)
    knn_opt.fit(features, targets)
    svc_linear_opt.fit(features, targets)
    extra_tree_opt.fit(features, targets)
    rf_opt.fit(features, targets)
    elastic_net_opt.fit(features, targets)

    best_params = {
        "naive_bayes": [naive_bayes_opt.best_params_, naive_bayes_opt.best_score_],
        "knn": [knn_opt.best_params_, knn_opt.best_score_],
        "svc_linear": [svc_linear_opt.best_params_, svc_linear_opt.best_score_],
        "extra_tree": [extra_tree_opt.best_params_, extra_tree_opt.best_score_],
        "rf": [rf_opt.best_params_, rf_opt.best_score_],
        "elastic_net": [elastic_net_opt.best_params_, elastic_net_opt.best_score_],
    }
    with open("best_params.params", "wb") as params_to_write:
        pickle.dump(best_params, params_to_write)
