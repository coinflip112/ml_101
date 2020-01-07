#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
import urllib.request
import zipfile
import pandas as pd
import numpy as np

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.metrics import f1_score, get_scorer, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from search_spaces import elastic_net_search_space, gb_search_space, rf_search_space
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from skopt import BayesSearchCV
from stacker import StackingClassifier


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(
        self,
        name="human_activity_recognition.train.csv.xz",
        url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/",
    ):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `train_target` (column "class")
        # and `train_data` (all other columns).
        dataset = pd.read_csv(name)
        self.data = dataset.drop("class", axis=1)
        self.target = np.array(
            [Dataset.CLASSES.index(target) for target in dataset["class"]], np.int32
        )


if __name__ == "__main__":
    train = Dataset()

    estimator_elastic_net = Pipeline(
        steps=[
            (
                "poly_features",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=True),
            ),
            ("preprocessing", MinMaxScaler()),
            (
                "estimator",
                SGDClassifier(
                    loss="log",
                    max_iter=100000,
                    n_jobs=-1,
                    penalty="elasticnet",
                    verbose=False,
                ),
            ),
        ],
        verbose=False,
    )
    estimator_rf = Pipeline(
        steps=[
            ("preprocessing", MinMaxScaler()),
            ("estimator", RandomForestClassifier(n_jobs=-1)),
        ]
    )
    estimator_gb = Pipeline(
        steps=[
            ("preprocessing", MinMaxScaler()),
            ("estimator", GradientBoostingClassifier()),
        ]
    )

    n_iter_mapping = {"en": 1, "rf": 1, "gb": 1}

    elastic_net_opt = BayesSearchCV(
        estimator=estimator_elastic_net,
        search_spaces=elastic_net_search_space,
        n_iter=n_iter_mapping["en"],
        n_jobs=-1,
        verbose=5,
        refit=False,
        cv=4,
    )

    rf_opt = BayesSearchCV(
        estimator=estimator_rf,
        search_spaces=rf_search_space,
        n_iter=n_iter_mapping["rf"],
        n_jobs=-1,
        verbose=3,
        refit=False,
        cv=4,
    )

    gb_opt = BayesSearchCV(
        estimator=estimator_gb,
        search_spaces=gb_search_space,
        n_iter=n_iter_mapping["gb"],
        n_jobs=-1,
        verbose=3,
        refit=False,
        cv=4,
    )

    nn = MLPClassifier(
        hidden_layer_sizes=(150, 125, 100, 75, 50, 30),
        verbose=4,
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=400,
        n_iter_no_change=10,
    )

    elastic_net_opt.fit(train.data, train.target)
    rf_opt.fit(train.data, train.target)
    gb_opt.fit(train.data, train.target)

    best_params = {
        "en": [elastic_net_opt.best_params_, elastic_net_opt.best_score_],
        "rf": [rf_opt.best_params_, rf_opt.best_score_],
        "gb": [gb_opt.best_params_, gb_opt.best_score_],
    }
    with open("best_params.params", "wb") as params_to_write:
        pickle.dump(best_params, params_to_write)

    final_estimators = [
        estimator_elastic_net.set_params(**elastic_net_opt.best_params_),
        estimator_gb.set_params(**gb_opt.best_params_),
        estimator_rf.set_params(**rf_opt.best_params_),
        nn,
    ]

    final_model = StackingClassifier(
        estimators=final_estimators, estimators_names=["en", "gb", "rf", "nn"]
    )
    final_model.fit(train.data, train.target)

    with lzma.open("human_activity_model.model", "wb") as model_file:
        pickle.dump(final_model, model_file)
