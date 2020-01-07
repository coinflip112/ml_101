#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
import urllib.request
import zipfile

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

from search_spaces import (
    elastic_net_search_space,
    gb_search_space,
    knn_search_space,
    naive_bayes_search_space,
    rf_search_space,
    svm_search_space,
)
from skopt import BayesSearchCV
from stacker import StackingClassifier

if __name__ == "__main__":

    class Dataset:
        def __init__(
            self,
            name="isnt_it_ironic.train.zip",
            url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/",
        ):
            if not os.path.exists(name):
                print("Downloading dataset {}...".format(name), file=sys.stderr)
                urllib.request.urlretrieve(url + name, filename=name)

            # Load the dataset and split it into `data` and `target`.
            self.data = []
            self.target = []

            with zipfile.ZipFile(name, "r") as dataset_file:
                with dataset_file.open(name.replace(".zip", ".txt"), "r") as train_file:
                    for line in train_file:
                        label, text = line.decode("utf-8").rstrip("\n").split("\t")
                        self.data.append(text)
                        self.target.append(int(label))
            self.target = np.array(self.target, np.int32)

    train = Dataset()

    estimator_elastic_net = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    input="content",
                    encoding="utf-8",
                    decode_error="ignore",
                    ngram_range=(1, 3),
                ),
            ),
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
    estimator_svm = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    input="content",
                    encoding="utf-8",
                    decode_error="ignore",
                    ngram_range=(1, 3),
                ),
            ),
            (
                "estimator",
                SGDClassifier(loss="modified_huber", n_jobs=-1, penalty="elasticnet"),
            ),
        ]
    )
    estimator_rf = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    input="content",
                    encoding="utf-8",
                    decode_error="ignore",
                    ngram_range=(1, 3),
                ),
            ),
            ("estimator", RandomForestClassifier(n_jobs=-1)),
        ]
    )
    estimator_gb = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    input="content",
                    encoding="utf-8",
                    decode_error="ignore",
                    ngram_range=(1, 3),
                ),
            ),
            ("estimator", GradientBoostingClassifier()),
        ]
    )

    estimator_knn = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    input="content",
                    encoding="utf-8",
                    decode_error="ignore",
                    ngram_range=(1, 1),
                ),
            ),
            ("estimator", KNeighborsClassifier()),
        ]
    )

    estimator_nb = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    input="content",
                    encoding="utf-8",
                    decode_error="ignore",
                    ngram_range=(1, 3),
                ),
            ),
            ("estimator", MultinomialNB(fit_prior=False)),
        ]
    )
    n_iter_mapping = {"en": 150, "svm": 150, "rf": 150, "gb": 150, "knn": 150, "nb": 150}

    elastic_net_opt = BayesSearchCV(
        estimator=estimator_elastic_net,
        search_spaces=elastic_net_search_space,
        n_iter=n_iter_mapping["en"],
        scoring=make_scorer(f1_score, greater_is_better=True),
        n_jobs=-1,
        verbose=5,
        refit=False,
        cv=5,
    )

    svm_opt = BayesSearchCV(
        estimator=estimator_svm,
        search_spaces=svm_search_space,
        n_iter=n_iter_mapping["svm"],
        scoring=make_scorer(f1_score, greater_is_better=True),
        n_jobs=-1,
        verbose=3,
        refit=False,
        cv=5,
    )

    rf_opt = BayesSearchCV(
        estimator=estimator_rf,
        search_spaces=rf_search_space,
        n_iter=n_iter_mapping["rf"],
        scoring=make_scorer(f1_score, greater_is_better=True),
        n_jobs=-1,
        verbose=3,
        refit=False,
        cv=5,
    )

    gb_opt = BayesSearchCV(
        estimator=estimator_gb,
        search_spaces=gb_search_space,
        n_iter=n_iter_mapping["gb"],
        scoring=make_scorer(f1_score, greater_is_better=True),
        n_jobs=-1,
        verbose=3,
        refit=False,
        cv=5,
    )

    knn_opt = BayesSearchCV(
        estimator=estimator_knn,
        search_spaces=knn_search_space,
        n_iter=n_iter_mapping["knn"],
        scoring=make_scorer(f1_score, greater_is_better=True),
        n_jobs=-1,
        verbose=3,
        refit=False,
        cv=5,
    )

    nb_opt = BayesSearchCV(
        estimator=estimator_nb,
        search_spaces=naive_bayes_search_space,
        n_iter=n_iter_mapping["gb"],
        scoring=make_scorer(f1_score, greater_is_better=True),
        n_jobs=-1,
        verbose=3,
        refit=False,
        cv=5,
    )

    nb_opt.fit(train.data, train.target)
    elastic_net_opt.fit(train.data, train.target)
    svm_opt.fit(train.data, train.target)
    rf_opt.fit(train.data, train.target)
    gb_opt.fit(train.data, train.target)
    knn_opt.fit(train.data, train.target)

    best_params = {
        "en": [elastic_net_opt.best_params_, elastic_net_opt.best_score_],
        "svm": [svm_opt.best_params_, svm_opt.best_score_],
        "nb": [nb_opt.best_params_, nb_opt.best_score_],
        "knn": [knn_opt.best_params_, knn_opt.best_score_],
        "rf": [rf_opt.best_params_, rf_opt.best_score_],
        "gb": [gb_opt.best_params_, gb_opt.best_score_],
    }
    with open("best_params.params", "wb") as params_to_write:
        pickle.dump(best_params, params_to_write)

    final_estimators = [
        estimator_elastic_net.set_params(**elastic_net_opt.best_params_),
        estimator_svm.set_params(**svm_opt.best_params_),
        estimator_nb.set_params(**nb_opt.best_params_),
        estimator_gb.set_params(**gb_opt.best_params_),
        estimator_knn.set_params(**knn_opt.best_params_),
        estimator_rf.set_params(**rf_opt.best_params_),
    ]

    final_model = StackingClassifier(
        estimators=final_estimators,
        estimators_names=["en", "svm", "nb", "gb", "knn", "rf"],
    )
    final_model.fit(train.data, train.target)

    with lzma.open("isnt_it_ironic.model", "wb") as model_file:
        pickle.dump(final_model, model_file)
