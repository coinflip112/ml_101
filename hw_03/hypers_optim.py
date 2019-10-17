import pandas as pd
import numpy as np
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

if __name__ == "__main__":
    binary_features = [1, 4]
    categorical_features = [0, 2, 3, 5, 6, 7]
    numerical_features = [8, 9, 10, 11]

    data = np.load("linear_regression_competition.train.npz")
    features, targets = data["data"], data["target"]

    preprocessing = lambda: ColumnTransformer(
        transformers=[
            ("numericals", StandardScaler(), numerical_features),
            ("binary_features", FunctionTransformer(validate=False), binary_features),
            (
                "categorical_features",
                OneHotEncoder(handle_unknown="ignore", categories="auto", sparse=False),
                categorical_features,
            ),
        ]
    )

    estimator_svr = Pipeline(
        steps=[("preprocessing", preprocessing()), ("estimator", SVR())]
    )
    estimator_rf = Pipeline(
        steps=[
            ("preprocessing", preprocessing()),
            ("estimator", RandomForestRegressor()),
        ]
    )
    estimator_gb = Pipeline(
        steps=[
            ("preprocessing", preprocessing()),
            ("estimator", GradientBoostingRegressor()),
        ]
    )
    estimator_elastic_net = Pipeline(
        steps=[
            ("preprocessing", preprocessing()),
            ("estimator", ElasticNet(max_iter=10000)),
        ]
    )

    svm_search_space = {
        "estimator__C": Real(1e-6, 1e6, prior="log-uniform"),
        "estimator__gamma": Real(1e-6, 1e1, prior="log-uniform"),
        "estimator__degree": Integer(1, 8),
        "estimator__kernel": Categorical(["linear", "poly", "rbf"]),
    }
    rf_search_space = {
        "estimator__max_depth": Integer(low=10, high=50),
        "estimator__max_features": Categorical(["log2", "sqrt", None]),
        "estimator__min_samples_leaf": Integer(low=2, high=10),
        "estimator__min_samples_split": Integer(low=2, high=4),
        "estimator__n_estimators": Integer(low=40, high=200),
        "estimator__bootstrap": Categorical([True, False]),
    }

    gb_search_space = {
        "estimator__n_estimators": Integer(low=100, high=1000),
        "estimator__learning_rate": Real(low=0.025, high=0.5, prior="log-uniform"),
        "estimator__max_depth": Integer(low=2, high=15),
        "estimator__subsample": Real(low=0.5, high=1),
    }
    elastic_net_search_space = {
        "estimator__alpha": Real(1e-2, 1e2, prior="log-uniform"),
        "estimator__l1_ratio": Real(0, 1),
    }

    elastic_net_opt = BayesSearchCV(
        estimator=estimator_elastic_net,
        search_spaces=elastic_net_search_space,
        n_iter=400,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        n_jobs=-1,
        verbose=3,
        refit=False,
    )

    svm_opt = BayesSearchCV(
        estimator=estimator_svr,
        search_spaces=svm_search_space,
        n_iter=400,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        n_jobs=-1,
        verbose=3,
        refit=False,
    )

    rf_opt = BayesSearchCV(
        estimator=estimator_rf,
        search_spaces=rf_search_space,
        n_iter=400,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        n_jobs=-1,
        verbose=3,
        refit=False,
    )

    gb_opt = BayesSearchCV(
        estimator=estimator_gb,
        search_spaces=gb_search_space,
        n_iter=400,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        n_jobs=-1,
        verbose=3,
        refit=False,
    )

    elastic_net_opt.fit(features, targets)
    gb_opt.fit(features, targets)
    svm_opt.fit(features, targets)
    rf_opt.fit(features, targets)

    best_params = {
        "elastic_net": elastic_net_opt.best_params_,
        "gb": gb_opt.best_params_,
        "svr": svm_opt.best_params_,
        "rf": rf_opt.best_params_,
    }
    with open("best_params.json", "w") as json_to_write:
        json.dump(best_params, json_to_write)
