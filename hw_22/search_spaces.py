from skopt.space import Categorical, Integer, Real

rf_search_space = {
    "estimator__max_depth": Integer(low=10, high=15),
    "estimator__max_features": Real(low=0.1, high=1.0),
    "estimator__min_samples_leaf": Integer(low=2, high=10),
    "estimator__min_samples_split": Integer(low=2, high=5),
    "estimator__n_estimators": Integer(low=40, high=200),
    "estimator__bootstrap": Categorical([True, False]),
}

gb_search_space = {
    "estimator__n_estimators": Integer(low=40, high=150),
    "estimator__learning_rate": Real(low=0.025, high=0.5, prior="log-uniform"),
    "estimator__max_depth": Integer(low=2, high=15),
    "estimator__subsample": Real(low=0.5, high=1),
    "estimator__max_features": Real(low=0.1, high=1.0),
    "estimator__min_samples_leaf": Integer(low=2, high=10),
    "estimator__min_samples_split": Integer(low=2, high=5),
}
elastic_net_search_space = {
    "estimator__alpha": Real(1e-2, 1e2, prior="log-uniform"),
    "estimator__l1_ratio": Real(0.0, 1.0),
}

