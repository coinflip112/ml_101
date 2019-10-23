from skopt.space import Categorical, Integer, Real
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    FunctionTransformer,
    RobustScaler,
)


preprocessing_options_general = {
    "preprocessing__numerics__poly_features__degree": Integer(low=1, high=3),
    "preprocessing__numerics__poly_features__interaction_only": Categorical(
        [True, False]
    ),
    "preprocessing__numerics__scaler": Categorical(
        [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
    ),
}

preprocessing_options_tree_based = {
    "preprocessing__numerics__poly_features__degree": Integer(low=1, high=3),
    "preprocessing__numerics__poly_features__interaction_only": Categorical(
        [True, False]
    ),
}


svc_linear_search_space = {
    "estimator__alpha": Real(1e-6, 1e2, prior="log-uniform"),
    "estimator__l1_ratio": Real(0, 1),
    **preprocessing_options_general,
}

svc_polynomial_search_space = {
    "estimator__C": Real(1e-6, 1e6, prior="log-uniform"),
    "estimator__degree": Integer(2, 5),
    **preprocessing_options_general,
}
svc_rbf_search_space = {
    "estimator__C": Real(1e-6, 1e6, prior="log-uniform"),
    "estimator__gamma": Real(1e-6, 1e1, prior="log-uniform"),
    **preprocessing_options_general,
}

rf_search_space = {
    "estimator__max_depth": Integer(low=10, high=50),
    "estimator__max_features": Real(low=0.3, high=1.0),
    "estimator__min_samples_leaf": Integer(low=2, high=10),
    "estimator__min_samples_split": Integer(low=2, high=5),
    "estimator__n_estimators": Integer(low=40, high=1000),
    "estimator__bootstrap": Categorical([True, False]),
    **preprocessing_options_tree_based,
}

extra_tree_search_space = {
    "estimator__max_depth": Integer(low=10, high=50),
    "estimator__max_features": Real(low=0.3, high=1.0),
    "estimator__min_samples_leaf": Integer(low=2, high=10),
    "estimator__min_samples_split": Integer(low=2, high=5),
    "estimator__n_estimators": Integer(low=40, high=1000),
    "estimator__bootstrap": Categorical([True, False]),
    **preprocessing_options_tree_based,
}


elastic_net_search_space = {
    "estimator__alpha": Real(1e-2, 1e2, prior="log-uniform"),
    "estimator__l1_ratio": Real(0, 1),
    **preprocessing_options_general,
}

gaussian_process_search_space = {**preprocessing_options_general}
naive_bayes_search_space = {**preprocessing_options_general}
knn_search_space = {
    "estimator__n_neighbors": Integer(low=3, high=20),
    "estimator__p": Integer(low=1, high=2),
    **preprocessing_options_general,
}

