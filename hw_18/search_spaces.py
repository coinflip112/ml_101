from skopt.space import Categorical, Integer, Real

vectorizer_search_space = {
    "vectorizer__strip_accents": Categorical(["unicode", None]),
    "vectorizer__lowercase": Categorical([True, False]),
    "vectorizer__stop_words": Categorical(["english", None]),
    "vectorizer__norm": Categorical(["l2", "l1"]),
}
rf_search_space = {
    **vectorizer_search_space,
    "estimator__max_depth": Integer(low=10, high=50),
    "estimator__max_features": Real(low=0.3, high=1.0),
    "estimator__min_samples_leaf": Integer(low=2, high=10),
    "estimator__min_samples_split": Integer(low=2, high=5),
    "estimator__n_estimators": Integer(low=40, high=1000),
    "estimator__bootstrap": Categorical([True, False]),
}

gb_search_space = {
    **vectorizer_search_space,
    "estimator__n_estimators": Integer(low=40, high=200),
    "estimator__learning_rate": Real(low=0.025, high=0.5, prior="log-uniform"),
    "estimator__max_depth": Integer(low=2, high=15),
    "estimator__subsample": Real(low=0.5, high=1),
}
elastic_net_search_space = {
    **vectorizer_search_space,
    "estimator__alpha": Real(1e-2, 1e2, prior="log-uniform"),
    "estimator__l1_ratio": Real(0.0, 1.0),
}
svm_search_space = {
    **vectorizer_search_space,
    "estimator__alpha": Real(1e-2, 1e2, prior="log-uniform"),
    "estimator__l1_ratio": Real(0.0, 1.0),
}
knn_search_space = {
    **vectorizer_search_space,
    "estimator__n_neighbors": Integer(low=3, high=20),
    "estimator__p": Integer(low=1, high=2),
}
naive_bayes_search_space = {**vectorizer_search_space}

