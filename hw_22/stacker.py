import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LogisticRegressionCV


class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators=[], estimators_names=[]):
        self.estimators = estimators
        self.estimators_names = estimators_names
        self.fitted = False

    def fit(self, X, y=None):
        estimators_count = len(self.estimators)

        lv_1_features = np.column_stack(
            tup=[
                cross_val_predict(
                    estimator, X, y, cv=5, method="predict_proba", n_jobs=4
                )[:, 1:]
                for estimator, estimator_name in zip(
                    self.estimators, self.estimators_names
                )
            ]
        )

        stacker = LogisticRegressionCV(cv=5, max_iter=1000, multi_class="multinomial")

        stacker.fit(X=lv_1_features, y=y)
        self.stacker = stacker

        for estimator_index, estimator_name in zip(
            range(estimators_count), self.estimators_names
        ):

            self.estimators[estimator_index].fit(X=X, y=y)

        self.naive_threshold = 0.5
        self.fitted = True
        return self

    def predict_proba(self, X, y=None):
        if not self.fitted:
            raise RuntimeError(
                "You must train the Stacker before attempting to make predictions"
            )

        lv_1_features = np.column_stack(
            tup=[estimator.predict_proba(X)[:, 1:] for estimator in self.estimators]
        )

        predictions = self.stacker.predict_proba(lv_1_features)[:, 1]

        return predictions

    def predict(self, X, y=None):
        if not self.fitted:
            raise RuntimeError(
                "You must train the Stacker before attempting to make predictions"
            )
        lv_1_features = np.column_stack(
            tup=[estimator.predict_proba(X)[:, 1:] for estimator in self.estimators]
        )

        predictions = self.stacker.predict(lv_1_features)
        return predictions
