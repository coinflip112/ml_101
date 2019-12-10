import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, prior_weight=100):
        self.prior_weight = prior_weight
        self.prior_ = None
        self.posteriors_ = None

    def fit(self, X, y=None):
        X = X.copy()
        self.col_names = X.columns
        X = pd.concat((X, pd.Series(y).rename("y")), axis="columns")
        self.prior_ = y.mean()
        self.posteriors_ = {}

        for col_name in self.col_names:
            agg = X.groupby(col_name)["y"].agg(["count", "mean"])
            counts = agg["count"]
            means = agg["mean"]
            pw = self.prior_weight
            self.posteriors_[col_name] = (
                (pw * self.prior_ + counts * means) / (pw + counts)
            ).to_dict()

        return self

    def transform(self, X, y=None):
        for col_name in self.col_names:
            posteriors = self.posteriors_[col_name]
            X[col_name] = X[col_name].map(posteriors).fillna(self.prior_).astype(float)
        return X
