#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

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
    OneHotEncoder,
)
from stacker import StackingClassifier
from target_encoder import TargetEncoder

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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="binary_classification_competition.model",
    type=str,
    help="Model path",
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
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
                SGDClassifier(
                    loss="modified_huber", penalty="elasticnet", max_iter=30000
                ),
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

    with open("best_params.params", "rb") as params_to_read:
        best_params = pickle.load(params_to_read)

    estimator_svc_linear.set_params(**best_params["svc_linear"][0])
    estimator_extra_trees.set_params(**best_params["extra_tree"][0])
    estimator_elastic_net.set_params(**best_params["elastic_net"][0])
    estimator_naive_bayes.set_params(**best_params["naive_bayes"][0])
    estimator_knn.set_params(**best_params["knn"][0])
    estimators = [
        estimator_svc_linear,
        estimator_extra_trees,
        estimator_elastic_net,
        estimator_naive_bayes,
        estimator_knn,
    ]
    estimator_names = ["svc_linear", "extra_trees", "elastic_net", "naive_bayes", "knn"]
    model = StackingClassifier(estimators=estimators, estimators_names=estimator_names)
    # model.fit(x_train, y_train)

def recodex_predict(data):
    # The `data` is a pandas.DataFrame containt test set input.

    args = parser.parse_args([])

    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)
    test_predictions = model.predict(data)
    return test_predictions
    # TODO: Return the predictions as a Numpy array.
