#!/usr/bin/env python3
# dcfac0e3-1ade-11e8-9de3-00505601122b
# 7d179d73-3e93-11e9-b0fd-00505601122b

import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression


def is_categorical(x):
    """Function which checks whether a value a integer or string/object type suggesting a categorical variable

    Arguments:
        x {Variouse formats} -- Constant which is subject to the is_integer test in the homework context it is a pandas dataframe

    Returns:
        np.bool_ -- Boolean value indicating whether the given argument is a instance of integer or string.
    """
    is_categorical = x.apply(lambda x: isinstance(x, int) or isinstance(x, str)).values
    # CHANGED testing whether a value is categorical is given by whether the columns is an integer/string
    return is_categorical


# return two lists with indexes of numerical and categorical columns
# function doesnt work as expected, integers can be numerics


def index_finder(dataset):
    """Function which indifies the positinal index of columns of categorical and numeric variables in a pandas dataframe
    
    Arguments:
        dataset {pd.DataFrame} -- Input pandas dataframe, features of which are subject ot being classified into either numeric or catgorical
    
    Returns:
        (list, list) --  A touple of lists, with the first containing positional indexes of categorical variables in the dataframe and the others being numeric.
    """
    cat_index = []
    num_index = []
    # CHANGED testing whether a value is categorical is given by whether the columns is an integer/string
    for index in range(dataset.shape[1]):  # number of columns in np array
        test = is_categorical(dataset.iloc[:, index])
        result = all(test)  # if whole array contains True
        if result:
            cat_index.append(index)
            # this works kinda but think about the case when there would be mixed column values
            # this would result into a column being classified as numeric as not all values are categorical
            # however the given column is much more likely to be categorical or in need of additional processing
        else:
            num_index.append(index)
    return (
        cat_index,
        num_index,
    )  # return the indexes of categorical colums and numerical columns


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

    # Note that `train.data` and `train.target` are a `pandas.DataFrame`.
    # It is similar to a Numpy array, but columns have names and types.
    # You can get column types using `train_data.dtypes`; note that
    # strings are actually encoded as `object`s.

    cat_index = features.select_dtypes(
        include=[np.object_]
    ).columns  # this includes all string columns in the dataframe
    num_index = features.select_dtypes(
        include=[np.number]
    ).columns  # this includes all numeric columns in the dataframe
    # either way it is better to select and map the types on a semi-individual basic
    # the same way you would do in R
    # explore a few rows of the data
    # decide what is numeric what is categorical and anything else
    # create a lists of those feature categories (almost manually just copy and paste)

    preprocessing = ColumnTransformer(
        [
            (
                "categoric",
                OneHotEncoder(handle_unknown="ignore", categories="auto", sparse=False),
                cat_index,
            ),
            ("numeric", sklearn.preprocessing.StandardScaler(), num_index),
        ]
    )
    # You had a double pipeline here. Or maybe i added it? Dont remember really Honestly not really most concious:D

    feat_engineering = PolynomialFeatures(2, include_bias=False)

    # you can combine all 3 steps into a single pipeline object
    # This object will represent an estimator which will:
    # 1. preprocess the features,
    # 2. create 2nd degree polynomial features
    #   (beware that ur doing the second power of one-hot-encoding) and therefore ur model will be overparametrized
    #   (hint apply Polynomial features only to the numerical features)
    #  finally fit a classifeir (try selecting a classifer which can handle non-linear relationships)

    classifier = LogisticRegression(solver="liblinear", C=10, max_iter=10000)

    estimator = Pipeline(
        steps=[
            ("feat_preprocessing", preprocessing),
            ("feat_engineering", feat_engineering),
            ("classifier", classifier),
        ]
    )
    estimator.fit(features, targets)

    # with lzma.open(args.model_path, "wb") as model_file:
    #     pickle.dump(estimator, model_file)

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we can also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    # with lzma.open(args.model_path, "wb") as model_file:
    #       pickle.dump(model, model_file)

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).


def recodex_predict(data):
    # The `data` is a hopefully pandas dataset
    with lzma.open("binary_classification_competition.model", "rb") as model_file:
        model = pickle.load(model_file)

    predictions = model.predict(data)
    return predictions
