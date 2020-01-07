#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
import pandas as pd

import sklearn.metrics


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="human_activity_recognition.model",
    type=str,
    help="Model path",
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)


def recodex_predict(data):
    # The `data` is a pandas.DataFrame containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open("human_activity_model.model", "rb") as model_file:
        model = pickle.load(model_file)
    predictions = model.predict(data)
    return predictions
    # TODO: Return the predictions as a Numpy array.
