import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np

import sklearn.metrics


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", default="isnt_it_ironic.model", type=str, help="Model path"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()

    # TODO: Train the model.

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:


def recodex_predict(data):
    # The `data` is a Python list containing tweets as `str`ings.

    args = parser.parse_args([])

    with lzma.open("isnt_it_ironic.model", "rb") as model_file:
        model = pickle.load(model_file)

    predictions = model.predict(data)

    return predictions.astype(np.int8)
    # TODO: Return the predictions as a Python list or Numpy array of
    # binary labels of the tweets.
