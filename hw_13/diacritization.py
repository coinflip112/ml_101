#!/usr/bin/env python3
# dcfac0e3-1ade-11e8-9de3-00505601122b
# dce9cf60-42b6-11e9-b0fd-00505601122b
# 7d179d73-3e93-11e9-b0fd-00505601122b
import argparse
import lzma
import os
import pickle
import string
import sys
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

flatten = lambda l: [item for sublist in l for item in sublist]


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(
        LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper()
    )

    def __init__(
        self,
        name="fiction-train.txt",
        url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/",
    ):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(
                url + name.replace(".txt", ".LICENSE"),
                filename=name.replace(".txt", ".LICENSE"),
            )

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.data = dataset_file.read()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", default="diacritization.model", type=str, help="Model path"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()
    # Set random seed
    np.random.seed(args.seed)

    # train = Dataset().data

    # characters = list(np.unique(list(remove_accents(train.lower())))[1:]) + ["#"]

    # sentences_train = train.split("\n")

    # input_len = 13
    # features = np.array(
    #     flatten(
    #         [
    #             list(
    #                 find_ngrams(
    #                     "".join(["#" for i in range(int((input_len - 1) / 2))])
    #                     + remove_accents(sentence.lower())
    #                     + "".join(["#" for i in range(int((input_len - 1) / 2))]),
    #                     input_len,
    #                 )
    #             )
    #             for sentence in sentences_train
    #         ]
    #     )
    # )
    # targets = [
    #     feature[int((input_len - 1) / 2)]
    #     for feature in flatten(
    #         [
    #             list(
    #                 find_ngrams(
    #                     "".join(["#" for i in range(int((input_len - 1) / 2))])
    #                     + sentence.lower()
    #                     + "".join(["#" for i in range(int((input_len - 1) / 2))]),
    #                     input_len,
    #                 )
    #             )
    #             for sentence in sentences_train
    #         ]
    #     )
    # ]
    # targets = np.array(targets)

    # le = LabelEncoder()
    # targets = le.fit_transform(targets)
    # ohe = OneHotEncoder(
    #     handle_unknown="ignore",
    #     categories=[sorted(characters) for i in range(input_len)],
    # )
    # features = ohe.fit_transform(features)

    # nn = MLPClassifier(
    #     hidden_layer_sizes=(250, 225, 200, 175, 150, 125, 100, 75, 50, 30),
    #     verbose=4,
    #     early_stopping=True,
    #     validation_fraction=0.05,
    #     max_iter=400,
    #     n_iter_no_change=5,
    # )

    # nn.fit(features, targets)


def recodex_predict(data):
    # The `data` is a `str` containing text without diacritics

    args = parser.parse_args([])
    with lzma.open("diacritization.model", "rb") as model_file:
        nn = pickle.load(model_file)

    with lzma.open("onehot.encoder", "rb") as model_file:
        ohe = pickle.load(model_file)

    with lzma.open("label.encoder", "rb") as model_file:
        le = pickle.load(model_file)

    def find_ngrams(s, n):
        return zip(*[s[i:] for i in range(n)])

    input_len = 13
    features = np.concatenate(
        [
            np.array(
                list(
                    find_ngrams(
                        "".join(["#" for i in range(int((input_len - 1) / 2))])
                        + sentence.lower()
                        + "".join(["#" for i in range(int((input_len - 1) / 2))]),
                        input_len,
                    )
                )
            )
            for sentence in data.split("\n")[:-1]
        ]
    )

    def convert_predictions(predictions, orig_text):
        to_diacritize = [
            "a",
            "c",
            "d",
            "e",
            "i",
            "n",
            "o",
            "r",
            "s",
            "t",
            "u",
            "y",
            "z",
            "A",
            "C",
            "D",
            "E",
            "I",
            "N",
            "O",
            "R",
            "S",
            "T",
            "U",
            "Y",
            "Z",
        ]
        predictions = "".join(
            np.where(
                (
                    (np.array(list(predictions)) == " ")
                    & (np.array(list(orig_text)) != " ")
                ),
                np.array(list(orig_text)),
                np.array(list(predictions)),
            )
        )
        predictions = "".join(
            np.where(
                (
                    (np.array(list(predictions)) != " ")
                    & (np.array(list(orig_text)) == " ")
                ),
                np.array(list(orig_text)),
                np.array(list(predictions)),
            )
        )
        predictions = "".join(
            np.where(
                np.isin(np.array(list(orig_text)), to_diacritize),
                np.array(list(predictions)),
                np.array(list(orig_text)),
            )
        )
        result = [
            predicted_char.upper() if orig_char.isupper() else predicted_char
            for predicted_char, orig_char in zip(predictions, orig_text)
        ]
        return "".join(result)

    def add_newlines(predictions, data):
        new_predictions = ""
        data_index = 0
        pred_index = 0
        while pred_index < len(predictions):
            if data[data_index] == "\n":
                new_predictions += "\n"
                data_index += 1
            new_predictions += predictions[pred_index]
            pred_index += 1
            data_index += 1
        return new_predictions + "\n"

    orig_text = "".join(data.split("\n"))

    features = ohe.transform(features)
    predictions = nn.predict(features)
    predictions = le.inverse_transform(predictions)
    predictions = "".join(predictions)

    predictions = convert_predictions(predictions, orig_text)

    result = add_newlines(predictions, data)

    return result
