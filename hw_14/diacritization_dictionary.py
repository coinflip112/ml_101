#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import string
import sys
import unicodedata
import urllib.request
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
flatten = lambda l: [item for sublist in l for item in sublist]


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


class Dictionary:
    def __init__(
        self,
        name="fiction-dictionary.txt",
        url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/",
    ):
        if not os.path.exists(name):
            print("Downloading {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(
                url + name.replace(".txt", ".LICENSE"),
                filename=name.replace(".txt", ".LICENSE"),
            )

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants


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

    with lzma.open("diacritization_new.model", "rb") as model_file:
        nn = pickle.load(model_file)

    with lzma.open("onehot_new.encoder", "rb") as model_file:
        ohe = pickle.load(model_file)

    sentences = data.split("\n")

    def find_ngrams(s, n):
        return zip(*[s[i:] for i in range(n)])

    diac_source_target_combo = namedtuple("diac_combo", ["source", "target"])
    diacritization_mapping = {
        diac_source_target_combo("a", "á"): "acute",
        diac_source_target_combo("c", "č"): "caron",
        diac_source_target_combo("d", "ď"): "caron",
        diac_source_target_combo("e", "ě"): "caron",
        diac_source_target_combo("e", "é"): "caron",
        diac_source_target_combo("i", "í"): "acute",
        diac_source_target_combo("n", "ň"): "caron",
        diac_source_target_combo("o", "ó"): "acute",
        diac_source_target_combo("r", "ř"): "caron",
        diac_source_target_combo("s", "š"): "caron",
        diac_source_target_combo("t", "ť"): "caron",
        diac_source_target_combo("u", "ů"): "ring",
        diac_source_target_combo("u", "ú"): "acute",
        diac_source_target_combo("y", "ý"): "acute",
        diac_source_target_combo("z", "ž"): "caron",
    }
    input_len = 13
    variants = Dictionary().variants
    sentences = data.split("\n")
    sentence_features = np.array(
        [
            list(
                find_ngrams(
                    "".join(["#" for i in range(int((input_len - 1) / 2))])
                    + remove_accents(sentence.lower())
                    + "".join(["#" for i in range(int((input_len - 1) / 2))]),
                    input_len,
                )
            )
            for sentence in sentences
        ]
    )

    def word_predict(word_features, orig_word):
        if orig_word in variants:
            word_candidates = variants[orig_word]
            orig_word = orig_word.lower()
        else:
            return orig_word
        if len(word_candidates) == 1:
            return word_candidates[0]
        changes = []
        for word in word_candidates:
            character_changes = []
            for char in list(word):
                char_pair = diac_source_target_combo(
                    source=str.translate(char, Dataset.DIA_TO_NODIA), target=char
                )
                if char_pair in diacritization_mapping:
                    character_changes.append(diacritization_mapping[char_pair])
                else:
                    character_changes.append("no_change")
            changes.append(character_changes)
        word_features = ohe.transform(word_features)
        predicted_proba = [
            dict(zip(["acute", "caron", "no_change", "ring"], predicted_prob))
            for predicted_prob in nn.predict_proba(word_features)
        ]
        candidate_probas = []
        for word_candidate_change in changes:
            single_proba = 1
            for char_change, proba_dict in zip(word_candidate_change, predicted_proba):
                single_proba *= proba_dict[char_change]
            candidate_probas.append(single_proba)
        return word_candidates[np.argmax(candidate_probas)]

    def sentence_predict(orig_sentence_features, orig_sentence):
        space_indexes = np.where(np.array(list(orig_sentence)) == " ")
        space_indexes = np.concatenate(
            [[-1], space_indexes[0], [len(orig_sentence) - 1]]
        )
        word_features = [
            orig_sentence_features[space_indexes[i - 1] + 1 : space_indexes[i]]
            for space_count, i in enumerate(range(1, len(space_indexes)))
        ]
        orig_words = orig_sentence.split(" ")
        predicted_words = [
            word_predict(word_feature, orig_word)
            for (word_feature, orig_word) in zip(word_features, orig_words)
        ]
        return " ".join(predicted_words)

    predicted_sentences = [
        sentence_predict(sentence_feature, orig_sentence)
        for sentence_feature, orig_sentence in zip(sentence_features, sentences)
    ]

    predictions = "\n".join(predicted_sentences)
    return predictions
