#!/usr/bin/env python3
from isnt_it_ironic import Dataset
import argparse
import lzma
import os
import pickle
import sys
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__ == "__main__":
    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4",
        output_shape=[512],
        input_shape=[],
        dtype=tf.string,
    )

    model = tf.keras.Sequential(
        [
            hub_layer,
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(model.summary())
    dataset = Dataset()
    features_train, targets_train = np.array(dataset.data), np.array(dataset.target)
    features_train, targets_train = shuffle(features_train, targets_train)
    features_train, features_val, targets_train, targets_val = train_test_split(
        features_train, targets_train
    )

    model.fit(
        features_train,
        targets_train,
        epochs=1000,
        batch_size=16,
        validation_data=(features_val, targets_val),
        callbacks=[tf.keras.callbacks.TensorBoard()],
    )
    # model.save("isnt_it_ironic_model.h5")

