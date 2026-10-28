"""
LSTM model.  Input: 30 s x 40 features.  Output: 7-class probabilities
("which anomaly starts in the next 10 s", or normal).

Run:  python models/02_lstm.py
"""

import _setup_path  # noqa: F401
import tensorflow as tf
from tensorflow.keras import layers

from common.train_utils import run_cv, keras_fit


def build(input_shape, n_classes):
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="lstm")


if __name__ == "__main__":
    run_cv("lstm", keras_fit(build))
