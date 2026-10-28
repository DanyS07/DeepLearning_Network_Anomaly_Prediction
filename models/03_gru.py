"""
GRU model. Same as the LSTM, with a GRU layer (fewer parameters).

Run:  python models/03_gru.py
"""

import _setup_path  # noqa: F401
import tensorflow as tf
from tensorflow.keras import layers

from common.train_utils import run_cv, keras_fit


def build(input_shape, n_classes):
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="gru")


if __name__ == "__main__":
    run_cv("gru", keras_fit(build))
