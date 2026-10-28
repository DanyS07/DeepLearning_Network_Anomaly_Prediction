"""
TCN (Temporal Convolutional Network).

Stack of causal 1-D convolutions with dilations 1, 2, 4, 8 and residual
connections. Kernel 3 -> receptive field 1 + 2 * (1+2+4+8) = 31 s, which
covers the whole 30 s window. "Causal" means each time step only sees the
past, like the recurrent models. The output at the last time step (the
anchor second) is used for the prediction.

Run:  python models/04_tcn.py
"""

import _setup_path  # noqa: F401
import tensorflow as tf
from tensorflow.keras import layers

from common.train_utils import run_cv, keras_fit

FILTERS = 64
KERNEL = 3
DILATIONS = [1, 2, 4, 8]


def residual_block(x, dilation):
    y = layers.Conv1D(FILTERS, KERNEL, padding="causal",
                      dilation_rate=dilation, activation="relu")(x)
    y = layers.Dropout(0.2)(y)
    y = layers.Conv1D(FILTERS, KERNEL, padding="causal",
                      dilation_rate=dilation, activation="relu")(y)
    y = layers.Dropout(0.2)(y)
    if x.shape[-1] != FILTERS:
        x = layers.Conv1D(FILTERS, 1)(x)       # match channels for the skip
    return layers.Activation("relu")(layers.Add()([x, y]))


def build(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = inp
    for d in DILATIONS:
        x = residual_block(x, d)
    # keep only the last (anchor) time step
    x = layers.Cropping1D((input_shape[0] - 1, 0))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="tcn")


if __name__ == "__main__":
    run_cv("tcn", keras_fit(build))
