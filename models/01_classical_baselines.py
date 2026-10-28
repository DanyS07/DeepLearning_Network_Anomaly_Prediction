"""
Classical baselines: Logistic Regression and Random Forest.

They cannot read a sequence, so each 30 s window is summarised into
3 numbers per feature:
    last   - value at the anchor second
    mean   - average over the window
    trend  - last minus first (rising / falling over 30 s)
-> 40 features x 3 = 120 inputs.

These are the scores the deep models must beat. If an LSTM is not clearly
better than a Random Forest on the same windows, that is a finding to
report, not a bug.

Run:  python models/01_classical_baselines.py
"""

import _setup_path  # noqa: F401
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from common.train_utils import run_cv, class_weights, MODEL_DIR


def summarise(X):
    return np.concatenate([X[:, -1, :], X.mean(axis=1),
                           X[:, -1, :] - X[:, 0, :]], axis=1)


def sklearn_fit(make_model, name):
    def fit(d, tag):
        Xtr, ytr = summarise(d["X_train"]), d["y_train"]
        model = make_model(class_weights(ytr))
        model.fit(Xtr, ytr)
        if tag == "final":
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_DIR / f"{name}_final.joblib")

        def predict(X):
            probs = np.zeros((len(X), 7))
            probs[:, model.classes_] = model.predict_proba(summarise(X))
            return probs

        return predict, {}
    return fit


def main():
    run_cv("logistic_regression", sklearn_fit(
        lambda cw: LogisticRegression(max_iter=3000, C=1.0,
                                      class_weight=cw),
        "logistic_regression"))

    run_cv("random_forest", sklearn_fit(
        lambda cw: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight=cw,
            n_jobs=-1, random_state=42),
        "random_forest"))


if __name__ == "__main__":
    main()
