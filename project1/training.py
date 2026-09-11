"""Training pipeline for Project 1, Task 4.

Plain scikit-learn and pandas, free of Django imports, so the pipeline can be
exercised without a request.

**What the user controls, and what is automatic.** The project description asks
this explicitly, so the split is deliberate:

* the user chooses the *model family* and the *test-set size* - the two decisions
  that need judgement and that change what the result means - and the *random
  seed* of the split, which changes nothing about the meaning but lets a reader
  see how much of the reported score is sampling noise;
* the app handles preprocessing, the hyperparameter sweep and the choice of score
  automatically, because those are mechanical given the problem type, and getting
  them wrong (leaking the test set into scaling, scoring a regression with
  accuracy) is the kind of mistake an interface should not let a person make.

Each family sweeps exactly one hyperparameter: the one that governs how much
structure the model may take on, so the sweep traces the underfit/overfit curve.
"""

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .dataset import CLASSIFICATION, REGRESSION

RANDOM_SEED = 0

MIN_TEST_SIZE = 0.1
MAX_TEST_SIZE = 0.5
DEFAULT_TEST_SIZE = 0.25


class TrainingError(ValueError):
    """Raised with a message meant for the user."""


def _tree_classifier(value, seed):
    return DecisionTreeClassifier(max_depth=value, random_state=seed)


def _tree_regressor(value, seed):
    return DecisionTreeRegressor(max_depth=value, random_state=seed)


def _knn_classifier(value, seed):
    return KNeighborsClassifier(n_neighbors=value)


def _knn_regressor(value, seed):
    return KNeighborsRegressor(n_neighbors=value)


def _logistic(value, seed):
    return LogisticRegression(C=value, max_iter=5000, random_state=seed)


def _ridge(value, seed):
    return Ridge(alpha=value, random_state=seed)


def _forest_classifier(value, seed):
    return RandomForestClassifier(n_estimators=value, random_state=seed, n_jobs=1)


def _forest_regressor(value, seed):
    return RandomForestRegressor(n_estimators=value, random_state=seed, n_jobs=1)


DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
NEIGHBOURS = [1, 3, 5, 7, 9, 11, 15, 21, 31]
STRENGTHS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
TREES = [10, 25, 50, 100, 200]

MODELS = {
    CLASSIFICATION: {
        "tree": {
            "label": "Decision tree",
            "parameter": "max_depth",
            "parameter_label": "Maximum depth",
            "values": DEPTHS,
            "build": _tree_classifier,
            "scale": False,
            "note": "Deeper trees carve out finer regions, and eventually memorise.",
        },
        "knn": {
            "label": "k-nearest neighbours",
            "parameter": "n_neighbors",
            "parameter_label": "Neighbours (k)",
            "values": NEIGHBOURS,
            "build": _knn_classifier,
            "scale": True,
            "note": "Small k follows every point; large k smooths the boundary.",
        },
        "logistic": {
            "label": "Logistic regression",
            "parameter": "C",
            "parameter_label": "Inverse regularisation (C)",
            "values": STRENGTHS,
            "build": _logistic,
            "scale": True,
            "note": "Larger C means weaker regularisation and a freer fit.",
        },
        "forest": {
            "label": "Random forest",
            "parameter": "n_estimators",
            "parameter_label": "Number of trees",
            "values": TREES,
            "build": _forest_classifier,
            "scale": False,
            "note": "More trees average away variance, with diminishing returns.",
        },
    },
    REGRESSION: {
        "tree": {
            "label": "Decision tree",
            "parameter": "max_depth",
            "parameter_label": "Maximum depth",
            "values": DEPTHS,
            "build": _tree_regressor,
            "scale": False,
            "note": "Deeper trees carve out finer regions, and eventually memorise.",
        },
        "knn": {
            "label": "k-nearest neighbours",
            "parameter": "n_neighbors",
            "parameter_label": "Neighbours (k)",
            "values": NEIGHBOURS,
            "build": _knn_regressor,
            "scale": True,
            "note": "Small k follows every point; large k smooths the prediction.",
        },
        "ridge": {
            "label": "Ridge regression",
            "parameter": "alpha",
            "parameter_label": "Regularisation (alpha)",
            "values": STRENGTHS,
            "build": _ridge,
            "scale": True,
            "note": "Larger alpha pulls the coefficients towards zero.",
        },
        "forest": {
            "label": "Random forest",
            "parameter": "n_estimators",
            "parameter_label": "Number of trees",
            "values": TREES,
            "build": _forest_regressor,
            "scale": False,
            "note": "More trees average away variance, with diminishing returns.",
        },
    },
}


def model_choices(problem_type):
    """(key, label) pairs the interface offers for this problem type."""
    family = MODELS.get(problem_type, {})
    return [(key, family[key]["label"]) for key in family]


# note to self: the preprocessor lives inside the pipeline so it is refitted on
# the training split alone. this is the one thing in project 1 that would be
# quietly wrong if i got it wrong: fit a scaler on the whole frame and the test
# score comes out optimistic while nothing visibly breaks. scaling is only
# switched on for the families that need it, k-nn and the linear models, since
# trees and forests are scale invariant.
def _build_preprocessor(frame, feature_names, scale):
    """Impute, scale numbers and one-hot encode labels.

    Fitted inside the pipeline, so it is refitted on the training split alone and
    nothing about the test set leaks into the transformation.
    """
    numeric = [name for name in feature_names if ptypes.is_numeric_dtype(frame[name])]
    categorical = [name for name in feature_names if name not in numeric]

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    blocks = []
    if numeric:
        blocks.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        blocks.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))

    if not blocks:
        raise TrainingError("This dataset has no usable feature columns.")

    return ColumnTransformer(blocks), numeric, categorical


def _prepare(dataset):
    """Features and target, with rows that cannot be used removed."""
    if not dataset.feature_names:
        raise TrainingError("This dataset has no feature columns to learn from.")

    frame = dataset.frame
    usable = frame[dataset.target_name].notna()
    if int(usable.sum()) < 10:
        raise TrainingError(
            "At least 10 rows with a target value are needed to train and evaluate.")

    frame = frame.loc[usable]
    return frame[dataset.feature_names], frame[dataset.target_name], frame


# note to self: task 4 asks me to decide what the user controls and what the app
# does automatically, so that split is the deliverable and not an implementation
# detail. the user picks the model family, the test size and the seed. the app
# picks the preprocessing, the hyperparameter to sweep and the score. the
# reasoning is that the first three change what the result means and need
# judgement, while the rest is mechanical once the problem type is known, and
# getting it wrong (a scaler fitted on the test set, a regression scored with
# accuracy) is exactly the kind of mistake an interface should make impossible.
# each family sweeps the one hyperparameter that governs how much structure the
# model may take on, so the sweep traces the underfit to overfit curve.
def train(dataset, model_key, test_size=DEFAULT_TEST_SIZE, seed=RANDOM_SEED):
    """Split, sweep one hyperparameter, score every fit, and report the best."""
    problem_type = dataset.problem_type
    family = MODELS.get(problem_type, {}).get(model_key)
    if family is None:
        raise TrainingError(f"Unknown model for a {problem_type} problem.")

    features, target, frame = _prepare(dataset)
    classification = problem_type == CLASSIFICATION

    if classification and target.nunique() < 2:
        raise TrainingError(
            "Every row carries the same label, so there is nothing to tell apart.")

    stratify = None
    if classification and int(target.value_counts().min()) >= 2:
        # Stratifying needs at least one row of every class on each side.
        stratify = target

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError as invalid:
        raise TrainingError(f"Could not split the data: {invalid}")

    if len(x_train) < 2 or len(x_test) < 1:
        raise TrainingError("The split leaves too few rows. Try a smaller test size.")

    preprocessor, numeric, categorical = _build_preprocessor(
        frame, dataset.feature_names, family["scale"])

    results = []
    best = None
    for value in family["values"]:
        pipeline = Pipeline([("prepare", preprocessor),
                             ("model", family["build"](value, seed))])
        try:
            pipeline.fit(x_train, y_train)
        except Exception as failure:                      # noqa: BLE001
            # One awkward hyperparameter should not sink the whole sweep.
            results.append({"value": value, "train_score": None,
                            "test_score": None, "error": str(failure)})
            continue

        row = {
            "value": value,
            "train_score": _score(pipeline, x_train, y_train, classification),
            "test_score": _score(pipeline, x_test, y_test, classification),
            "error": None,
        }
        results.append(row)

        if best is None or row["test_score"] > best["test_score"]:
            best = {**row, "pipeline": pipeline}

    if best is None:
        raise TrainingError("Every model in the sweep failed to fit.")

    return {
        "problem_type": problem_type,
        "model_key": model_key,
        "model_label": family["label"],
        "parameter": family["parameter"],
        "parameter_label": family["parameter_label"],
        "note": family["note"],
        "score_label": "Accuracy" if classification else "R²",
        "results": results,
        "best": best,
        "detail": _describe_best(best["pipeline"], x_test, y_test, classification, target),
        "importance": feature_importance(best["pipeline"]),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_size": test_size,
        "seed": seed,
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "stratified": stratify is not None,
    }


# note to self: read off the fitted estimator, never recomputed. trees and
# forests expose feature_importances_, the linear models expose coefficients,
# and k-nn genuinely has no such notion so this returns none for it. reporting
# none is the point, inventing a number there would be worse than admitting the
# model does not have one.
def feature_importance(pipeline, top=10):
    """Which inputs the winning model actually leaned on.

    Read off the fitted estimator rather than recomputed: a tree exposes the
    impurity decrease it credits to each column, and a linear model its
    coefficients. The two are not the same quantity and are labelled as such.

    Returns None for k-nearest neighbours, which has no per-feature notion of
    importance at all -- saying so is better than inventing a number.
    """
    prepare = pipeline.named_steps["prepare"]
    model = pipeline.named_steps["model"]

    try:
        names = [str(name) for name in prepare.get_feature_names_out()]
    except Exception:                                     # noqa: BLE001
        return None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity decrease"
        note = ("How much each column reduced impurity across the tree. It says "
                "what the model used, not what causes the target.")
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = (np.abs(coefficients).sum(axis=0) if coefficients.ndim > 1
                  else np.abs(coefficients))
        kind = "absolute coefficient"
        note = ("Size of each coefficient on the standardised inputs, summed "
                "across classes. Comparable only because the features were "
                "scaled first.")
    else:
        return None

    if len(values) != len(names):                         # pragma: no cover
        return None

    total = float(values.sum())
    order = np.argsort(-values)[:top]

    return {
        "kind": kind,
        "note": note,
        "rows": [
            {
                "name": names[index].split("__", 1)[-1],
                "value": float(values[index]),
                "share": float(values[index] / total) if total else 0.0,
            }
            for index in order if values[index] > 0
        ],
    }


def _score(pipeline, features, target, classification):
    """Accuracy for labels, R-squared for quantities."""
    predicted = pipeline.predict(features)
    if classification:
        return float(accuracy_score(target, predicted))
    return float(r2_score(target, predicted))


def _describe_best(pipeline, x_test, y_test, classification, target):
    """Extra numbers worth showing for the winning model."""
    predicted = pipeline.predict(x_test)

    if classification:
        labels = sorted(pd.Series(target).dropna().unique().tolist(), key=str)
        matrix = confusion_matrix(y_test, predicted, labels=labels)
        return {
            "kind": "classification",
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "labels": [str(label) for label in labels],
            "matrix": matrix.tolist(),
        }

    actual = np.asarray(y_test, dtype=float)
    guessed = np.asarray(predicted, dtype=float)
    return {
        "kind": "regression",
        "r2": float(r2_score(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "actual": actual.tolist(),
        "predicted": guessed.tolist(),
    }
