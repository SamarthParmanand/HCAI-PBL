"""Model families and the complexity trade-off for Project 2, Tasks 1 to 3.

Named `learning` rather than `models` so it is never confused with Django's own
`models.py`. No Django imports here either.

The lecture's objective is

    argmin_f  (1/n) sum_i loss(f(x_i), y_i) + lambda * Omega(f)

and the interface works with the equivalent maximisation the task sheet states:

    acc_test - lambda * Omega(f)

`lambda` here is **not** the fitting-time regularisation parameter. Each family is
fitted once per regularisation setting, producing a menu of models; lambda then
picks one off that menu. Turning lambda up therefore never refits anything, it
just changes which already-fitted model is judged the best buy.

**The two complexity measures**

* Decision tree: `Omega` is the number of leaves, as the task sheet specifies,
  and `max_leaf_nodes` is the fitting-time control.
* Logistic regression: `Omega` is the number of non-zero coefficients, with an
  L1 penalty (strength `C`) as the fitting-time control. This is the honest
  analogue of counting leaves: both count the pieces of the model a person has
  to read to understand it, and L1 is what actually drives coefficients to zero
  so the count can fall. A norm such as ||w||_2 would also measure "size", but it
  shrinks smoothly without ever removing a term, so it would not correspond to
  anything the reader could skip.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .penguins import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
                       class_names, load)

TREE = "tree"
LOGREG = "logreg"

MODEL_CLASSES = (
    (TREE, "Decision tree"),
    (LOGREG, "Logistic regression"),
)

MODEL_LABELS = dict(MODEL_CLASSES)

# Fitting-time regularisation grids.
LEAF_LIMITS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
L1_STRENGTHS = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]

# The lambda slider. The range is chosen to cover the transitions that actually
# matter: every switch between models happens below about 0.02, and by the top of
# the range the selection has collapsed to a two- or three-piece model. Pushing
# far enough to force the degenerate model (no split at all, or every coefficient
# zero) would need lambda above 0.14, which would squeeze the whole interesting
# region into the first tenth of the slider for no benefit.
LAMBDA_MIN = 0.0
LAMBDA_MAX = 0.05
LAMBDA_STEP = 0.0005
DEFAULT_LAMBDA = 0.002

DEFAULT_TEST_SIZE = 0.3
DEFAULT_SEED = 0

OMEGA_LABELS = {
    TREE: "leaves",
    LOGREG: "non-zero coefficients",
}


class FittedModel:
    """One fitted model plus everything the interface reports about it."""

    def __init__(self, model_class, setting, pipeline, accuracy_test,
                 accuracy_train, omega, extra=None):
        self.model_class = model_class
        self.setting = setting                  # the fitting-time control
        self.pipeline = pipeline
        self.accuracy_test = accuracy_test
        self.accuracy_train = accuracy_train
        self.omega = omega
        self.extra = extra or {}

    def objective(self, lam):
        """acc_test - lambda * Omega(f) -- the quantity the slider maximises."""
        return self.accuracy_test - lam * self.omega

    @property
    def estimator(self):
        return self.pipeline.named_steps["model"]

    @property
    def omega_label(self):
        return OMEGA_LABELS[self.model_class]

    def __repr__(self):                                   # pragma: no cover
        return (f"<FittedModel {self.model_class} setting={self.setting} "
                f"acc={self.accuracy_test:.3f} omega={self.omega}>")


def build_preprocessor(scale):
    """Numeric columns first, then the one-hot block.

    Keeping the numeric block first means the transformed matrix starts with
    NUMERIC_FEATURES in order, which the exact-gradient code in `effects` relies
    on. `numeric_column_index` is the checked way to ask for that mapping.
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def numeric_column_index(pipeline, feature):
    """Where `feature` sits in the matrix the estimator actually sees."""
    names = list(pipeline.named_steps["prepare"].get_feature_names_out())
    wanted = f"numeric__{feature}"
    if wanted not in names:
        raise KeyError(f"{feature} is not a numeric column of the transformed matrix")
    return names.index(wanted)


def _split(frame, test_size, seed):
    features, target = frame[FEATURES], frame[TARGET]
    return train_test_split(features, target, test_size=test_size,
                            random_state=seed, stratify=target)


# note to self: omega is read back off the fitted tree with get_n_leaves rather
# than assumed to equal the cap, because a tree capped at 30 leaves often
# realises fewer than 30.
def _fit_trees(x_train, x_test, y_train, y_test, seed):
    family = []
    for limit in LEAF_LIMITS:
        pipeline = Pipeline([
            ("prepare", build_preprocessor(scale=False)),
            ("model", DecisionTreeClassifier(max_leaf_nodes=limit, random_state=seed)),
        ])
        pipeline.fit(x_train, y_train)
        tree = pipeline.named_steps["model"]
        family.append(FittedModel(
            TREE, limit, pipeline,
            accuracy_test=float(accuracy_score(y_test, pipeline.predict(x_test))),
            accuracy_train=float(accuracy_score(y_train, pipeline.predict(x_train))),
            # The realised number of leaves, which can be below the cap.
            omega=int(tree.get_n_leaves()),
            extra={"depth": int(tree.get_depth())},
        ))
    return family


# note to self: omega for logistic regression is the number of non-zero
# coefficients, and i need to be able to defend that choice rather than just
# state it. it counts the same thing leaves count, the pieces of the model a
# reader has to hold in their head, and l1 is what actually drives coefficients
# to zero so the count can fall. the l2 norm also measures size, but it shrinks
# everything smoothly and never removes a term, so the number would not
# correspond to anything a reader could skip.
def _fit_logistic(x_train, x_test, y_train, y_test, seed):
    family = []
    for strength in L1_STRENGTHS:
        pipeline = Pipeline([
            ("prepare", build_preprocessor(scale=True)),
            # l1_ratio=1 is pure L1. scikit-learn 1.8 deprecated `penalty=`
            # in favour of this, and mixing the two raises an inconsistency
            # warning, so the new spelling is the only one used here.
            ("model", LogisticRegression(solver="saga", l1_ratio=1.0, C=strength,
                                         max_iter=20000, tol=1e-4,
                                         random_state=seed)),
        ])
        pipeline.fit(x_train, y_train)
        model = pipeline.named_steps["model"]

        coefficients = np.asarray(model.coef_)
        non_zero = int(np.count_nonzero(coefficients))
        used = int(np.count_nonzero(np.abs(coefficients).sum(axis=0)))

        family.append(FittedModel(
            LOGREG, strength, pipeline,
            accuracy_test=float(accuracy_score(y_test, pipeline.predict(x_test))),
            accuracy_train=float(accuracy_score(y_train, pipeline.predict(x_train))),
            omega=non_zero,
            extra={"features_used": used,
                   "columns": list(pipeline.named_steps["prepare"].get_feature_names_out())},
        ))
    return family


@lru_cache(maxsize=4)
# note to self: task 1 is kept separate from task 2 on purpose. task 1 asks for
# a decision tree with its test accuracy and its leaf count and says nothing
# about a trade-off, and every member of the task 2 family is capped by
# max_leaf_nodes, so none of them is the plain unconstrained tree task 1 is
# describing. this card also must not move when the lambda slider moves.
def _fit_default_tree_cached(test_size, seed):
    """Task 1's tree: `DecisionTreeClassifier()` with nothing constrained.

    Task 1 asks for *a* decision tree, its test accuracy and its number of
    leaves, with no mention of a complexity trade-off -- that is Task 2. So this
    is the library default, grown until the leaves are pure, and its realised
    leaf count is whatever the data produces. It is deliberately fitted outside
    `LEAF_LIMITS`: every member of the Task 2 family is capped, so none of them
    is the unconstrained tree Task 1 describes.

    Same split as the families (same `test_size` and `seed`), so the accuracies
    on the page are comparable.
    """
    frame = load()
    x_train, x_test, y_train, y_test = _split(frame, test_size, seed)

    pipeline = Pipeline([
        ("prepare", build_preprocessor(scale=False)),
        ("model", DecisionTreeClassifier(random_state=seed)),
    ])
    pipeline.fit(x_train, y_train)
    tree = pipeline.named_steps["model"]

    return FittedModel(
        TREE, None, pipeline,
        accuracy_test=float(accuracy_score(y_test, pipeline.predict(x_test))),
        accuracy_train=float(accuracy_score(y_train, pipeline.predict(x_train))),
        omega=int(tree.get_n_leaves()),
        extra={"depth": int(tree.get_depth()), "unconstrained": True},
    )


def fit_default_tree(test_size=DEFAULT_TEST_SIZE, seed=DEFAULT_SEED):
    """The unconstrained tree Task 1 asks for. Cached; see `fit_family`."""
    return _fit_default_tree_cached(float(test_size), int(seed))


@lru_cache(maxsize=16)
def _fit_family_cached(model_class, test_size, seed):
    frame = load()
    x_train, x_test, y_train, y_test = _split(frame, test_size, seed)

    if model_class == TREE:
        family = _fit_trees(x_train, x_test, y_train, y_test, seed)
    elif model_class == LOGREG:
        family = _fit_logistic(x_train, x_test, y_train, y_test, seed)
    else:
        raise ValueError(f"unknown model class {model_class!r}")

    return {
        "family": family,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "frame": frame,
        "classes": class_names(frame),
    }


def fit_family(model_class, test_size=DEFAULT_TEST_SIZE, seed=DEFAULT_SEED):
    """Every model of one class, fitted across the regularisation grid.

    Cached: the slider changes which model is *selected*, never the fitting, so a
    slider drag must not retrain anything.
    """
    return _fit_family_cached(model_class, float(test_size), int(seed))


# note to self: this is the whole of task 2 in about ten lines, and the bit i
# should be able to write from memory. every model in the family already carries
# its test accuracy and its omega, so selecting is just taking the max of
# acc_test - lam * omega over the family. nothing is refitted here, which is the
# point i keep making about lambda: the family is fitted once per fitting-time
# setting and lambda only picks one off that menu. ties go to the smaller omega
# deliberately, since at the lambda where a bigger model stops paying for itself
# the interpretable one is the answer the question is after.
def select(family, lam):
    """The maximiser of acc_test - lambda * Omega(f).

    Ties go to the simpler model: at the point where a bigger model stops paying
    for itself, the interpretable one is the intended answer.
    """
    best = None
    for model in family:
        value = model.objective(lam)
        if best is None:
            best = model
            continue
        current = best.objective(lam)
        if value > current or (value == current and model.omega < best.omega):
            best = model
    return best


def trade_off_table(family, lam):
    """Every model in the family, with its objective at this lambda."""
    chosen = select(family, lam)
    rows = []
    for model in family:
        rows.append({
            "setting": model.setting,
            "omega": model.omega,
            "accuracy_test": model.accuracy_test,
            "accuracy_train": model.accuracy_train,
            "objective": model.objective(lam),
            "selected": model is chosen,
            "extra": model.extra,
        })
    return rows


def switch_points(family):
    """The lambda values at which the selected model changes.

    Useful for the report: it tells the reader where the interesting settings of
    the slider are, rather than making them hunt.
    """
    points = []
    previous = None
    steps = int(round((LAMBDA_MAX - LAMBDA_MIN) / LAMBDA_STEP)) + 1
    for step in range(steps):
        lam = LAMBDA_MIN + step * LAMBDA_STEP
        chosen = select(family, lam)
        if previous is None or chosen.setting != previous.setting:
            points.append({"lam": lam, "setting": chosen.setting,
                           "omega": chosen.omega,
                           "accuracy_test": chosen.accuracy_test})
            previous = chosen
    return points


def evaluation(model, data):
    """Confusion matrix and per-class counts for a fitted model."""
    predicted = model.pipeline.predict(data["x_test"])
    labels = data["classes"]
    matrix = confusion_matrix(data["y_test"], predicted, labels=labels)
    return {"labels": labels, "matrix": matrix.tolist(),
            "accuracy": float(accuracy_score(data["y_test"], predicted))}


def coefficient_table(model):
    """Non-zero logistic coefficients, largest first, for display."""
    if model.model_class != LOGREG:
        return []

    columns = model.extra["columns"]
    coefficients = np.asarray(model.estimator.coef_)
    classes = list(model.estimator.classes_)

    rows = []
    for row, species in enumerate(classes):
        for index, column in enumerate(columns):
            weight = float(coefficients[row][index])
            if weight == 0:
                continue
            rows.append({
                "species": species,
                "feature": column.split("__", 1)[-1],
                "weight": weight,
                "magnitude": abs(weight),
            })
    rows.sort(key=lambda entry: entry["magnitude"], reverse=True)
    return rows
