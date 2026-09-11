"""Counterfactual explanations for Project 2, Task 4.

The method from the task sheet: sample N points locally around x, keep the ones
the model predicts as the desired class, rank them by MAD-weighted L1 distance to
x, and show the best k.

**Noising each kind of feature.** The sheet asks what to do about data that is not
decimal, and the four measurements and the three labels genuinely need different
treatment:

* *numeric* features get Gaussian noise scaled by the feature's own MAD, so a
  feature that naturally varies little is nudged little. Samples are clipped to
  the range actually observed, because a counterfactual with a negative body mass
  would be arithmetic rather than an explanation.
* *categorical* features (island, sex, year) cannot be "nudged" at all -- there is
  no meaningful midpoint between two islands. Each is instead resampled with a
  small probability, uniformly among the *other* observed values. This keeps most
  of them fixed, so a counterfactual differs from x in only a few labels, which is
  what makes it readable.

**Distance.** MAD-weighted L1 over the numeric features, plus a flat cost of 1 for
each categorical feature that differs. Dividing by the MAD puts every numeric
feature on a comparable footing; the flat cost says that changing an island is a
change of about the size of one typical numeric step, so the ranking will not
quietly prefer swapping a penguin's island over moving its bill by a millimetre.

**When nothing is found.** The search widens: more samples and larger variance on
each attempt, as the sheet suggests. Some requests have no answer at all (asking
for a Gentoo while holding a small Adelie's measurements nearly fixed), and the
interface says so rather than pretending.
"""

import numpy as np
import pandas as pd

from .penguins import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                       categories, median_absolute_deviations, readable)

# Widening schedule: (samples, spread in MADs, chance of relabelling a category).
ATTEMPTS = [
    (2000, 0.5, 0.15),
    (4000, 1.0, 0.25),
    (8000, 2.0, 0.40),
    (16000, 3.5, 0.60),
]

DEFAULT_K = 5


class NoCounterfactual(Exception):
    """Raised when the search cannot reach the requested class."""


# note to self: the noising scheme, which the sheet calls out specifically as
# something more than decimal noise. a numeric feature gets gaussian noise
# scaled by its own mad, a categorical one is resampled from the categories
# actually observed in the data, and a binary one is flipped with some
# probability. adding 0.3 to "island = biscoe" would be meaningless, which is
# the whole reason the scheme has to be type aware.
def _sample(example, frame, deviations, options, size, spread, flip, rng):
    """`size` neighbours of `example`, noised according to feature type."""
    sample = {}

    for name in NUMERIC_FEATURES:
        centre = float(example[name])
        noise = rng.normal(0.0, spread * deviations[name], size)
        column = frame[name].astype(float)
        sample[name] = np.clip(centre + noise, column.min(), column.max())

    for name in CATEGORICAL_FEATURES:
        current = example[name]
        values = options[name]
        drawn = np.array([current] * size, dtype=object)

        if len(values) > 1:
            others = [value for value in values if value != current]
            replace = rng.random(size) < flip
            picks = rng.integers(0, len(others), size)
            drawn[replace] = [others[index] for index in picks[replace]]

        sample[name] = drawn

    return pd.DataFrame(sample, columns=FEATURES)


# note to self: mad weighted l1, and the weighting is the part that matters.
# each feature's difference is divided by that feature's median absolute
# deviation, so a millimetre of bill length and a hundred grams of body mass
# become comparable. without it the ranking is just whichever feature happens to
# have the largest units.
def distances(example, candidates, deviations):
    """MAD-weighted L1 distance from `example` to each candidate row."""
    total = np.zeros(len(candidates))

    for name in NUMERIC_FEATURES:
        gap = np.abs(candidates[name].astype(float) - float(example[name]))
        total += gap.to_numpy() / deviations[name]

    for name in CATEGORICAL_FEATURES:
        total += (candidates[name].to_numpy() != example[name]).astype(float)

    return total


def _changes(example, row):
    """Human-readable difference between x and one counterfactual."""
    changed = []
    for name in FEATURES:
        before, after = example[name], row[name]
        if name in NUMERIC_FEATURES:
            delta = float(after) - float(before)
            if abs(delta) < 1e-9:
                continue
            changed.append({
                "feature": name,
                "label": readable(name),
                "before": f"{float(before):.1f}",
                "after": f"{float(after):.1f}",
                "delta": f"{delta:+.1f}",
                "numeric": True,
            })
        elif before != after:
            changed.append({
                "feature": name,
                "label": readable(name),
                "before": str(before),
                "after": str(after),
                "delta": "",
                "numeric": False,
            })
    return changed


# note to self: the loop the sheet describes. sample n points locally around the
# chosen example, keep the ones the currently selected model predicts as the
# target class, rank them by the weighted distance above, show the best k. if
# nothing is found, widen the spread and raise n and try again rather than
# reporting failure on the first pass. it takes the pipeline of the selected
# model, which is what actually makes this region linked to the model class and
# the lambda from tasks 1 to 3 instead of just sitting on the same page as them.
def generate(pipeline, frame, example, target, k=DEFAULT_K, seed=0):
    """Counterfactuals for `example` that the model predicts as `target`.

    Returns `(rows, search)` where `search` records how hard the search had to
    work, so the interface can be honest about it.
    """
    if target not in set(pipeline.classes_):
        raise NoCounterfactual(f"{target} is not a class this model predicts.")

    deviations = median_absolute_deviations(frame)
    options = categories(frame)
    rng = np.random.default_rng(seed)

    original = pipeline.predict(pd.DataFrame([example], columns=FEATURES))[0]
    if original == target:
        raise NoCounterfactual(
            f"This penguin is already predicted as {target}, so there is nothing "
            f"to explain. Pick a different target class.")

    attempts = []
    for index, (size, spread, flip) in enumerate(ATTEMPTS, start=1):
        candidates = _sample(example, frame, deviations, options, size, spread,
                             flip, rng)
        predicted = pipeline.predict(candidates)
        matches = candidates[predicted == target]

        attempts.append({"attempt": index, "samples": size, "spread": spread,
                         "flip": flip, "found": int(len(matches))})

        if matches.empty:
            continue

        matches = matches.copy()
        matches["_distance"] = distances(example, matches, deviations)
        matches = matches.sort_values("_distance").head(k)

        rows = []
        for _, row in matches.iterrows():
            probabilities = pipeline.predict_proba(
                pd.DataFrame([row[FEATURES]], columns=FEATURES))[0]
            confidence = float(probabilities[list(pipeline.classes_).index(target)])
            rows.append({
                "distance": float(row["_distance"]),
                "confidence": confidence,
                "values": {name: row[name] for name in FEATURES},
                "changes": _changes(example, row),
            })

        return rows, {"attempts": attempts, "widened": index > 1,
                      "original": str(original)}

    raise NoCounterfactual(
        f"No counterfactual reaching {target} was found, even after "
        f"{ATTEMPTS[-1][0]:,} samples at {ATTEMPTS[-1][1]} MADs of spread. This "
        f"penguin may be too far inside its own region for the selected model.")


def example_choices(frame, limit=None):
    """Rows offered in the picker, labelled so they can be told apart."""
    rows = []
    for index, row in frame.iterrows():
        rows.append({
            "index": int(index),
            "label": (f"#{index} - {row['species']}, {row['island']}, "
                      f"bill {row['bill_length_mm']:.1f}mm, "
                      f"{row['body_mass_g']:.0f}g"),
        })
        if limit and len(rows) >= limit:
            break
    return rows
