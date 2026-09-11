"""The Palmer Penguins dataset for Project 2.

Free of Django imports so the data handling can be exercised on its own.

The dataset is bundled as `penguins.csv` at the project root and read from there,
with the `palmerpenguins` package as a fallback. Reading the file first keeps the
app working on a machine where the package is not installed, which matters for a
repository that has to run somewhere else.
"""

import os

import numpy as np
import pandas as pd

TARGET = "species"

# The four measurements. PDP and ALE are offered for exactly these.
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# `year` is numeric in the file but only ever takes three values, so it behaves
# like a label and is treated as one.
CATEGORICAL_FEATURES = ["island", "sex", "year"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SPECIES = ["Adelie", "Chinstrap", "Gentoo"]

BUNDLED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "penguins.csv")

READABLE_NAMES = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
    "island": "Island",
    "sex": "Sex",
    "year": "Year",
    "species": "Species",
}


def readable(name):
    return READABLE_NAMES.get(name, name)


def load_raw():
    """The dataset exactly as published."""
    if os.path.exists(BUNDLED):
        return pd.read_csv(BUNDLED)

    from palmerpenguins import load_penguins       # imported lazily on purpose
    return load_penguins()


def load():
    """Model-ready data: the eight columns, with unusable rows removed.

    Rows missing a measurement or the target are dropped rather than imputed. The
    dataset has only a handful of them (two rows lack every measurement, eleven
    lack `sex`), and dropping keeps every number downstream - counterfactual
    distances, ALE bin averages - computed on values that were really observed
    rather than on invented ones.
    """
    frame = load_raw()
    frame = frame[[TARGET] + FEATURES].copy()

    # `year` is a label here, so hold it as one; it also stops the scaler from
    # treating 2007..2009 as a quantity.
    frame["year"] = frame["year"].astype("Int64").astype(str)

    before = len(frame)
    frame = frame.dropna(subset=[TARGET] + FEATURES).reset_index(drop=True)

    frame.attrs["dropped_rows"] = before - len(frame)
    frame.attrs["total_rows"] = before
    return frame


def features_and_target(frame):
    return frame[FEATURES], frame[TARGET]


def class_names(frame=None):
    """Species in a fixed order, so colours never move between charts."""
    if frame is None:
        return list(SPECIES)
    present = frame[TARGET].unique().tolist()
    ordered = [name for name in SPECIES if name in present]
    return ordered + sorted(name for name in present if name not in SPECIES)


def median_absolute_deviations(frame):
    """MAD per numeric feature, used to weight counterfactual distances.

    A feature whose values barely move should count a given absolute change as a
    big change, which is exactly what dividing by the MAD does. Where the MAD is
    zero (a constant column) it is replaced so the weighting cannot divide by nil.
    """
    deviations = {}
    for name in NUMERIC_FEATURES:
        values = frame[name].astype(float)
        mad = float(np.median(np.abs(values - values.median())))
        if mad <= 0:
            spread = float(values.std())
            mad = spread if spread > 0 else 1.0
        deviations[name] = mad
    return deviations


def categories(frame):
    """Observed values of each categorical feature, for sampling and for forms."""
    return {name: sorted(frame[name].dropna().unique().tolist())
            for name in CATEGORICAL_FEATURES}


def summary(frame):
    """Per-column description for the interface."""
    rows = []
    for name in [TARGET] + FEATURES:
        column = frame[name]
        numeric = name in NUMERIC_FEATURES
        rows.append({
            "name": name,
            "label": readable(name),
            "role": "target" if name == TARGET else "feature",
            "kind": "numeric" if numeric else "categorical",
            "distinct": int(column.nunique()),
            "minimum": f"{column.min():.1f}" if numeric else "-",
            "maximum": f"{column.max():.1f}" if numeric else "-",
            "values": "" if numeric else ", ".join(str(v) for v in sorted(column.unique())),
        })
    return rows
