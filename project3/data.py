"""The AG News dataset for Project 3.

Free of Django imports so it can be used from a script or a test.

The two parquet files are fetched from Hugging Face into `data/ag_news/` on first
use and cached there. They are not committed (19 MB), and `data/` is gitignored.
"""

import os
import urllib.request

import pandas as pd

TRAIN_URL = ("https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/"
             "data/train-00000-of-00001.parquet")
TEST_URL = ("https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/"
            "data/test-00000-of-00001.parquet")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "ag_news")

# The label order used by the dataset itself.
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
N_CLASSES = len(CLASS_NAMES)


class DatasetMissing(RuntimeError):
    """The data is not cached and could not be fetched."""


def _fetch(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as failure:                          # noqa: BLE001
        if os.path.exists(path):
            os.remove(path)
        raise DatasetMissing(
            f"AG News could not be downloaded ({failure}). Put the parquet files "
            f"in {CACHE} by hand, or run this once with a network connection."
        ) from failure


def _load_split(name, url):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        _fetch(url, path)
    frame = pd.read_parquet(path)
    frame["label"] = frame["label"].astype(int)
    return frame


def load(limit=None, seed=0):
    """`(train, test)` frames with columns `text` and `label`.

    `limit` takes a stratified subsample of the *training* set, which is what the
    tests use so they never depend on a full 120k-row fit.
    """
    train = _load_split("train.parquet", TRAIN_URL)
    test = _load_split("test.parquet", TEST_URL)

    if limit is not None and limit < len(train):
        # Iterating the groups rather than `groupby().apply()`: pandas 3 drops the
        # grouping column from the frame handed to apply, which silently loses
        # `label` here.
        per_class = max(1, limit // N_CLASSES)
        parts = [group.sample(n=min(per_class, len(group)), random_state=seed)
                 for _, group in train.groupby("label", sort=True)]
        train = (pd.concat(parts)
                 .sample(frac=1.0, random_state=seed)
                 .reset_index(drop=True))

    return train.reset_index(drop=True), test.reset_index(drop=True)


def is_cached():
    return all(os.path.exists(os.path.join(CACHE, name))
               for name in ("train.parquet", "test.parquet"))


def describe(train, test):
    """Counts for the interface and the report."""
    return {
        "n_train": len(train),
        "n_test": len(test),
        "classes": CLASS_NAMES,
        "train_per_class": {CLASS_NAMES[label]: int(count) for label, count
                            in train["label"].value_counts().sort_index().items()},
        "test_per_class": {CLASS_NAMES[label]: int(count) for label, count
                           in test["label"].value_counts().sort_index().items()},
        "median_characters": int(train["text"].str.len().median()),
    }
