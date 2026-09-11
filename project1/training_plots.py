"""Figures for the training pipeline (Task 4).

Kept apart from `plots.py`, which draws the dataset itself: these draw results.
Both share the palette and chrome defined in `plots.py` so the app reads as one
piece of work.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from django.conf import settings

from .plots import (DPI, GRID, INK, MARKER_SIZE, MUTED, RING_WIDTH,
                    SEQUENTIAL_HUE, SERIES_COLOURS, SURFACE, _label_cells,
                    _style_axes, MAX_LABELLED_CELLS)

# Two series only: the training score and the test score.
TRAIN_COLOUR = SERIES_COLOURS[0]
TEST_COLOUR = SERIES_COLOURS[1]

LINE_WIDTH = 2


def _save(figure, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    return settings.MEDIA_URL + filename.replace(os.sep, "/")


def _new_axes(width=8, height=4.4):
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    return figure, axes


def save_sweep(result, filename):
    """Score against the swept hyperparameter, for train and test.

    Both curves are drawn on one axis on purpose: they share a scale, and the gap
    between them is the thing worth seeing (it is the overfitting).
    """
    figure, axes = _new_axes()

    rows = [row for row in result["results"] if row["error"] is None]
    values = [row["value"] for row in rows]
    train = [row["train_score"] for row in rows]
    test = [row["test_score"] for row in rows]

    positions = list(range(len(values)))

    axes.plot(positions, train, color=TRAIN_COLOUR, linewidth=LINE_WIDTH,
              marker="o", markersize=5, label="Training score",
              markeredgecolor=SURFACE, markeredgewidth=RING_WIDTH)
    axes.plot(positions, test, color=TEST_COLOUR, linewidth=LINE_WIDTH,
              marker="o", markersize=5, label="Test score",
              markeredgecolor=SURFACE, markeredgewidth=RING_WIDTH)

    # Mark the winner: it is the one number the reader is looking for.
    best_value = result["best"]["value"]
    if best_value in values:
        index = values.index(best_value)
        axes.scatter([index], [result["best"]["test_score"]], s=MARKER_SIZE * 3,
                     facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
        axes.annotate(f"best: {_tidy(best_value)}",
                      (index, result["best"]["test_score"]),
                      textcoords="offset points", xytext=(0, -18),
                      ha="center", va="top", color=INK, fontsize=9)

    axes.set_xticks(positions)
    axes.set_xticklabels([_tidy(value) for value in values])

    _style_axes(axes, result["parameter_label"], result["score_label"])
    _legend(axes)

    return _save(figure, filename)


def save_confusion(detail, filename):
    """Predicted against actual counts for the winning classifier."""
    matrix = np.asarray(detail["matrix"])
    labels = detail["labels"]

    figure, axes = _new_axes(width=max(5, 1.3 * len(labels) + 3),
                             height=max(4, 1.1 * len(labels) + 2))
    ramp = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_HUE)
    image = axes.imshow(matrix, cmap=ramp, aspect="auto")

    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=30, ha="right")
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels)

    highest = float(matrix.max()) if matrix.size else 0.0
    _label_cells(axes, matrix,
                 weight=lambda value: (value / highest) if highest else 0.0,
                 text=lambda value: f"{int(value):,}",
                 max_cells=MAX_LABELLED_CELLS)

    bar = figure.colorbar(image, ax=axes)
    bar.set_label("rows", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_edgecolor(GRID)

    _style_axes(axes, "Predicted", "Actual", grid=False)
    return _save(figure, filename)


def save_predictions(detail, filename):
    """Predicted against actual values, with the line a perfect model would sit on."""
    figure, axes = _new_axes(width=6.4, height=4.6)

    actual = np.asarray(detail["actual"], dtype=float)
    predicted = np.asarray(detail["predicted"], dtype=float)

    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    axes.plot([low, high], [low, high], color=MUTED, linewidth=1,
              linestyle="-", zorder=1, label="Perfect prediction")

    axes.scatter(actual, predicted, s=MARKER_SIZE, color=TRAIN_COLOUR, alpha=0.9,
                 edgecolors=SURFACE, linewidths=RING_WIDTH, zorder=2,
                 label="Test rows")

    _style_axes(axes, "Actual", "Predicted")
    _legend(axes)
    return _save(figure, filename)


def _legend(axes):
    legend = axes.legend(loc="best", fontsize=9, frameon=True, framealpha=1,
                         edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK)
    return legend


def _tidy(value):
    """Short label for a hyperparameter value."""
    if isinstance(value, float):
        if value >= 1 or value == 0:
            return f"{value:g}"
        return f"{value:g}"
    return str(value)
