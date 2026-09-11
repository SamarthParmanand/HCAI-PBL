"""Figures for Project 3.

Palette and chrome are imported from Project 1 so every project in this
submission draws with the same validated colours.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from django.conf import settings

from project1.plots import (DPI, GRID, INK, MARKER_SIZE, MAX_LABELLED_CELLS,
                            MUTED, RING_WIDTH, SEQUENTIAL_HUE, SERIES_COLOURS,
                            SURFACE, _label_cells, _style_axes)

from .active import STRATEGY_LABELS
from .data import CLASS_NAMES

LINE_WIDTH = 2
OTHER = "#94a3b8"


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


def _legend(axes, title=None):
    legend = axes.legend(title=title, loc="best", fontsize=9, frameon=True,
                         framealpha=1, edgecolor=GRID)
    if title:
        legend.get_title().set_fontsize(9)
        legend.get_title().set_color(MUTED)
    for text in legend.get_texts():
        text.set_color(INK)
    return legend


def save_accuracy_comparison(bundle, filename):
    """Who is how good: classifier, expert, team, and the oracle ceiling."""
    policies = bundle["policies"]
    rows = [
        ("Expert alone", policies["expert_only"]["team_accuracy"], OTHER),
        ("Classifier alone", policies["classifier_only"]["team_accuracy"],
         SERIES_COLOURS[0]),
        ("Confidence rule (hindsight)", policies["confidence_best"]["team_accuracy"],
         SERIES_COLOURS[1]),
        ("Competence rule", policies["competence_zero_margin"]["team_accuracy"],
         SERIES_COLOURS[2]),
        ("Oracle ceiling", policies["oracle"]["team_accuracy"], OTHER),
    ]

    figure, axes = _new_axes(width=8, height=4.0)
    positions = np.arange(len(rows))
    axes.barh(positions, [row[1] for row in rows],
              color=[row[2] for row in rows], height=0.6)

    for position, (_, value, _colour) in zip(positions, rows):
        axes.text(value + 0.004, position, f"{value:.4f}", va="center",
                  fontsize=9, color=INK)

    axes.set_yticks(positions)
    axes.set_yticklabels([row[0] for row in rows], fontsize=9)
    axes.invert_yaxis()
    axes.set_xlim(0.6, 1.02)
    _style_axes(axes, "Test accuracy", "")
    return _save(figure, filename)


def save_deferral_curves(bundle, filename):
    """Team accuracy against how much work is handed over."""
    figure, axes = _new_axes()

    for key, label, colour in (
        ("confidence_sweep", "Confidence threshold", SERIES_COLOURS[1]),
        ("competence_sweep", "Competence rule (calibrated)", SERIES_COLOURS[2]),
        ("competence_text_sweep", "Competence rule (text model)", SERIES_COLOURS[3]),
    ):
        rows = sorted(bundle[key], key=lambda row: row["deferral_rate"])
        axes.plot([row["deferral_rate"] for row in rows],
                  [row["team_accuracy"] for row in rows],
                  color=colour, linewidth=LINE_WIDTH, label=label)

    classifier_only = bundle["policies"]["classifier_only"]["team_accuracy"]
    axes.axhline(classifier_only, color=MUTED, linewidth=1, linestyle="-",
                 label="Classifier alone")

    _style_axes(axes, "Fraction of articles handed to the expert",
                "Team test accuracy")
    _legend(axes)
    return _save(figure, filename)


def save_active_learning(bundle, filename, metric="team_accuracy",
                         ylabel="Team test accuracy"):
    """Learning curves: what each strategy buys per expert label."""
    figure, axes = _new_axes()

    palette = list(SERIES_COLOURS) + [OTHER]
    for index, (strategy, history) in enumerate(bundle["active"].items()):
        axes.plot([row["queries"] for row in history],
                  [row[metric] for row in history],
                  color=palette[index % len(palette)], linewidth=LINE_WIDTH,
                  marker="o", markersize=4, markeredgecolor=SURFACE,
                  markeredgewidth=RING_WIDTH,
                  label=STRATEGY_LABELS.get(strategy, strategy))

    if metric == "team_accuracy":
        axes.axhline(bundle["active_target"], color=MUTED, linewidth=1,
                     linestyle="-", label="Full expert supervision")

    _style_axes(axes, "Expert labels bought", ylabel)
    _legend(axes)
    return _save(figure, filename)


def save_confusion(evaluation, filename):
    """The classifier's mistakes by topic."""
    matrix = np.asarray(evaluation["matrix"])
    labels = evaluation["labels"]

    figure, axes = _new_axes(width=5.8, height=4.4)
    ramp = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_HUE)
    image = axes.imshow(matrix, cmap=ramp, aspect="auto")

    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=20, ha="right")
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels)

    highest = float(matrix.max()) if matrix.size else 0.0
    _label_cells(axes, matrix,
                 weight=lambda value: (value / highest) if highest else 0.0,
                 text=lambda value: f"{int(value):,}",
                 max_cells=MAX_LABELLED_CELLS)

    bar = figure.colorbar(image, ax=axes)
    bar.set_label("articles", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_edgecolor(GRID)

    _style_axes(axes, "Predicted", "Actual", grid=False)
    return _save(figure, filename)


def save_expert_profile(bundle, filename):
    """Where the expert is strong and where it is not, beside the classifier."""
    expert = bundle["expert"]
    system = bundle["system"]

    figure, axes = _new_axes(width=8, height=4.0)
    positions = np.arange(len(CLASS_NAMES))
    width = 0.38

    expert_values = [expert["per_class"][name]["accuracy"] for name in CLASS_NAMES]
    model_values = [system["per_class"][name]["recall"] for name in CLASS_NAMES]

    axes.bar(positions - width / 2, model_values, width=width,
             color=SERIES_COLOURS[0], label="Classifier")
    axes.bar(positions + width / 2, expert_values, width=width,
             color=SERIES_COLOURS[1], label="Expert")

    axes.set_xticks(positions)
    axes.set_xticklabels(CLASS_NAMES)
    axes.set_ylim(0, 1.05)

    _style_axes(axes, "Topic", "Accuracy on that topic")
    _legend(axes)
    return _save(figure, filename)
