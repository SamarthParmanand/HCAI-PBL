"""Figures for Project 2.

The palette and chrome are imported from Project 1 rather than restated, so both
projects in this submission read as one piece of work and the categorical colours
stay validated in one place. Species are always drawn in the same fixed order, so
a colour means the same thing on every chart in the app.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import plot_tree

from django.conf import settings

from project1.plots import (DPI, GRID, INK, MARKER_SIZE, MAX_LABELLED_CELLS,
                            MUTED, RING_WIDTH, SEQUENTIAL_HUE, SERIES_COLOURS,
                            SURFACE, _label_cells, _style_axes)

from .penguins import readable

LINE_WIDTH = 2


def species_colour(classes, name):
    """Fixed slot per species, so Adelie is the same blue everywhere."""
    return SERIES_COLOURS[list(classes).index(name) % len(SERIES_COLOURS)]


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


def _tidy_names(pipeline):
    """Readable feature names for the transformed columns."""
    raw = pipeline.named_steps["prepare"].get_feature_names_out()
    names = []
    for column in raw:
        stripped = column.split("__", 1)[-1]
        names.append(readable(stripped) if stripped in ("bill_length_mm",
                                                        "bill_depth_mm",
                                                        "flipper_length_mm",
                                                        "body_mass_g")
                     else stripped.replace("_", " "))
    return names


def save_tree(model, classes, filename):
    """The fitted tree itself: Task 1 asks for the tree to be shown."""
    tree = model.estimator
    leaves = max(int(tree.get_n_leaves()), 2)
    depth = max(int(tree.get_depth()), 1)

    # Sized to the tree's own shape, and capped: past this a tree is read by
    # opening the image full size rather than by squinting at the page.
    figure, axes = plt.subplots(figsize=(min(2.0 + 1.05 * leaves, 15),
                                         min(1.6 + 1.25 * depth, 9)))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    plot_tree(
        tree,
        feature_names=_tidy_names(model.pipeline),
        class_names=list(tree.classes_),
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        # None lets matplotlib pick a size that actually fits the boxes; a fixed
        # size leaves either overflowing text or a mostly empty figure.
        fontsize=None,
        ax=axes,
    )
    axes.set_axis_off()
    return _save(figure, filename)


def save_coefficients(model, classes, filename, top=14):
    """Non-zero logistic coefficients, grouped by species."""
    from .learning import coefficient_table

    rows = coefficient_table(model)[:top]
    if not rows:
        figure, axes = _new_axes(width=7, height=2.4)
        axes.text(0.5, 0.5, "Every coefficient is zero at this penalty:\n"
                            "the model predicts one class for everything.",
                  ha="center", va="center", color=MUTED, fontsize=11)
        axes.set_axis_off()
        return _save(figure, filename)

    figure, axes = _new_axes(width=8, height=max(3.0, 0.34 * len(rows) + 1.4))

    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        axes.barh(position, row["weight"], height=0.62,
                  color=species_colour(classes, row["species"]))

    axes.set_yticks(positions)
    axes.set_yticklabels([f"{row['feature']}  ({row['species']})" for row in rows],
                         fontsize=9)
    axes.axvline(0, color=GRID, linewidth=1)

    # A legend is the dependable identity channel, since the bars are grouped.
    handles = [plt.Line2D([0], [0], color=species_colour(classes, name),
                          linewidth=6, label=name)
               for name in classes]
    legend = axes.legend(handles=handles, title="Species", loc="best", fontsize=9,
                         frameon=True, framealpha=1, edgecolor=GRID)
    legend.get_title().set_fontsize(9)
    legend.get_title().set_color(MUTED)
    for text in legend.get_texts():
        text.set_color(INK)

    _style_axes(axes, "Coefficient (standardised features)", "")
    return _save(figure, filename)


def save_trade_off(family, lam, chosen, filename):
    """Accuracy and the objective against complexity, with the winner marked.

    Two series on one axis, sharing a 0..1 scale: the point is the vertical gap
    between them, which is exactly the penalty lambda * Omega.
    """
    figure, axes = _new_axes(width=8, height=4.4)

    ordered = sorted(family, key=lambda model: model.omega)
    omegas = [model.omega for model in ordered]
    accuracy = [model.accuracy_test for model in ordered]
    objective = [model.objective(lam) for model in ordered]

    axes.plot(omegas, accuracy, color=SERIES_COLOURS[0], linewidth=LINE_WIDTH,
              marker="o", markersize=5, markeredgecolor=SURFACE,
              markeredgewidth=RING_WIDTH, label="Test accuracy")
    axes.plot(omegas, objective, color=SERIES_COLOURS[1], linewidth=LINE_WIDTH,
              marker="o", markersize=5, markeredgecolor=SURFACE,
              markeredgewidth=RING_WIDTH,
              label=f"Objective: accuracy - {lam:g} x complexity")

    axes.scatter([chosen.omega], [chosen.objective(lam)], s=MARKER_SIZE * 3,
                 facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
    axes.annotate("selected", (chosen.omega, chosen.objective(lam)),
                  textcoords="offset points", xytext=(0, -18), ha="center",
                  va="top", color=INK, fontsize=9)

    _style_axes(axes, f"Complexity: {chosen.omega_label}", "Score")
    _legend(axes)
    return _save(figure, filename)


def save_confusion(evaluation, filename):
    """Predicted against actual species for the selected model."""
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
    bar.set_label("penguins", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_edgecolor(GRID)

    _style_axes(axes, "Predicted", "Actual", grid=False)
    return _save(figure, filename)


def save_effect(curves, feature, classes, kind, filename, counts=None):
    """One curve per species for a PDP or an ALE."""
    figure, axes = _new_axes(width=8, height=4.4)

    x = curves["x"]
    for name in classes:
        axes.plot(x, curves["curves"][name], color=species_colour(classes, name),
                  linewidth=LINE_WIDTH, label=name)

    if kind == "ale":
        axes.axhline(0, color=GRID, linewidth=1, zorder=0)
        ylabel = "Change in predicted probability"
    else:
        ylabel = "Average predicted probability"

    # Where the data actually is, so nobody reads a curve drawn over thin air.
    if counts is not None and len(counts) == len(x) - 1:
        low = axes.get_ylim()[0]
        for index, count in enumerate(counts):
            if count == 0:
                continue
            axes.plot([x[index], x[index + 1]], [low, low], color=GRID,
                      linewidth=3, solid_capstyle="butt", zorder=0)

    axes.set_xlim(min(x), max(x))
    _style_axes(axes, readable(feature), ylabel)
    _legend(axes, title="Species")
    return _save(figure, filename)
