"""Charts for the uploaded dataset: scatter, distribution and correlation.

There is no way to hand a matplotlib figure straight to a browser, so the figure is
written into MEDIA_ROOT and the template loads it as an `<img>` - the same approach
as the `demos` app.
"""

import os
from collections import namedtuple

import matplotlib

# Must be selected before pyplot is imported: there is no GUI on the server.
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from django.conf import settings

SCATTER = "scatter"
DISTRIBUTION = "distribution"
CORRELATION = "correlation"

CHART_TYPES = (
    (SCATTER, "Scatter"),
    (DISTRIBUTION, "Distribution"),
    (CORRELATION, "Correlation matrix"),
)

# Categorical slots in fixed order - assigned by class, never cycled. A scatter
# puts every pair of classes on screen together, and only these four clear the
# all-pairs colour-vision and normal-vision separation floors on a white surface
# (checked with the palette validator; a fifth hue fails). Further classes fold
# into OTHER_COLOUR rather than inventing a colour that cannot be told apart.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
MAX_SERIES = len(SERIES_COLOURS)

# slate-400 - reads as "not one of the named classes" without competing with them.
OTHER_COLOUR = "#94a3b8"

# Single-hue ramp for a continuous target, light to dark. One hue on purpose: a
# multi-hue ramp (viridis and the like) misstates magnitude.
SEQUENTIAL_HUE = ("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b")

# Correlation runs -1..+1, which is polarity rather than magnitude: two opposed
# hues either side of a neutral grey, so "no correlation" reads as nothing.
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")

# Points sharing a category are spread over this fraction of the slot so that
# overlapping rows stay countable.
JITTER = 0.18

# Above these sizes the printed numbers collide, so the colourbar carries it alone.
MAX_LABELLED_CELLS = 144      # counts are short ("50")
MAX_CORRELATION_CELLS = 64    # correlations are five characters ("-0.43")

# Chart chrome, kept in slate so the data is the only loud thing on the page.
# These four are deliberately the same values as --app-ink, --app-muted and
# --app-line in static/style.css, so a figure and the page around it agree.
SURFACE = "#ffffff"
INK = "#0f172a"      # slate-900
MUTED = "#64748b"    # slate-500
GRID = "#e2e8f0"     # slate-200

MARKER_SIZE = 42     # ~6.5pt across, which clears the 8px floor at this dpi
RING_WIDTH = 1.0     # ring in the surface colour, so overlapping dots stay legible

# A figure is 8in wide, so 110dpi gave 880px while the card it sits in is about
# 1080px on a wide window: every chart was being upscaled by a quarter and came
# out soft. 150dpi gives 1200px, so the browser scales down instead of up, and
# the same files are sharper in the PDF reports at 16cm wide.
DPI = 150

# `folded` lists the classes that had to share the "Other" colour, so the
# interface can say so instead of quietly misrepresenting them.
PlotResult = namedtuple("PlotResult", ["url", "folded"])


def class_groups(target):
    """Classes to colour individually, and the tail that folds into "Other".

    Ordered by frequency so the largest classes keep their own colour.
    """
    counts = target.value_counts()
    return list(counts.index[:MAX_SERIES]), list(counts.index[MAX_SERIES:])


def save_chart(dataset, chart, x_name, y_name, filename):
    """Draw the requested chart and save it under MEDIA_ROOT.

    Returns a `PlotResult` holding the URL for the template's `<img>`.
    """
    figure, axes = plt.subplots(figsize=(8, 4.8))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    if chart == CORRELATION:
        folded = _draw_correlation(figure, axes, dataset)
    elif chart == DISTRIBUTION:
        folded = _draw_distribution(axes, dataset, x_name)
    else:
        folded = _draw_scatter(figure, axes, dataset, x_name, y_name)

    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    figure.tight_layout()
    figure.savefig(image_path, dpi=DPI, facecolor=SURFACE)
    # The server is long-lived, so figures have to be released explicitly.
    plt.close(figure)

    url = settings.MEDIA_URL + filename.replace(os.sep, "/")
    return PlotResult(url=url, folded=[str(label) for label in folded])


def save_scatter(dataset, x_name, y_name, filename):
    """Backwards-compatible entry point for the scatter chart."""
    return save_chart(dataset, SCATTER, x_name, y_name, filename)


def _draw_scatter(figure, axes, dataset, x_name, y_name):
    """`y_name` against `x_name`, coloured by the target."""
    x_categorical = dataset.is_categorical(x_name)
    y_categorical = dataset.is_categorical(y_name)

    if x_categorical and y_categorical:
        _draw_count_heatmap(figure, axes, dataset, x_name, y_name)
        return []

    x_values, x_ticks = _axis_values(dataset, x_name, jitter=x_categorical)
    y_values, y_ticks = _axis_values(dataset, y_name, jitter=y_categorical)
    folded = []

    if dataset.is_classification:
        folded = _draw_classes(axes, dataset, x_values, y_values)
    elif dataset.target_name in (x_name, y_name) or not dataset.target_is_numeric:
        # Either the target is already on an axis, so colouring by it would say
        # nothing new, or it is not numeric and cannot drive a colour scale.
        _draw_points(axes, x_values, y_values, colour=SERIES_COLOURS[0])
    else:
        ramp = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_HUE)
        points = _draw_points(axes, x_values, y_values, values=dataset.target, ramp=ramp)
        _add_colourbar(figure, axes, points, dataset.target_name)

    _style_axes(axes, x_name, y_name)
    _apply_category_ticks(axes, x_ticks, y_ticks)
    return folded


def _axis_values(dataset, name, jitter=False):
    """Values for one axis, mapping categories onto evenly spaced positions.

    Returns `(values, ticks)`; `ticks` is None for a numeric axis.
    """
    column = dataset.frame[name]
    if not jitter:
        return column, None

    categories = [value for value in column.dropna().unique()]
    categories.sort(key=str)
    positions = {value: index for index, value in enumerate(categories)}

    placed = column.map(positions).astype(float)
    # A fixed seed keeps the picture reproducible; the offsets must stay
    # uncorrelated with row order, or evenly spaced jitter draws diagonal streaks
    # that read as structure the data does not have.
    spread = np.random.default_rng(0).uniform(-JITTER, JITTER, size=len(column))
    return placed + spread, (range(len(categories)), [str(value) for value in categories])


def _apply_category_ticks(axes, x_ticks, y_ticks):
    if x_ticks:
        axes.set_xticks(list(x_ticks[0]))
        axes.set_xticklabels(x_ticks[1])
    if y_ticks:
        axes.set_yticks(list(y_ticks[0]))
        axes.set_yticklabels(y_ticks[1])


def _draw_count_heatmap(figure, axes, dataset, x_name, y_name):
    """Two categorical axes: show how many rows fall in each combination."""
    counts = dataset.frame.groupby([y_name, x_name], dropna=True).size().unstack(fill_value=0)
    ramp = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_HUE)
    values = counts.values

    image = axes.imshow(values, cmap=ramp, aspect="auto")
    axes.set_xticks(range(len(counts.columns)))
    axes.set_xticklabels([str(value) for value in counts.columns])
    axes.set_yticks(range(len(counts.index)))
    axes.set_yticklabels([str(value) for value in counts.index])

    # Colour alone would make the reader estimate a count off a colourbar.
    highest = float(values.max()) if values.size else 0.0
    _label_cells(axes, values,
                 weight=lambda value: (value / highest) if highest else 0.0,
                 text=lambda value: f"{int(value):,}",
                 max_cells=MAX_LABELLED_CELLS)

    _add_colourbar(figure, axes, image, "rows")
    _style_axes(axes, x_name, y_name, grid=False)


def _label_cells(axes, values, weight, text, max_cells):
    """Write each cell value into the grid, in ink or white by fill darkness.

    A label sitting inside a filled cell is the one place text may take its colour
    from the mark: it is chosen by the fill's luminance so it always stays legible.
    """
    rows, columns = values.shape
    if rows * columns > max_cells:
        return False

    for row in range(rows):
        for column in range(columns):
            value = values[row][column]
            shade = SURFACE if weight(value) > 0.55 else INK
            axes.text(column, row, text(value), ha="center", va="center",
                      color=shade, fontsize=8)
    return True


def _draw_distribution(axes, dataset, name):
    """How one feature is spread out, split by class where that applies."""
    if dataset.is_categorical(name):
        return _draw_category_counts(axes, dataset, name)

    values = dataset.frame[name].dropna()

    if dataset.is_classification:
        named, folded = class_groups(dataset.target)
        edges = np.histogram_bin_edges(values, bins="auto")
        for colour, label in zip(SERIES_COLOURS, named):
            rows = dataset.frame[dataset.frame[dataset.target_name] == label][name].dropna()
            axes.hist(rows, bins=edges, color=colour, alpha=0.75, label=str(label))
        if folded:
            rows = dataset.frame[dataset.frame[dataset.target_name].isin(folded)][name].dropna()
            axes.hist(rows, bins=edges, color=OTHER_COLOUR, alpha=0.75,
                      label=f"Other ({len(folded)} classes)")
        _add_legend(axes, dataset.target_name)
    else:
        axes.hist(values, bins="auto", color=SERIES_COLOURS[0], alpha=0.9)
        folded = []

    _style_axes(axes, name, "rows")
    return folded


def _draw_category_counts(axes, dataset, name):
    """A count per category - a histogram makes no sense for labels."""
    counts = dataset.frame[name].value_counts().sort_index(key=lambda index: index.map(str))
    positions = range(len(counts))

    axes.bar(positions, counts.values, color=SERIES_COLOURS[0], width=0.7)
    axes.set_xticks(list(positions))
    axes.set_xticklabels([str(value) for value in counts.index])

    _style_axes(axes, name, "rows")
    return []


def _draw_correlation(figure, axes, dataset):
    """Linear correlation between every pair of numeric columns."""
    numeric = dataset.frame[[name for name in dataset.frame.columns
                             if name in dataset.numeric_feature_names
                             or (name == dataset.target_name and dataset.target_is_numeric)]]
    matrix = numeric.corr()

    ramp = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    # Anchored at zero so the neutral midpoint always means "no relationship".
    image = axes.imshow(matrix.values, cmap=ramp, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))

    labels = [str(name) for name in matrix.columns]
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels)

    # Dark at both ends of a diverging ramp, so weight by magnitude.
    _label_cells(axes, matrix.values,
                 weight=lambda value: abs(value),
                 text=lambda value: f"{value:.2f}",
                 max_cells=MAX_CORRELATION_CELLS)

    _add_colourbar(figure, axes, image, "correlation")
    _style_axes(axes, "", "", grid=False)
    return []


def _add_colourbar(figure, axes, mappable, label):
    bar = figure.colorbar(mappable, ax=axes)
    bar.set_label(label, color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_edgecolor(GRID)
    return bar


def _add_legend(axes, title):
    """A legend is the dependable identity channel - never rely on colour alone."""
    legend = axes.legend(title=title, loc="best", fontsize=9,
                         frameon=True, framealpha=1, edgecolor=GRID)
    legend.get_title().set_fontsize(9)
    legend.get_title().set_color(MUTED)
    for text in legend.get_texts():
        text.set_color(INK)
    return legend


def _draw_classes(axes, dataset, x_values, y_values):
    """One colour per class, which is what makes class structure readable."""
    named, folded = class_groups(dataset.target)
    target = dataset.target

    for colour, label in zip(SERIES_COLOURS, named):
        rows = target == label
        _draw_points(axes, x_values[rows], y_values[rows],
                     colour=colour, label=str(label))

    if folded:
        rows = target.isin(folded)
        _draw_points(axes, x_values[rows], y_values[rows], colour=OTHER_COLOUR,
                     label=f"Other ({len(folded)} classes)")

    _add_legend(axes, dataset.target_name)
    return folded


def _draw_points(axes, x_values, y_values, colour=None, values=None, ramp=None,
                 label=None):
    """A scatter layer with the shared mark spec."""
    return axes.scatter(
        x_values, y_values,
        c=values if values is not None else colour,
        cmap=ramp,
        label=label,
        s=MARKER_SIZE,
        alpha=0.9,
        # A ring in the surface colour rather than a dark outline: it separates
        # overlapping dots without adding ink that is not data.
        edgecolors=SURFACE,
        linewidths=RING_WIDTH,
    )


def _style_axes(axes, x_name, y_name, grid=True):
    """Recessive chrome: hairline grid, muted text, no boxed-in frame."""
    axes.set_xlabel(x_name, color=MUTED, fontsize=10)
    axes.set_ylabel(y_name, color=MUTED, fontsize=10)
    axes.tick_params(colors=MUTED, labelsize=9, length=0)

    # Passing line properties alongside False turns the grid back on, so the
    # disabled case has to be a bare call.
    if grid:
        axes.grid(True, color=GRID, linewidth=1, linestyle="-")
    else:
        axes.grid(False)
    axes.set_axisbelow(True)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
