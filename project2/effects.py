"""Feature-effect curves for Project 2, Task 5: PDP and ALE, written by hand.

No explainability library is used; everything here is computed from the model's
own predictions (or, where possible, its coefficients).

**PDP.** For a feature j and a grid value v, every row in the data is copied with
x_j replaced by v and the predicted class probabilities are averaged:

    PDP_c(v) = (1/n) sum_i  p_c(v, x_i,-j)

That average runs over the *marginal* distribution of the other features, which is
what makes a PDP easy to read and also what makes it lie when features are
correlated: it evaluates the model on combinations that never occur, such as a
short bill on a heavy Gentoo.

**ALE.** ALE answers the same question without inventing rows. The feature range
is cut into bins at its own quantiles, and inside each bin only the rows that
actually fall there are used. Their *local* change in prediction across the bin is
averaged, and those averages are accumulated across bins:

    Delta_k = mean over rows in bin k of [ f(z_k, x_-j) - f(z_k-1, x_-j) ]
    ALE(z_K) = sum_{k <= K} Delta_k,   then centred to average zero.

**The derivative question the task sheet asks.** The local effect in a bin is an
integral of the partial derivative of the prediction with respect to the feature:

* For **logistic regression** the derivative is available *exactly*, in closed
  form. The prediction is a softmax of a linear function, so with
  eta_c = w_c . z + b_c and z the standardised features,

      d eta_c / d x_j  =  w_c,j / scale_j
      d p_c   / d x_j  =  p_c * ( d eta_c/d x_j  -  sum_k p_k * d eta_k/d x_j )

  `exact_gradient` implements exactly that, and the tests check it against a
  central finite difference of `predict_proba`.

* For a **decision tree** it cannot be. A tree is piecewise constant: its
  derivative is zero everywhere inside a leaf and undefined at the split points,
  so it carries no information about the effect. The only workable route is the
  discretised one, taking a finite difference across each bin, which is what
  `_discrete_local_effects` does.

Both paths share the binning, accumulation and centring below, so the two curves
are directly comparable.
"""

import numpy as np
import pandas as pd

from .penguins import NUMERIC_FEATURES

PDP_GRID = 40
ALE_BINS = 20


# --- shared helpers -------------------------------------------------------

def _class_columns(pipeline, classes):
    """Map species name to its column in `predict_proba`."""
    order = list(pipeline.classes_)
    return [order.index(name) for name in classes]


def _proba(pipeline, frame, columns):
    """Predicted probabilities, restricted and reordered to `columns`."""
    return np.asarray(pipeline.predict_proba(frame))[:, columns]


def _replace(frame, feature, value):
    """A copy of the data with one feature pinned to `value`."""
    copy = frame.copy()
    copy[feature] = value
    return copy


# note to self: ale needs bins before anything else, so this is where the
# binning lives. quantile edges rather than equal width, so every bin holds
# roughly the same number of rows and no bin ends up estimating an effect from
# two points. duplicate edges are dropped, which is why the realised number of
# bins can come out below what was asked for.
def bin_edges(values, bins=ALE_BINS):
    """Bin boundaries at the feature's own quantiles.

    Quantiles rather than an even split: they put the boundaries where the data
    actually is, so no bin ends up empty and the local effects are all estimated
    from real rows.
    """
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(np.asarray(values, dtype=float), quantiles))
    if len(edges) < 2:
        raise ValueError("This feature does not vary, so it has no effect to show.")
    return edges


def _assign_bins(values, edges):
    """Index of the bin each row falls in, from 1 to len(edges) - 1."""
    raw = np.searchsorted(edges, np.asarray(values, dtype=float), side="left")
    return np.clip(raw, 1, len(edges) - 1)


# note to self: the accumulate and centre step of ale. accumulating is a
# cumulative sum of the per bin effects, and centring subtracts the count
# weighted mean so the curve reads as a deviation from the average prediction
# and is zero on average over the rows i actually have.
def _accumulate_and_centre(local_effects, counts, edges):
    """Turn per-bin effects into a centred curve over the bin boundaries.

    `local_effects` is (n_bins, n_classes). The curve starts at zero on the first
    boundary and accumulates; it is then shifted so that its average over the data
    is zero, which is what makes ALE readable as a deviation from the mean
    prediction rather than an arbitrary offset.
    """
    accumulated = np.vstack([
        np.zeros((1, local_effects.shape[1])),
        np.cumsum(local_effects, axis=0),
    ])

    total = counts.sum()
    if total > 0:
        # Average of the piecewise-linear curve under the data distribution.
        midpoints = (accumulated[:-1] + accumulated[1:]) / 2.0
        mean = (counts[:, None] * midpoints).sum(axis=0) / total
    else:                                                  # pragma: no cover
        mean = np.zeros(local_effects.shape[1])

    return accumulated - mean


# --- partial dependence ---------------------------------------------------

# note to self: the pdp, written by hand because the sheet says no library. the
# whole idea in one sentence: for each value v on the grid, overwrite the chosen
# column with v for every single row, predict, and average the probabilities,
# giving one curve per species. the weakness i should volunteer before being
# asked is that it averages over feature combinations that never occur in the
# data, for instance a gentoo sized bill on an adelie sized body, and that is
# exactly where it parts company with ale.
def partial_dependence(pipeline, frame, feature, classes, grid_size=PDP_GRID):
    """One curve per class: the average predicted probability as x_j is swept."""
    if feature not in NUMERIC_FEATURES:
        raise ValueError(f"{feature} is not one of the numerical features.")

    columns = _class_columns(pipeline, classes)
    values = frame[feature].astype(float)
    grid = np.linspace(float(values.min()), float(values.max()), grid_size)

    curves = np.zeros((len(grid), len(classes)))
    for position, value in enumerate(grid):
        probabilities = _proba(pipeline, _replace(frame, feature, value), columns)
        curves[position] = probabilities.mean(axis=0)

    return {"x": grid.tolist(),
            "curves": {name: curves[:, index].tolist()
                       for index, name in enumerate(classes)}}


# --- accumulated local effects -------------------------------------------

def _discrete_local_effects(pipeline, frame, feature, columns, edges, assignment):
    """Finite difference across each bin, averaged over the rows inside it.

    This is the only route open for a decision tree, whose exact derivative is
    zero almost everywhere and undefined at its split points.
    """
    n_bins = len(edges) - 1
    effects = np.zeros((n_bins, len(columns)))
    counts = np.zeros(n_bins)

    for index in range(1, n_bins + 1):
        rows = frame[assignment == index]
        counts[index - 1] = len(rows)
        if rows.empty:
            continue

        upper = _proba(pipeline, _replace(rows, feature, edges[index]), columns)
        lower = _proba(pipeline, _replace(rows, feature, edges[index - 1]), columns)
        effects[index - 1] = (upper - lower).mean(axis=0)

    return effects, counts


# note to self: this is the "exact for one model class" half of task 5 and the
# single thing i am most likely to be asked to derive on the spot.
# with z = w x + b and p = softmax(z), differentiating gives
#     d p_c / d x_j = p_c * (w_cj - sum_k p_k * w_kj)
# which comes straight out of the quotient rule on exp(z_c) / sum_k exp(z_k),
# using d z_k / d x_j = w_kj. a good sanity check to mention: the three
# derivatives sum to zero across classes, because the probabilities sum to one.
# the subtlety worth remembering is that the coefficient belongs to the
# standardised column while the derivative i want is with respect to the
# original feature, so it has to be divided by the scaler's scale. get that
# wrong and the magnitude is off while the shape still looks plausible, which is
# the worst kind of bug to find late.
def exact_gradient(pipeline, frame, feature, classes):
    """d p_c / d x_j for every row, computed in closed form.

    Only valid for the softmax logistic model: it reads the fitted coefficients
    and undoes the scaler by the chain rule.
    """
    from .learning import numeric_column_index

    model = pipeline.named_steps["model"]
    prepare = pipeline.named_steps["prepare"]
    index = numeric_column_index(pipeline, feature)

    scaler = prepare.named_transformers_["numeric"]
    scale = float(scaler.scale_[NUMERIC_FEATURES.index(feature)])

    # d eta_c / d x_j for each class, in the estimator's own class order.
    slopes = np.asarray(model.coef_)[:, index] / scale

    probabilities = np.asarray(pipeline.predict_proba(frame))
    weighted = probabilities @ slopes                      # sum_k p_k * slope_k
    gradients = probabilities * (slopes[None, :] - weighted[:, None])

    order = list(model.classes_)
    columns = [order.index(name) for name in classes]
    return gradients[:, columns]


def _exact_local_effects(pipeline, frame, feature, classes, edges, assignment):
    """Local effects from the analytic derivative, integrated across each bin.

    Inside a bin the effect is the integral of the derivative over the bin's
    width; the derivative is evaluated at the bin's midpoint, on the rows that
    genuinely fall in that bin.
    """
    n_bins = len(edges) - 1
    effects = np.zeros((n_bins, len(classes)))
    counts = np.zeros(n_bins)

    for index in range(1, n_bins + 1):
        rows = frame[assignment == index]
        counts[index - 1] = len(rows)
        if rows.empty:
            continue

        width = float(edges[index] - edges[index - 1])
        midpoint = float((edges[index] + edges[index - 1]) / 2.0)
        gradients = exact_gradient(pipeline, _replace(rows, feature, midpoint),
                                   feature, classes)
        effects[index - 1] = width * gradients.mean(axis=0)

    return effects, counts


# note to self: ale, also by hand. the difference from pdp in one sentence: ale
# only ever uses rows that really fall in each bin, so it never invents a
# combination of features that does not exist. the procedure is bin the feature,
# take the local effect inside each bin (the exact gradient above for logistic
# regression, a finite difference across the bin for the tree), accumulate
# across bins, then centre on the data. with correlated features, and the four
# penguin measurements are heavily correlated, pdp comes out the more optimistic
# of the two, which is the comparison the sheet is really asking for.
def accumulated_local_effects(pipeline, frame, feature, classes, exact=False,
                              bins=ALE_BINS):
    """One centred ALE curve per class.

    `exact=True` uses the closed-form derivative (logistic regression only);
    otherwise a finite difference is taken across each bin.
    """
    if feature not in NUMERIC_FEATURES:
        raise ValueError(f"{feature} is not one of the numerical features.")

    values = frame[feature].astype(float)
    edges = bin_edges(values, bins)
    assignment = _assign_bins(values, edges)

    if exact:
        effects, counts = _exact_local_effects(
            pipeline, frame, feature, classes, edges, assignment)
    else:
        columns = _class_columns(pipeline, classes)
        effects, counts = _discrete_local_effects(
            pipeline, frame, feature, columns, edges, assignment)

    curves = _accumulate_and_centre(effects, counts, edges)

    return {
        "x": edges.tolist(),
        "curves": {name: curves[:, index].tolist()
                   for index, name in enumerate(classes)},
        "counts": counts.astype(int).tolist(),
        "exact": bool(exact),
        "method": ("closed-form derivative" if exact
                   else "finite difference across each bin"),
    }


# note to self: and this is the "discretised for the other" half. a decision
# tree is piecewise constant, so its derivative is zero almost everywhere and
# undefined exactly on the split points. there is no closed form to have, which
# is the actual answer to "which class did you do exactly and why", and a small
# step finite difference is the honest substitute. it also explains why the
# tree's curves come out as steps that are flat between splits.
def numerical_gradient(pipeline, frame, feature, classes, step=1e-4):
    """Central finite difference of predict_proba, for checking the exact one."""
    columns = _class_columns(pipeline, classes)
    values = frame[feature].astype(float)

    forward = _proba(pipeline, _replace(frame, feature, values + step), columns)
    backward = _proba(pipeline, _replace(frame, feature, values - step), columns)
    return (forward - backward) / (2 * step)
