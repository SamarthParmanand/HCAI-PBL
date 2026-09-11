"""Project 2: Explainability.

All interface state lives in the query string rather than the session: the model
class, lambda, the counterfactual example and the chosen feature. That is what
makes the regions "linked" in the sense the task sheet asks for -- every region
reads the same model -- and it also means a particular view of the app can be
bookmarked, shared, or reloaded without surprises.
"""

import hashlib
import os

from django.conf import settings
from django.http import HttpResponse
from django.template import loader
from django.urls import reverse

from . import counterfactuals as cf
from . import effects, learning, plots, report
from .penguins import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                       class_names, readable, summary)

MEDIA_SUBDIR = "project2"

DEFAULT_FEATURE = "bill_length_mm"
MIN_K, MAX_K, DEFAULT_K = 1, 10, 5


def _clamp(value, low, high):
    return max(low, min(high, value))


def _read_state(request):
    """Everything the page is showing, parsed and validated from the query."""
    query = request.GET

    model_class = query.get("model")
    if model_class not in dict(learning.MODEL_CLASSES):
        model_class = learning.TREE

    try:
        lam = float(query.get("lam", learning.DEFAULT_LAMBDA))
    except (TypeError, ValueError):
        lam = learning.DEFAULT_LAMBDA
    lam = _clamp(lam, learning.LAMBDA_MIN, learning.LAMBDA_MAX)

    feature = query.get("feature")
    if feature not in NUMERIC_FEATURES:
        feature = DEFAULT_FEATURE

    try:
        k = int(query.get("cf_k", DEFAULT_K))
    except (TypeError, ValueError):
        k = DEFAULT_K
    k = _clamp(k, MIN_K, MAX_K)

    return {
        "model_class": model_class,
        "lam": lam,
        "feature": feature,
        "cf_k": k,
        "cf_index_raw": query.get("cf_index"),
        "cf_target_raw": query.get("cf_target"),
    }


def _token(*parts):
    """Short stable name for a figure, so identical states reuse the same file."""
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def _figure_paths(name):
    """(relative filename, absolute path) for a generated figure."""
    filename = os.path.join(MEDIA_SUBDIR, name)
    return filename, os.path.join(settings.MEDIA_ROOT, filename)


def _assemble(request):
    """Fit, select and draw everything the page and the report both need."""
    state = _read_state(request)
    data = learning.fit_family(state["model_class"])
    family = data["family"]
    frame = data["frame"]
    classes = data["classes"]

    model = learning.select(family, state["lam"])
    stamp = _token(state["model_class"], state["lam"], model.setting)

    # Task 1 on its own: one unconstrained tree, independent of the model class
    # and of lambda, so the card never changes as the rest of the page is driven.
    default_tree = learning.fit_default_tree()
    name, default_tree_path = _figure_paths("tree_task1.png")
    default_tree_url = plots.save_tree(default_tree, classes, name)

    # --- the selected model ------------------------------------------
    if model.model_class == learning.TREE:
        name, path = _figure_paths(f"tree_{stamp}.png")
        model_url = plots.save_tree(model, classes, name)
    else:
        name, path = _figure_paths(f"coefficients_{stamp}.png")
        model_url = plots.save_coefficients(model, classes, name)
    model_path = path

    name, trade_off_path = _figure_paths(f"tradeoff_{stamp}.png")
    trade_off_url = plots.save_trade_off(family, state["lam"], model, name)

    evaluation = learning.evaluation(model, data)
    name, _ = _figure_paths(f"confusion_{stamp}.png")
    confusion_url = plots.save_confusion(evaluation, name)

    # --- counterfactuals ---------------------------------------------
    examples = cf.example_choices(frame)
    valid_indexes = {entry["index"] for entry in examples}

    try:
        cf_index = int(state["cf_index_raw"])
    except (TypeError, ValueError):
        cf_index = examples[0]["index"] if examples else None
    if cf_index not in valid_indexes:
        cf_index = examples[0]["index"] if examples else None

    cf_target = state["cf_target_raw"]
    if cf_target not in classes:
        # Default to a class the penguin is not already predicted as.
        cf_target = None

    counterfactual_rows = []
    counterfactual_error = None
    counterfactual_search = None
    example_row = None
    predicted_species = None

    if cf_index is not None:
        example_row = frame.loc[cf_index, FEATURES]
        import pandas as pd
        predicted_species = str(model.pipeline.predict(
            pd.DataFrame([example_row], columns=FEATURES))[0])

        if cf_target is None:
            alternatives = [name for name in classes if name != predicted_species]
            cf_target = alternatives[0] if alternatives else classes[0]

        try:
            counterfactual_rows, counterfactual_search = cf.generate(
                model.pipeline, frame, example_row, cf_target, k=state["cf_k"])
        except cf.NoCounterfactual as failure:
            counterfactual_error = str(failure)

    # --- feature effects ---------------------------------------------
    feature = state["feature"]
    exact = model.model_class == learning.LOGREG

    pdp = effects.partial_dependence(model.pipeline, frame, feature, classes)
    name, pdp_path = _figure_paths(f"pdp_{stamp}_{feature}.png")
    pdp_url = plots.save_effect(pdp, feature, classes, "pdp", name)

    ale = effects.accumulated_local_effects(model.pipeline, frame, feature,
                                            classes, exact=exact)
    name, ale_path = _figure_paths(f"ale_{stamp}_{feature}.png")
    ale_url = plots.save_effect(ale, feature, classes, "ale", name,
                                counts=ale["counts"])

    return {
        "state": state,
        "data": data,
        "frame": frame,
        "classes": classes,
        "family": family,
        "model": model,
        "stamp": stamp,
        "model_url": model_url,
        "model_path": model_path,
        "default_tree": default_tree,
        "default_tree_url": default_tree_url,
        "default_tree_path": default_tree_path,
        "trade_off_url": trade_off_url,
        "trade_off_path": trade_off_path,
        "confusion_url": confusion_url,
        "evaluation": evaluation,
        "trade_off": learning.trade_off_table(family, state["lam"]),
        "switch_points": learning.switch_points(family),
        "examples": examples,
        "cf_index": cf_index,
        "cf_target": cf_target,
        "example_row": example_row,
        "predicted_species": predicted_species,
        "counterfactuals": counterfactual_rows,
        "counterfactual_error": counterfactual_error,
        "counterfactual_search": counterfactual_search,
        "feature": feature,
        "pdp": pdp,
        "pdp_url": pdp_url,
        "pdp_path": pdp_path,
        "ale": ale,
        "ale_url": ale_url,
        "ale_path": ale_path,
        "exact": exact,
    }


def index(request):
    """The whole interface: model choice, lambda, counterfactuals, effects."""
    assembled = _assemble(request)
    state = assembled["state"]
    model = assembled["model"]
    frame = assembled["frame"]

    example_values = []
    if assembled["example_row"] is not None:
        row = assembled["example_row"]
        for name in FEATURES:
            value = row[name]
            example_values.append({
                "label": readable(name),
                "value": f"{float(value):.1f}" if name in NUMERIC_FEATURES else str(value),
            })

    template = loader.get_template("project2/index.html")
    context = {
        "title": "Project 2 - Explainability",
        "model_classes": learning.MODEL_CLASSES,
        "model_class": state["model_class"],
        "model_label": learning.MODEL_LABELS[state["model_class"]],
        "model": model,
        "lam": state["lam"],
        "lambda_min": learning.LAMBDA_MIN,
        "lambda_max": learning.LAMBDA_MAX,
        "lambda_step": learning.LAMBDA_STEP,
        "omega_label": model.omega_label,
        "setting_label": ("max_leaf_nodes" if model.model_class == learning.TREE
                          else "C"),
        "model_url": assembled["model_url"],
        "default_tree": assembled["default_tree"],
        "default_tree_url": assembled["default_tree_url"],
        "trade_off_url": assembled["trade_off_url"],
        "confusion_url": assembled["confusion_url"],
        "evaluation": assembled["evaluation"],
        "trade_off": assembled["trade_off"],
        "switch_points": assembled["switch_points"],
        "coefficients": _coefficients_with_colour(model, assembled["classes"]),
        "rows": len(frame),
        "dropped_rows": frame.attrs.get("dropped_rows", 0),
        "total_rows": frame.attrs.get("total_rows", len(frame)),
        "summary": summary(frame),
        "classes": assembled["classes"],
        "examples": assembled["examples"],
        "cf_index": assembled["cf_index"],
        "cf_target": assembled["cf_target"],
        "cf_k": state["cf_k"],
        "example_values": example_values,
        "predicted_species": assembled["predicted_species"],
        "counterfactuals": assembled["counterfactuals"],
        "counterfactual_error": assembled["counterfactual_error"],
        "counterfactual_search": assembled["counterfactual_search"],
        "features": [(name, readable(name)) for name in NUMERIC_FEATURES],
        "feature": state["feature"],
        "feature_label": readable(state["feature"]),
        "pdp_url": assembled["pdp_url"],
        "ale_url": assembled["ale_url"],
        "ale_method": assembled["ale"]["method"],
        "exact": assembled["exact"],
        "report_query": request.GET.urlencode(),
        # The lambda values where the selection changes, offered as tick marks on
        # the slider and as one-click jumps: otherwise they can only be found by
        # dragging and watching.
        "jump_points": _jump_points(request, assembled),
    }
    return HttpResponse(template.render(context, request))


def _jump_points(request, assembled):
    """Each switch point, with the query string that selects it."""
    query = request.GET.copy()
    points = []
    for point in assembled["switch_points"]:
        query["lam"] = f"{point['lam']:.4f}"
        points.append({
            "lam": point["lam"],
            "setting": point["setting"],
            "omega": point["omega"],
            "accuracy_test": point["accuracy_test"],
            "query": query.urlencode(),
            "current": abs(point["lam"] - assembled["state"]["lam"]) < 1e-9,
        })
    return points


def _coefficients_with_colour(model, classes):
    """Coefficient rows carrying the species colour used in every chart.

    The swatch beside the name is what keeps identity off colour alone.
    """
    rows = learning.coefficient_table(model)[:14]
    for row in rows:
        row["colour"] = plots.species_colour(classes, row["species"])
    return rows


def download_report(request):
    """The PDF report, describing exactly the state the interface is in."""
    assembled = _assemble(request)
    model = assembled["model"]
    frame = assembled["frame"]

    summary_line = ""
    if assembled["counterfactuals"]:
        summary_line = (
            f"Penguin #{assembled['cf_index']} is predicted as "
            f"{assembled['predicted_species']}; the rows below are the closest "
            f"variations the model would call {assembled['cf_target']}.")
    elif assembled["counterfactual_error"]:
        summary_line = assembled["counterfactual_error"]

    effects_figures = [
        {"path": assembled["pdp_path"],
         "caption": (f"PDP for {readable(assembled['feature'])}: the average "
                     f"predicted probability of each species as the feature is "
                     f"swept across its range.")},
        {"path": assembled["ale_path"],
         "caption": (f"ALE for {readable(assembled['feature'])}, computed by "
                     f"{assembled['ale']['method']}. The shaded strip along the "
                     f"bottom marks the bins that contain data.")},
    ]

    context = {
        "model": model,
        "family": assembled["family"],
        "lam": assembled["state"]["lam"],
        "trade_off": assembled["trade_off"],
        "switch_points": assembled["switch_points"],
        "trade_off_path": assembled["trade_off_path"],
        "model_path": assembled["model_path"],
        "default_tree": assembled["default_tree"],
        "default_tree_path": assembled["default_tree_path"],
        "counterfactuals": assembled["counterfactuals"],
        "counterfactual_summary": summary_line,
        "effects": effects_figures,
        "rows": len(frame),
        "dropped_rows": frame.attrs.get("dropped_rows", 0),
        "total_rows": frame.attrs.get("total_rows", len(frame)),
        "test_size": learning.DEFAULT_TEST_SIZE,
        "seed": learning.DEFAULT_SEED,
        "reading": _reading(assembled),
    }

    pdf = report.build(context)
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        'attachment; filename="HCAI-project2-report.pdf"')
    return response


def _reading(assembled):
    """A short, honest reading of the current curves for the report."""
    model = assembled["model"]
    feature = readable(assembled["feature"])
    method = assembled["ale"]["method"]

    if model.model_class == learning.TREE:
        shape = ("The tree's curves move in steps: it is piecewise constant, so a "
                 "feature only matters where a split falls, and both curves are "
                 "flat between splits.")
    else:
        shape = ("The logistic curves are smooth and monotone in each class, "
                 "because the model is a softmax of a linear function of the "
                 "features.")

    return (
        f"{shape} The PDP and the ALE for {feature} agree on the direction of the "
        f"effect but not on its size: the PDP averages over combinations of "
        f"features that do not occur in the data, while the ALE only ever uses "
        f"rows that really fall in each bin, so where the four measurements are "
        f"strongly correlated the PDP is the more optimistic of the two. The ALE "
        f"here was computed by {method}. At the current setting the model has "
        f"{model.omega} {model.omega_label} and scores "
        f"{model.accuracy_test:.3f} on the test set.")
