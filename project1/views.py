import os
import zlib

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template import loader
from django.urls import reverse
from django.utils.text import get_valid_filename

from . import plots, training, training_plots
from .dataset import PROBLEM_TYPES, load_dataset
from .forms import CSVUploadForm

# Uploaded CSVs and generated figures both live under MEDIA_ROOT/project1/.
MEDIA_SUBDIR = "project1"

# The uploaded dataset has to survive across requests so the user can change the
# plot without re-uploading. The file sits in media; the session just remembers it.
SESSION_KEY = "project1_dataset"

# Shipped with the repo, offered so the interface can be tried without an upload.
SAMPLE_CSV = os.path.join(settings.BASE_DIR, "iris.csv")


def index(request):
    """Upload a CSV, or load the bundled Iris sample."""
    error = None

    if request.method == "POST":
        if "use_sample" in request.POST:
            if os.path.exists(SAMPLE_CSV):
                with open(SAMPLE_CSV, "rb") as sample:
                    error = _accept_dataset(request, sample, "iris.csv")
                if error is None:
                    return redirect("project1:explore")
            else:
                error = "The bundled iris.csv is missing from the project root."
            form = CSVUploadForm()
        else:
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded = request.FILES["file"]
                error = _accept_dataset(request, uploaded, uploaded.name)
                if error is None:
                    return redirect("project1:explore")
    else:
        form = CSVUploadForm()

    template = loader.get_template("project1/index.html")
    context = {
        "title": "Project 1 – Supervised Learning Interface",
        "form": form,
        "error": error,
        "sample_available": os.path.exists(SAMPLE_CSV),
    }
    return HttpResponse(template.render(context, request))


def explore(request):
    """Show the loaded dataset and a scatter plot the user controls."""
    state = request.session.get(SESSION_KEY)
    if not state:
        return redirect("project1:index")

    if not default_storage.exists(state["path"]):
        # Media was cleared out from under us; start over rather than 500.
        _forget_dataset(request)
        return redirect("project1:index")

    if request.method == "POST":
        # A control that is absent from the POST - disabled, or on a form that does
        # not carry it - must leave the stored choice alone. Reading absence as
        # "reset to default" silently threw away the user's decision.
        state["problem_type"] = (_chosen(request.POST, "problem_type", PROBLEM_TYPES)
                                 or state.get("problem_type"))
        state["chart"] = (_chosen(request.POST, "chart",
                                  [key for key, _ in plots.CHART_TYPES])
                          or state.get("chart"))
        state["x"] = request.POST.get("x") or state.get("x")
        state["y"] = request.POST.get("y") or state.get("y")

        # Looking at the same pair the other way round is a common thing to want
        # and fiddly to do with two selects.
        if "swap" in request.POST:
            state["x"], state["y"] = state.get("y"), state.get("x")

        target = request.POST.get("target") or state.get("target")
        if target != state.get("target"):
            # The override applied to the previous target; re-detect for the new one.
            state["problem_type"] = None
        state["target"] = target
        # A checkbox genuinely does mean "off" when absent, but only on a POST that
        # carried the control in the first place.
        if "controls" in request.POST:
            state["keep_ids"] = "keep_ids" in request.POST

    try:
        with default_storage.open(state["path"], "rb") as stored:
            dataset = load_dataset(
                stored,
                problem_type=state.get("problem_type"),
                keep_id_columns=state.get("keep_ids", False),
                target_name=state.get("target"),
            )
    except ValueError as invalid:
        _forget_dataset(request)
        template = loader.get_template("project1/index.html")
        context = {
            "title": "Project 1 – Supervised Learning Interface",
            "form": CSVUploadForm(),
            "error": str(invalid),
            "sample_available": os.path.exists(SAMPLE_CSV),
        }
        return HttpResponse(template.render(context, request))

    # Axis choices depend on the problem type, so validate them against the
    # dataset we actually ended up with rather than trusting the posted values.
    choices = dataset.axis_choices()
    default_x, default_y = dataset.default_axes()
    x_name = _chosen({"x": state.get("x")}, "x", choices) or default_x
    y_name = _chosen({"y": state.get("y")}, "y", choices) or default_y
    state["x"], state["y"] = x_name, y_name
    request.session[SESSION_KEY] = state

    chart = state.get("chart") or plots.SCATTER
    if chart == plots.CORRELATION and len(dataset.numeric_feature_names) < 2:
        # Nothing to correlate; fall back rather than draw an empty grid.
        chart = plots.SCATTER
    state["chart"] = chart

    image_url = None
    plot_error = None
    folded = []
    if chart == plots.CORRELATION:
        # The correlation grid ignores the x/y choice: it covers every numeric
        # column, and `DataFrame.corr` computes each cell from the rows where
        # that pair is present. So there is no single row count to quote, and
        # quoting the one for x and y would describe a different figure.
        kept, skipped = None, None
    else:
        plotted_columns = ([x_name] if chart == plots.DISTRIBUTION
                           else [x_name, y_name])
        kept, skipped = dataset.rows_plotted(plotted_columns)

    if x_name and y_name:
        filename = os.path.join(MEDIA_SUBDIR, f"chart_{_session_token(request)}.png")
        drawn = plots.save_chart(dataset, chart, x_name, y_name, filename)
        folded = drawn.folded
        # Same filename every time, so tell the browser when the content changed.
        image_url = f"{drawn.url}?v={_cache_token(state)}"
    else:
        plot_error = "This dataset has no numeric columns to plot."

    columns, rows = dataset.preview()

    template = loader.get_template("project1/explore.html")
    context = {
        "title": "Explore the dataset",
        "filename": state["name"],
        "dataset": dataset,
        "axis_choices": choices,
        "x_name": x_name,
        "y_name": y_name,
        "keep_ids": state.get("keep_ids", False),
        "preview_columns": columns,
        "preview_rows": rows,
        "summary": dataset.summary(),
        "class_table": _class_table(dataset) if dataset.is_classification else [],
        "folded_classes": folded,
        "max_series": plots.MAX_SERIES,
        "image_url": image_url,
        "plot_error": plot_error,
        "chart": chart,
        "chart_types": plots.CHART_TYPES,
        "column_names": list(dataset.frame.columns),
        "correlation_available": len(dataset.numeric_feature_names) >= 2,
        "same_axes": bool(x_name and x_name == y_name and chart == plots.SCATTER),
        "rows_plotted": kept,
        "rows_skipped": skipped,
        "explanations": dataset.explanations(),
        "audit": dataset.audit(),
        "download_url": reverse("project1:download"),
    }
    return HttpResponse(template.render(context, request))


def train(request):
    """Task 4: split, sweep a hyperparameter, score, and report the best model."""
    state = request.session.get(SESSION_KEY)
    if not state or not default_storage.exists(state["path"]):
        return redirect("project1:index")

    try:
        dataset = _load_from_session(state)
    except ValueError:
        _forget_dataset(request)
        return redirect("project1:index")

    choices = training.model_choices(dataset.problem_type)
    keys = [key for key, _ in choices]

    if request.method == "POST":
        state["model"] = _chosen(request.POST, "model", keys) or state.get("model")
        state["test_size"] = _as_fraction(request.POST.get("test_size"),
                                          state.get("test_size"))
        state["seed"] = _as_seed(request.POST.get("seed"), state.get("seed"))
        request.session[SESSION_KEY] = state

    model_key = state.get("model") if state.get("model") in keys else (keys[0] if keys else None)
    test_size = state.get("test_size") or training.DEFAULT_TEST_SIZE
    seed = state.get("seed") if state.get("seed") is not None else training.RANDOM_SEED

    result = None
    error = None
    sweep_url = confusion_url = predictions_url = None

    if model_key is None:
        error = "No model is available for this problem type."
    else:
        try:
            result = training.train(dataset, model_key, test_size=test_size, seed=seed)
        except training.TrainingError as failure:
            error = str(failure)

    if result is not None:
        token = _session_token(request)
        stamp = _cache_token({**state, "model": model_key,
                              "test_size": test_size, "seed": seed})
        base = os.path.join(MEDIA_SUBDIR, f"sweep_{token}.png")
        sweep_url = f"{training_plots.save_sweep(result, base)}?v={stamp}"

        detail = result["detail"]
        if detail["kind"] == "classification":
            name = os.path.join(MEDIA_SUBDIR, f"confusion_{token}.png")
            confusion_url = f"{training_plots.save_confusion(detail, name)}?v={stamp}"
        else:
            name = os.path.join(MEDIA_SUBDIR, f"predictions_{token}.png")
            predictions_url = f"{training_plots.save_predictions(detail, name)}?v={stamp}"

    template = loader.get_template("project1/train.html")
    context = {
        "title": "Train a model",
        "filename": state["name"],
        "dataset": dataset,
        "result": result,
        "error": error,
        "model_choices": choices,
        "model_key": model_key,
        "test_size": test_size,
        "test_percent": int(round(test_size * 100)),
        "seed": seed,
        "sweep_url": sweep_url,
        "confusion_url": confusion_url,
        "predictions_url": predictions_url,
        "audit": dataset.audit(),
    }
    return HttpResponse(template.render(context, request))


def download(request):
    """Hand back the parsed dataset - the columns as the app actually read them."""
    state = request.session.get(SESSION_KEY)
    if not state or not default_storage.exists(state["path"]):
        return redirect("project1:index")

    try:
        with default_storage.open(state["path"], "rb") as stored:
            dataset = load_dataset(
                stored,
                problem_type=state.get("problem_type"),
                keep_id_columns=state.get("keep_ids", False),
            )
    except ValueError:
        return redirect("project1:explore")

    name = get_valid_filename(f"parsed_{state['name']}") or "parsed.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"

    response = HttpResponse(dataset.frame.to_csv(index=False), content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{name}"'
    return response


def _load_from_session(state):
    """Rebuild the dataset described by the session state."""
    with default_storage.open(state["path"], "rb") as stored:
        return load_dataset(
            stored,
            problem_type=state.get("problem_type"),
            keep_id_columns=state.get("keep_ids", False),
            target_name=state.get("target"),
        )


def _as_fraction(raw, fallback):
    """A test-set size the user typed, clamped to something trainable."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback if fallback is not None else training.DEFAULT_TEST_SIZE
    if value > 1:            # accept "25" as well as "0.25"
        value = value / 100
    return min(max(value, training.MIN_TEST_SIZE), training.MAX_TEST_SIZE)


def _as_seed(raw, fallback):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback if fallback is not None else training.RANDOM_SEED


def _class_table(dataset):
    """Class rows annotated with the colour the plot actually gave them."""
    named, _ = plots.class_groups(dataset.target)
    colours = dict(zip([str(label) for label in named], plots.SERIES_COLOURS))

    table = []
    for name, count in dataset.class_counts():
        colour = colours.get(name)
        table.append({
            "name": name,
            "count": count,
            "colour": colour or plots.OTHER_COLOUR,
            "shown_as": name if colour else "Other",
        })
    return table


def _accept_dataset(request, file_object, display_name):
    """Validate an incoming CSV and remember it. Returns an error string or None."""
    try:
        load_dataset(file_object)
    except ValueError as invalid:
        return str(invalid)
    except Exception as unexpected:
        return f"That file could not be read: {unexpected}"

    file_object.seek(0)
    _forget_dataset(request)
    # Never build a storage path out of a client-supplied name as-is.
    safe_name = get_valid_filename(os.path.basename(display_name)) or "dataset.csv"
    path = default_storage.save(f"{MEDIA_SUBDIR}/{safe_name}", file_object)
    request.session[SESSION_KEY] = {
        "path": path,
        "name": display_name,
        "problem_type": None,
        "keep_ids": False,
        "x": None,
        "y": None,
        "target": None,
        "chart": None,
    }
    return None


def _forget_dataset(request):
    """Drop the remembered dataset, deleting the stored upload with it."""
    state = request.session.pop(SESSION_KEY, None)
    if state and default_storage.exists(state["path"]):
        default_storage.delete(state["path"])


def _chosen(source, key, allowed):
    """Return `source[key]` only if it is one of `allowed` - never trust input."""
    value = source.get(key)
    return value if value in allowed else None


def _session_token(request):
    """A per-session filename fragment, so concurrent users do not overwrite."""
    if request.session.session_key is None:
        request.session.save()
    return request.session.session_key[:8]


def _cache_token(state):
    """Short, deterministic token that changes when the plot settings change."""
    keys = ("path", "problem_type", "keep_ids", "x", "y", "target", "chart")
    signature = "|".join(str(state.get(key)) for key in keys)
    return zlib.crc32(signature.encode("utf-8"))
