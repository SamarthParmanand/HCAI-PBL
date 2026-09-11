"""Project 3: Active Learning for Learning-to-Defer.

The experiment takes about two minutes on the full 120,000 articles, so it is not
run per request. `python manage.py build_project3` runs it once and caches the
result; these views read the cache. If it has not been built the page says so and
gives the command rather than hanging.
"""

import os

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template import loader
from django.urls import reverse

from . import artifacts, classifier, data, experts, plots, report

MEDIA_SUBDIR = "project3"

SESSION_KEY = "project3_expert_round"
ROUND_SIZE = 8


def _figures(bundle):
    """Draw every figure for the cached bundle and return url + path for each."""
    tag = f"{bundle['expert_kind']}_{bundle['seed']}_{bundle['limit'] or 'full'}"

    def paths(name):
        filename = os.path.join(MEDIA_SUBDIR, f"{name}_{tag}.png")
        return filename, os.path.join(settings.MEDIA_ROOT, filename)

    drawn = {}

    name, path = paths("accuracy")
    drawn["accuracy"] = (plots.save_accuracy_comparison(bundle, name), path)

    name, path = paths("deferral")
    drawn["deferral"] = (plots.save_deferral_curves(bundle, name), path)

    name, path = paths("active")
    drawn["active"] = (plots.save_active_learning(bundle, name), path)

    name, path = paths("active_auc")
    drawn["active_auc"] = (
        plots.save_active_learning(bundle, name, metric="expert_model_auc",
                                   ylabel="Expert-competence model AUC"), path)

    name, path = paths("confusion")
    drawn["confusion"] = (plots.save_confusion(bundle["system"], name), path)

    name, path = paths("expert_profile")
    drawn["expert_profile"] = (plots.save_expert_profile(bundle, name), path)

    return drawn


def _not_built(request, message=None):
    template = loader.get_template("project3/not_built.html")
    context = {
        "message": message,
        "command": "python manage.py build_project3",
        "data_cached": data.is_cached(),
    }
    return HttpResponse(template.render(context, request))


def index(request):
    """Tasks 1 to 4, read from the cached experiment."""
    bundle = artifacts.load()
    if bundle is None:
        return _not_built(request)

    drawn = _figures(bundle)

    # Ordered for display: the strategy actually proposed first.
    strategies = [
        {"key": key, **row} for key, row in bundle["active_summary"].items()
    ]

    template = loader.get_template("project3/index.html")
    context = {
        "bundle": bundle,
        "dataset": bundle["dataset"],
        "split": bundle["split"],
        "baseline": bundle["baseline"],
        "system": bundle["system"],
        "expert": bundle["expert"],
        "quality": bundle["competence_quality"],
        "policies": bundle["policies"],
        "policy_rows": [
            {"label": "Classifier alone", **bundle["policies"]["classifier_only"]},
            {"label": "Expert alone", **bundle["policies"]["expert_only"]},
            {"label": "Confidence threshold, tuned in hindsight",
             **bundle["policies"]["confidence_best"]},
            {"label": "Competence rule (margin 0)",
             **bundle["policies"]["competence_zero_margin"]},
            {"label": "Competence rule, margin tuned in hindsight",
             **bundle["policies"]["competence_best"]},
            {"label": "Competence rule, text model",
             **bundle["policies"]["competence_text_zero_margin"]},
            {"label": "Oracle ceiling", **bundle["policies"]["oracle"]},
        ],
        "strategies": strategies,
        "active_target": bundle["active_target"],
        "active_settings": bundle["active_settings"],
        "class_names": data.CLASS_NAMES,
        "per_class": [
            {
                "name": name,
                "expert": bundle["expert"]["per_class"][name]["accuracy"],
                "classifier": bundle["system"]["per_class"][name]["recall"],
                "n": bundle["system"]["per_class"][name]["n"],
            }
            for name in data.CLASS_NAMES
        ],
        "figures": {key: url for key, (url, _path) in drawn.items()},
        "n_deferred": len(bundle.get("deferred_test_indices", [])),
    }
    return HttpResponse(template.render(context, request))


def download_report(request):
    """The PDF the task sheet asks for."""
    bundle = artifacts.load()
    if bundle is None:
        return _not_built(request)

    drawn = _figures(bundle)
    figures = {key: path for key, (_url, path) in drawn.items()}

    pdf = report.build(bundle, figures=figures)
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        'attachment; filename="HCAI-project3-report.pdf"')
    return response


def be_the_expert(request):
    """Task 5: let a person answer the articles the system chose to defer.

    The same query loop as Task 4, with a reader in place of the simulation. The
    articles are exactly the ones the deferral rule handed over, and the score at
    the end compares the reader against both the simulated expert and the
    classifier on those same articles.
    """
    bundle = artifacts.load()
    if bundle is None:
        return _not_built(request)

    model = artifacts.load_classifier()
    if model is None:
        return _not_built(request, "The cached classifier is missing.")

    _train, test = data.load()
    deferred = bundle.get("deferred_test_indices") or []
    if not deferred:
        return _not_built(request, "This run deferred no articles at all.")

    state = request.session.get(SESSION_KEY)

    if request.method == "POST" and "restart" in request.POST:
        state = None

    if state is None or not isinstance(state, dict):
        import numpy as np
        rng = np.random.default_rng()
        picks = rng.choice(len(deferred), size=min(ROUND_SIZE, len(deferred)),
                           replace=False)
        state = {
            "indices": [int(deferred[position]) for position in picks],
            "answers": [],
        }
        request.session[SESSION_KEY] = state

    if request.method == "POST" and "answer" in request.POST:
        try:
            answer = int(request.POST["answer"])
        except (TypeError, ValueError):
            answer = -1
        if 0 <= answer < len(data.CLASS_NAMES) and \
                len(state["answers"]) < len(state["indices"]):
            state["answers"].append(answer)
            request.session[SESSION_KEY] = state
        return redirect("project3:expert")

    position = len(state["answers"])
    finished = position >= len(state["indices"])

    context = {
        "class_names": data.CLASS_NAMES,
        "total": len(state["indices"]),
        "position": position,
        "finished": finished,
        "expert_name": bundle["expert"]["name"],
    }

    if not finished:
        row = int(state["indices"][position])
        context["article"] = test["text"].iloc[row]
    else:
        rows = [int(index) for index in state["indices"]]
        texts = test["text"].to_numpy()[rows]
        truth = test["label"].to_numpy()[rows]
        answers = list(state["answers"])

        simulated = experts.build(bundle["expert_kind"], seed=bundle["seed"])
        simulated_labels = simulated.predict(texts, truth)
        predicted = model.predict(texts)

        context["review"] = [
            {
                "text": texts[position][:260],
                "truth": data.CLASS_NAMES[int(truth[position])],
                "yours": data.CLASS_NAMES[answers[position]],
                "expert": data.CLASS_NAMES[int(simulated_labels[position])],
                "classifier": data.CLASS_NAMES[int(predicted[position])],
                "you_right": int(answers[position]) == int(truth[position]),
            }
            for position in range(len(rows))
        ]
        context["scores"] = {
            "you": sum(1 for item in context["review"] if item["you_right"]) / len(rows),
            "expert": float((simulated_labels == truth).mean()),
            "classifier": float((predicted == truth).mean()),
            "n": len(rows),
        }

    template = loader.get_template("project3/expert.html")
    return HttpResponse(template.render(context, request))
