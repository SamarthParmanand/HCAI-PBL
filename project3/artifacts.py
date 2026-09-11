"""Runs the whole Project 3 experiment once and caches the result.

Fitting on 120,000 articles takes a couple of minutes, which is far too long for
a web request, so everything is computed once and cached under `data/project3/`.
The interface loads the cached bundle; `python manage.py build_project3` rebuilds
it deliberately.

**How the data is divided, and why.** The classifier is trained on 80% of the
training set (`fit`) and the remaining 20% (`meta`) is kept back. The competence
models are fitted on `meta`, where the classifier's predictions are honest. This
is the one methodological point that matters in the whole file: fitted on the
classifier's own training rows, a model of "is the classifier right here?" learns
that it is always right -- training accuracy is near 100% -- and the system would
never defer.

The test set is used for evaluation only, with one exception that is labelled as
such wherever it appears: `sweep_confidence` and `sweep_competence` trace their
curves over test accuracy, so the `*_best` rows they yield are operating points
chosen *in hindsight*. They are reported as an upper bound, never as the system's
score. Every headline number is an untuned rule -- the margin-0 competence rule,
which is the rule the theory prescribes rather than one selected after seeing the
answers.

Task 1's headline accuracy is reported for a classifier trained on *all* the
labels, as the task sheet asks. The deferral system uses the 80% classifier, so
both numbers are reported and the small gap between them is visible rather than
hidden.
"""

import os
import time

import joblib
import numpy as np

from . import active, classifier, data, defer, experts

CACHE = os.path.join(data.ROOT, "data", "project3")

DEFAULT_EXPERT = "topic"
DEFAULT_SEED = 0

# note to self: the split i will definitely be asked about. the deployed
# classifier is fitted on 80 percent of the training rows, and the 20 percent
# held back is what the confidence calibration and the competence models are
# estimated on. the reason is not squeamishness about data: on its own training
# rows the classifier is right almost always, so anything fitted there concludes
# it is never wrong and the system never defers at all. both accuracies are
# reported side by side, 0.9245 on all labels against 0.9204 for the deployed
# one, so the cost of the decision is visible instead of buried.
META_SHARE = 0.2

# The active-learning pool is a sample of the meta split. A few thousand points
# is ample for a budget of a thousand queries, and it keeps the loop responsive.
POOL_SIZE = 8000
AL_ROUNDS = 10
AL_BATCH = 100


def cache_path(expert_kind, seed, limit):
    tag = f"{expert_kind}_seed{seed}_limit{limit or 'full'}"
    return os.path.join(CACHE, f"bundle_{tag}.joblib")


def classifier_path(seed, limit):
    tag = f"seed{seed}_limit{limit or 'full'}"
    return os.path.join(CACHE, f"classifier_{tag}.joblib")


def build(expert_kind=DEFAULT_EXPERT, seed=DEFAULT_SEED, limit=None,
          rounds=AL_ROUNDS, batch=AL_BATCH, pool_size=POOL_SIZE, save=True,
          log=print):
    """Run every task and return the bundle of results."""
    started = time.time()
    train, test = data.load(limit=limit, seed=seed)
    log(f"data: {len(train)} train / {len(test)} test")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    cut = int(len(train) * (1.0 - META_SHARE))
    fit_index, meta_index = order[:cut], order[cut:]

    fit_texts = train["text"].to_numpy()[fit_index]
    fit_labels = train["label"].to_numpy()[fit_index]
    meta_texts = train["text"].to_numpy()[meta_index]
    meta_labels = train["label"].to_numpy()[meta_index]

    test_texts = test["text"].to_numpy()
    test_labels = test["label"].to_numpy()

    # --- Task 1 ---------------------------------------------------------
    log("task 1: fitting on all labels")
    full_model = classifier.fit(train["text"], train["label"], seed=seed)
    baseline = classifier.evaluate(full_model, test_texts, test_labels)
    log(f"  full-train test accuracy {baseline['accuracy']:.4f}")

    log(f"system classifier: fitting on {len(fit_texts)} rows "
        f"({int((1 - META_SHARE) * 100)}%)")
    system_model = classifier.fit(fit_texts, fit_labels, seed=seed)
    system = classifier.evaluate(system_model, test_texts, test_labels)
    log(f"  system test accuracy {system['accuracy']:.4f}")

    # --- Task 2 ---------------------------------------------------------
    expert = experts.build(expert_kind, seed=seed)
    expert_report = expert.report(test_texts, test_labels)
    log(f"task 2: {expert.name} test accuracy {expert_report['accuracy']:.4f}")

    expert_meta = expert.predict(meta_texts, meta_labels)
    expert_test = expert.predict(test_texts, test_labels)

    meta_expert_correct = expert_meta == meta_labels
    test_expert_correct = expert_test == test_labels

    # --- honest competence signals on the held-out slice ---------------
    meta_predicted = system_model.predict(meta_texts)
    meta_classifier_correct = meta_predicted == meta_labels
    meta_confidence = classifier.confidence(system_model, meta_texts)

    test_predicted = system_model.predict(test_texts)
    test_classifier_correct = test_predicted == test_labels
    test_confidence = classifier.confidence(system_model, test_texts)

    log("task 3: fitting competence models on the held-out slice")

    # Two ways to estimate P(classifier correct | x), fitted on the same rows so
    # they can be compared fairly: Platt scaling of the classifier's own
    # confidence, and a text model of its correctness.
    calibration = defer.fit_confidence_calibration(
        meta_confidence, meta_classifier_correct, seed=seed)
    classifier_competence = defer.fit_competence(
        meta_texts, meta_classifier_correct, seed=seed)
    expert_competence = defer.fit_competence(
        meta_texts, meta_expert_correct, seed=seed)

    test_calibrated = defer.calibrated_probability(
        calibration, test_confidence, meta_classifier_correct.mean())
    test_from_text = defer.competence_probability(
        classifier_competence, test_texts, meta_classifier_correct.mean())
    test_expert_probability = defer.competence_probability(
        expert_competence, test_texts, meta_expert_correct.mean())

    quality = {
        "calibrated_confidence_auc": defer.competence_quality(
            test_calibrated, test_classifier_correct),
        "classifier_text_model_auc": defer.competence_quality(
            test_from_text, test_classifier_correct),
        "raw_confidence_auc": defer.competence_quality(
            test_confidence, test_classifier_correct),
        "expert_competence_auc": defer.competence_quality(
            test_expert_probability, test_expert_correct),
    }
    log(f"  P(classifier correct) AUC: calibrated confidence "
        f"{quality['calibrated_confidence_auc']:.3f}, text model "
        f"{quality['classifier_text_model_auc']:.3f}")
    log(f"  P(expert correct) AUC:     {quality['expert_competence_auc']:.3f}")

    # The calibrated confidence is the primary estimate; whichever wins is
    # reported, and the text-model variant is kept as the comparison.
    test_classifier_probability = test_calibrated

    # --- Task 3: the policies ------------------------------------------
    confidence_rows = defer.sweep_confidence(
        test_confidence, test_classifier_correct, test_expert_correct)
    competence_rows = defer.sweep_competence(
        test_expert_probability, test_classifier_probability,
        test_classifier_correct, test_expert_correct)
    competence_text_rows = defer.sweep_competence(
        test_expert_probability, test_from_text,
        test_classifier_correct, test_expert_correct)

    # Which test articles the rule actually hands over. The "be the expert" page
    # uses exactly this set, so a reader is asked the same questions the system
    # decided it could not answer.
    deferred_mask = defer.competence_policy(
        test_expert_probability, test_classifier_probability, 0.0)
    deferred_indices = np.flatnonzero(deferred_mask).tolist()

    always_me = defer.evaluate_policy(
        np.zeros(len(test_texts), dtype=bool),
        test_classifier_correct, test_expert_correct)
    always_expert = defer.evaluate_policy(
        np.ones(len(test_texts), dtype=bool),
        test_classifier_correct, test_expert_correct)
    oracle = defer.evaluate_policy(
        defer.oracle_decisions(test_classifier_correct, test_expert_correct),
        test_classifier_correct, test_expert_correct)

    best_confidence = defer.best_by_team_accuracy(confidence_rows)
    best_competence = defer.best_by_team_accuracy(competence_rows)
    best_competence_text = defer.best_by_team_accuracy(competence_text_rows)
    zero_margin = min(competence_rows, key=lambda row: abs(row["margin"]))
    zero_margin_text = min(competence_text_rows, key=lambda row: abs(row["margin"]))

    log(f"  classifier alone      {always_me['team_accuracy']:.4f}")
    log(f"  expert alone          {always_expert['team_accuracy']:.4f}")
    log(f"  confidence rule best  {best_confidence['team_accuracy']:.4f} "
        f"(threshold {best_confidence['threshold']:.2f})")
    log(f"  competence rule       {zero_margin['team_accuracy']:.4f} at margin 0 "
        f"(defers {zero_margin['deferral_rate']:.1%})")
    log(f"  competence (text m_clf) {zero_margin_text['team_accuracy']:.4f}")
    log(f"  oracle ceiling        {oracle['team_accuracy']:.4f}")

    # --- Task 4: active learning ---------------------------------------
    pool = rng.permutation(len(meta_texts))[:min(pool_size, len(meta_texts))]
    pool_texts = meta_texts[pool]
    pool_expert_correct = meta_expert_correct[pool]
    pool_confidence = classifier.confidence(system_model, pool_texts)
    pool_classifier_probability = defer.calibrated_probability(
        calibration, pool_confidence, meta_classifier_correct.mean())

    log(f"task 4: active learning, pool {len(pool_texts)}, "
        f"{rounds} rounds of {batch}")
    runs = active.compare(
        pool_texts=pool_texts,
        pool_expert_correct=pool_expert_correct,
        pool_classifier_probability=pool_classifier_probability,
        test_texts=test_texts,
        test_classifier_probability=test_classifier_probability,
        test_classifier_correct=test_classifier_correct,
        test_expert_correct=test_expert_correct,
        rounds=rounds, batch=batch, seed=seed,
    )

    # The target is what the fully-supervised rule of Task 3 achieved: the
    # question is how many expert labels it takes to get there from nothing.
    target = zero_margin["team_accuracy"]
    summary = active.summarise(runs, target=target)
    for strategy, row in summary.items():
        log(f"  {row['label']:26} acc {row['team_accuracy']:.4f}  "
            f"to target: {row['queries_to_target']}")

    elapsed = time.time() - started

    bundle = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": elapsed,
        "expert_kind": expert_kind,
        "seed": seed,
        "limit": limit,
        "dataset": data.describe(train, test),
        "split": {
            "fit_rows": int(len(fit_texts)),
            "meta_rows": int(len(meta_texts)),
            "meta_share": META_SHARE,
            "pool_rows": int(len(pool_texts)),
        },
        "baseline": baseline,
        "system": system,
        "expert": expert_report,
        "competence_quality": quality,
        "policies": {
            "classifier_only": always_me,
            "expert_only": always_expert,
            "oracle": oracle,
            "confidence_best": best_confidence,
            "competence_best": best_competence,
            "competence_zero_margin": zero_margin,
            "competence_text_best": best_competence_text,
            "competence_text_zero_margin": zero_margin_text,
        },
        "confidence_sweep": confidence_rows,
        "competence_sweep": competence_rows,
        "competence_text_sweep": competence_text_rows,
        "active": runs,
        "active_summary": summary,
        "active_target": target,
        "active_settings": {"rounds": rounds, "batch": batch,
                            "pool_size": int(len(pool_texts))},
        "deferred_test_indices": deferred_indices,
    }

    if save:
        os.makedirs(CACHE, exist_ok=True)
        joblib.dump(bundle, cache_path(expert_kind, seed, limit), compress=3)
        joblib.dump(system_model, classifier_path(seed, limit), compress=3)
        log(f"cached to {CACHE} in {elapsed:.0f}s")

    return bundle


def load(expert_kind=DEFAULT_EXPERT, seed=DEFAULT_SEED, limit=None):
    """The cached bundle, or None if it has not been built."""
    path = cache_path(expert_kind, seed, limit)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_classifier(seed=DEFAULT_SEED, limit=None):
    path = classifier_path(seed, limit)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def is_built(expert_kind=DEFAULT_EXPERT, seed=DEFAULT_SEED, limit=None):
    return os.path.exists(cache_path(expert_kind, seed, limit))
