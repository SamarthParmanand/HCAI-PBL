"""Active learning for expert-competence discovery: Project 3, Task 4.

The setting the task sheet sets out: the classifier may use the labelled training
set, there are **no** expert labels to start with, and the expert can be queried
point by point. So the classifier and its calibrated confidence can both be
fitted upfront -- neither needs the expert -- and the only thing that has to be
bought is P(expert correct | x).

The classifier used here is the deployed one, fitted on 80% of the training rows
rather than all of them (`artifacts.py`). The remaining 20% is what the
confidence calibration is estimated on: a classifier is right almost always on
its own training rows, so a confidence model fitted there would conclude it is
never wrong. Both accuracies are reported in Task 1, and the gap is small.

**Which points are worth buying.** The system does not need a good model of the
expert everywhere. It needs one binary decision to come out right:

    defer(x)  <=>  P(expert correct | x) > P(classifier correct | x)

so the informative labels are the ones that pin down *that* boundary, which is
what `decision_boundary` targets: the pool points where the two estimated
competences are closest, where a single label can flip the decision.

**Why that is not enough on its own, measured.** Querying only near the boundary
makes the training set unrepresentative of the data the model is later asked
about, and a competence model fitted on it does not transfer. On the full run
this is stark -- the expert model reaches AUC 0.59 under pure boundary sampling
against 0.73 under random, and pure `competence_uncertainty` is worse still: it
drifts into a skewed corner of the pool and its estimate of the expert's base
rate collapses from 0.71 to 0.51. This is the standard sampling-bias failure of
greedy active learning, and it is why the strategy actually proposed here,
`hybrid_boundary`, splits every batch: half the labels are drawn uniformly to
keep the sample representative and the base rate honest, and half are spent on
the boundary. The pure variants are kept in the comparison so the size of the
bias is visible rather than asserted.

**Cold start.** Round one has no expert model, so strategies that need one fall
back: the boundary strategies to classifier uncertainty, which is the best
available proxy, and `competence_uncertainty` to random.
"""

import numpy as np

from . import defer

SEED = 0

# Fraction of each batch spent on uniform exploration by the hybrid strategy.
EXPLORE_SHARE = 0.5

STRATEGIES = ("hybrid_boundary", "decision_boundary", "competence_uncertainty",
              "classifier_uncertainty", "random")

STRATEGY_LABELS = {
    "hybrid_boundary": "Boundary + exploration (ours)",
    "decision_boundary": "Deferral boundary only",
    "competence_uncertainty": "Expert-model uncertainty",
    "classifier_uncertainty": "Classifier uncertainty",
    "random": "Random",
}

# How much of each batch each strategy draws uniformly at random.
EXPLORE_FRACTION = {
    "hybrid_boundary": EXPLORE_SHARE,
    "decision_boundary": 0.0,
    "competence_uncertainty": 0.0,
    "classifier_uncertainty": 0.0,
    "random": 1.0,
}

# The acquisition score each strategy sorts by, ascending. `None` means random.
# note to self: this is where each strategy decides what it wants to buy next,
# and the argument behind my proposed one goes here. the system does not need a
# good model of the expert everywhere, it needs one binary comparison to come out
# right, so the labels worth buying are the ones near where the two competences
# cross, because a single label there can flip the decision.
def _scores(strategy, remaining, classifier_probability, expert_probability):
    if strategy == "random":
        return None

    if expert_probability is None:
        # Cold start: no expert model yet.
        if strategy in ("decision_boundary", "hybrid_boundary",
                        "classifier_uncertainty"):
            return classifier_probability[remaining]
        return None                                    # competence_uncertainty

    if strategy == "classifier_uncertainty":
        return classifier_probability[remaining]

    if strategy == "competence_uncertainty":
        return np.abs(expert_probability[remaining] - 0.5)

    if strategy in ("decision_boundary", "hybrid_boundary"):
        return np.abs(expert_probability[remaining]
                      - classifier_probability[remaining])

    raise ValueError(f"unknown strategy {strategy!r}")


# note to self: and this is where the argument above has to be repaired, which
# is the actual finding of task 4. picking purely by score makes the queried
# sample stop resembling the pool the model is later asked about, so the fitted
# competence model does not transfer, auc 0.594 against 0.730 for plain uniform
# sampling. mixing a fixed share of uniform picks into every batch fixes the
# mechanism and brings it back to 0.706. this is the classic sampling bias
# failure of greedy active learning, and here it is measured rather than
# hypothesised.
def _select(strategy, size, remaining, rng, classifier_probability,
            expert_probability):
    """Positions within `remaining` to query next."""
    size = min(size, len(remaining))
    if size == 0:
        return np.array([], dtype=int)

    explore = EXPLORE_FRACTION.get(strategy, 0.0)
    n_explore = int(round(size * explore))
    n_exploit = size - n_explore

    chosen = []
    available = np.arange(len(remaining))

    if n_exploit > 0:
        scores = _scores(strategy, remaining, classifier_probability,
                         expert_probability)
        if scores is None:
            n_explore, n_exploit = size, 0
        else:
            order = np.argsort(scores)[:n_exploit]
            chosen.append(order)
            available = np.setdiff1d(available, order, assume_unique=False)

    if n_explore > 0 and len(available):
        picks = rng.choice(available, size=min(n_explore, len(available)),
                           replace=False)
        chosen.append(picks)

    return np.concatenate(chosen) if chosen else np.array([], dtype=int)


# note to self: one strategy's learning curve. rounds of a fixed batch, and after
# every round the competence model is refitted on everything bought so far and
# the whole policy is re-evaluated, which is what makes the curves comparable
# across strategies. the expert is deterministic per article, so re-querying a
# point cannot smuggle in a fresh coin flip. the honest conclusion to state: on
# this problem none of the targeted strategies beats uniform sampling on team
# accuracy, because this expert's competence is defined by topic and topic is
# easy to read off the text, and because the deferral decision only changes the
# answer on about 8 percent of articles. where they differ is the quality of the
# deferral decisions at small budgets.
def run(strategy, pool_texts, pool_expert_correct, pool_classifier_probability,
        test_texts, test_classifier_probability, test_classifier_correct,
        test_expert_correct, rounds=10, batch=100, seed=SEED):
    """One active-learning run. Returns a row per round.

    `pool_expert_correct` is the oracle the loop may not look at except through a
    query: a point's value is read only after its index has been selected.
    """
    rng = np.random.default_rng(seed)
    pool_texts = np.asarray(pool_texts, dtype=object)
    pool_expert_correct = np.asarray(pool_expert_correct, dtype=bool)

    remaining = np.arange(len(pool_texts))
    queried = np.array([], dtype=int)

    expert_probability_pool = None
    history = []

    for step in range(1, rounds + 1):
        picks = _select(strategy, batch, remaining, rng,
                        pool_classifier_probability, expert_probability_pool)
        if len(picks) == 0:
            break

        newly = remaining[picks]
        queried = np.concatenate([queried, newly])
        remaining = np.setdiff1d(remaining, newly, assume_unique=False)

        # The query: only now may these labels be read.
        observed = pool_expert_correct[queried]
        model = defer.fit_competence(pool_texts[queried], observed, seed=seed)
        base_rate = float(observed.mean()) if len(observed) else 0.5

        expert_probability_pool = defer.competence_probability(
            model, pool_texts, base_rate)
        expert_probability_test = defer.competence_probability(
            model, test_texts, base_rate)

        mask = defer.competence_policy(expert_probability_test,
                                       test_classifier_probability)
        scores = defer.evaluate_policy(mask, test_classifier_correct,
                                       test_expert_correct)

        history.append({
            "round": step,
            "queries": int(len(queried)),
            "expert_base_rate": base_rate,
            "expert_model_auc": defer.competence_quality(expert_probability_test,
                                                         test_expert_correct),
            **scores,
        })

        if len(remaining) == 0:
            break

    return history


def compare(strategies=STRATEGIES, **kwargs):
    """Run several strategies on the same pool and return them keyed by name."""
    return {strategy: run(strategy, **kwargs) for strategy in strategies}


def queries_to_reach(history, target):
    """How many expert labels a run needed to first reach `target` accuracy.

    The headline number for an active-learning comparison: not the score at the
    end of the budget, but the budget needed to arrive.
    """
    for row in history:
        if row["team_accuracy"] >= target:
            return row["queries"]
    return None


def summarise(runs, target=None):
    """Final scores per strategy, and the budget each needed to reach `target`."""
    summary = {}
    for strategy, history in runs.items():
        last = history[-1]
        best = max(history, key=lambda row: row["team_accuracy"])
        summary[strategy] = {
            "label": STRATEGY_LABELS.get(strategy, strategy),
            "queries": last["queries"],
            "team_accuracy": last["team_accuracy"],
            "best_team_accuracy": best["team_accuracy"],
            "deferral_f1": last["deferral_f1"],
            "expert_model_auc": last["expert_model_auc"],
            "expert_base_rate": last["expert_base_rate"],
            "deferral_rate": last["deferral_rate"],
            "queries_to_target": (queries_to_reach(history, target)
                                  if target is not None else None),
        }
    return summary
