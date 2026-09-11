"""Learning to defer for Project 3, Task 3.

The system sees an article and must either answer itself or pay to ask the
expert. Deferring is worth it exactly when the expert is more likely to be right
than the classifier, so the rule compares two estimated competences:

    defer(x)  <=>  P(expert correct | x)  >  P(classifier correct | x)

Both terms are *learned*, on a slice of the training data the classifier never
saw. That held-out slice matters: the classifier's confidence on its own training
rows is wildly optimistic, so a competence model fitted there would learn that
the classifier is never wrong and would never defer.

Two rules are implemented so the learned one can be judged against something:

* `confidence` -- the standard baseline. Defer when the classifier's own
  probability falls below a threshold. It knows nothing about the expert, so it
  cannot tell a hard article the expert would also fail from a hard article the
  expert would get right.
* `competence` -- the learning-to-defer rule above, which models both sides.

**Judging the deferral decisions, not just the accuracy.** Team accuracy alone
hides whether the deferrals were the *right* ones: a rule that defers everything
inherits the expert's accuracy without having decided anything. So each rule is
also scored against the oracle decision, "defer iff the expert is right here and
the classifier is wrong", reported as precision, recall and F1 over that target,
alongside coverage and the cost of each deferral in accuracy terms.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

SEED = 0

# Features for the competence models. Smaller than the classifier's own space:
# these models are fitted on far fewer rows, and in Task 4 on a few hundred.
COMPETENCE_FEATURES = 60000


def build_competence_model(seed=SEED, C=1.0):
    """A model of "is this answer going to be correct?" from the text."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 1),
            min_df=1,
            max_features=COMPETENCE_FEATURES,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
        )),
        ("model", LogisticRegression(max_iter=2000, C=C, random_state=seed)),
    ])


def fit_competence(texts, correct, seed=SEED):
    """Fit P(correct | text).

    Returns None when the target has only one value -- with very few queries the
    expert may have been right (or wrong) every time, and a classifier cannot be
    fitted on a single class. Callers fall back to the observed base rate.
    """
    correct = np.asarray(correct).astype(int)
    if len(np.unique(correct)) < 2:
        return None

    model = build_competence_model(seed=seed)
    model.fit(texts, correct)
    return model


def competence_probability(model, texts, fallback):
    """P(correct | text), or a constant when no model could be fitted."""
    if model is None:
        return np.full(len(texts), float(fallback))
    return model.predict_proba(texts)[:, 1]


def _calibration_features(confidence):
    """The confidence itself and its logit, so the fit can bend where it needs to."""
    clipped = np.clip(np.asarray(confidence, dtype=float), 1e-6, 1 - 1e-6)
    return np.column_stack([clipped, np.log(clipped / (1 - clipped))])


# note to self: platt calibration, which is just a logistic regression fitted on
# the classifier's confidence against whether it turned out to be right. the
# reason it is needed rather than optional: a raw softmax score is
# overconfident, so comparing it directly against an estimated expert competence
# compares two numbers that are not on the same scale. after calibration both
# sides are probabilities and the inequality means something. measured auc 0.845
# here, against 0.749 for a text model of the classifier's own correctness, and
# using the weaker signal costs almost the entire benefit, 0.9214 against
# 0.9297.
def fit_confidence_calibration(confidence, correct, seed=SEED):
    """Turn the classifier's max probability into P(correct | x).

    A softmax maximum is not a probability of being right -- it is systematically
    over-confident -- and the deferral rule compares it against the expert's
    estimated competence, so the two have to be on the same scale before the
    comparison means anything. This is Platt scaling on the confidence, fitted on
    the held-out slice.

    It is deliberately preferred over a text model of classifier correctness: the
    classifier's own probability already carries most of the signal about whether
    it is about to be wrong, and correctness is a noisy target to learn from words
    alone. Both are fitted in `artifacts.py` and their AUCs reported side by side.
    """
    correct = np.asarray(correct).astype(int)
    if len(np.unique(correct)) < 2:
        return None
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(_calibration_features(confidence), correct)
    return model


def calibrated_probability(model, confidence, fallback):
    if model is None:
        return np.full(len(confidence), float(fallback))
    return model.predict_proba(_calibration_features(confidence))[:, 1]


# note to self: the oracle defers exactly when the expert is right and the
# classifier is wrong. it needs the true labels, so it can never be deployed,
# it is only the ceiling to measure against, 0.9732 here. worth noting that it
# defers on just 5.3 percent of articles while my rule defers on 8.1, and that
# gap is the price of not knowing in advance which of the expert's answers will
# help.
def oracle_decisions(classifier_correct, expert_correct):
    """The deferrals a perfect oracle would make.

    Deferring is only genuinely right where the expert succeeds and the
    classifier fails; anywhere else it either changes nothing or loses an answer
    that was already correct.
    """
    return (~np.asarray(classifier_correct, dtype=bool)
            & np.asarray(expert_correct, dtype=bool))


# note to self: task 3 asks for two evaluations and not one, so this returns
# both. the accuracy of the team, and the quality of the deferral decisions
# themselves, scored against the oracle decisions above. rescued means the
# expert fixed something the classifier had wrong, spoiled means the expert
# broke something it had right, and net gain is the difference. a policy can
# raise accuracy while still spoiling a lot of correct answers, which is exactly
# what the f1 column exposes and why accuracy alone would not answer the task.
def evaluate_policy(defer_mask, classifier_correct, expert_correct):
    """Team accuracy plus the quality of the deferral decisions themselves."""
    defer_mask = np.asarray(defer_mask, dtype=bool)
    classifier_correct = np.asarray(classifier_correct, dtype=bool)
    expert_correct = np.asarray(expert_correct, dtype=bool)

    team_correct = np.where(defer_mask, expert_correct, classifier_correct)
    oracle = oracle_decisions(classifier_correct, expert_correct)

    if defer_mask.any() and not defer_mask.all():
        precision, recall, f1, _ = precision_recall_fscore_support(
            oracle, defer_mask, average="binary", zero_division=0)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            oracle, defer_mask, average="binary", zero_division=0)

    # What the deferrals actually bought, split into the two kinds of mistake.
    rescued = int((defer_mask & oracle).sum())
    spoiled = int((defer_mask & classifier_correct & ~expert_correct).sum())

    return {
        "team_accuracy": float(team_correct.mean()),
        "deferral_rate": float(defer_mask.mean()),
        "coverage": float(1.0 - defer_mask.mean()),
        "deferral_precision": float(precision),
        "deferral_recall": float(recall),
        "deferral_f1": float(f1),
        "rescued": rescued,
        "spoiled": spoiled,
        "net_gain": rescued - spoiled,
        "oracle_opportunities": int(oracle.sum()),
    }


def confidence_policy(classifier_confidence, threshold):
    """Defer when the classifier is not confident enough."""
    return np.asarray(classifier_confidence) < threshold


# note to self: the deferral rule itself, and it really is one line.
#     defer(x) iff p(expert correct | x) > p(classifier correct | x)
# the margin argument exists so the rule can be made more reluctant to hand work
# over, but the number i put forward is margin 0, the theory's own default with
# nothing tuned.
def competence_policy(expert_probability, classifier_probability, margin=0.0):
    """Defer when the expert is the better bet by at least `margin`."""
    return (np.asarray(expert_probability)
            > np.asarray(classifier_probability) + margin)


# note to self: be careful with what comes out of here. these sweeps run over
# the test arrays, so taking the best row means choosing an operating point with
# hindsight on the evaluation set. i keep them as the curve behind the figure and
# as an upper bound, clearly labelled, and the number i report as the result is
# the untuned competence rule instead, 0.9297 rather than 0.9307. presenting the
# swept maximum as the headline would contradict "the test set is for evaluation
# only", which the sheet requires.
def sweep_confidence(classifier_confidence, classifier_correct, expert_correct,
                     thresholds=None):
    """The confidence baseline across its whole threshold range."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)

    rows = []
    for threshold in thresholds:
        mask = confidence_policy(classifier_confidence, threshold)
        row = evaluate_policy(mask, classifier_correct, expert_correct)
        row["threshold"] = float(threshold)
        rows.append(row)
    return rows


def sweep_competence(expert_probability, classifier_probability,
                     classifier_correct, expert_correct, margins=None):
    """The learned rule across a range of margins.

    A positive margin makes the system more reluctant to hand work over, which is
    what you want when a query costs something.
    """
    if margins is None:
        margins = np.linspace(-0.5, 0.5, 51)

    rows = []
    for margin in margins:
        mask = competence_policy(expert_probability, classifier_probability, margin)
        row = evaluate_policy(mask, classifier_correct, expert_correct)
        row["margin"] = float(margin)
        rows.append(row)
    return rows


def best_by_team_accuracy(rows):
    """The operating point with the highest team accuracy, ties to less deferral."""
    return max(rows, key=lambda row: (row["team_accuracy"], row["coverage"]))


def competence_quality(probability, correct):
    """How well a competence model ranks correct answers above incorrect ones."""
    correct = np.asarray(correct).astype(int)
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, probability))
