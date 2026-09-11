"""The preference model for Project 4, Task 2: Bradley-Terry, and its extension
to rankings.

Free of Django imports.

**Bradley-Terry.** With utility U(x) = w.x, the probability that film i is
preferred to film j is the logistic function of the utility difference:

    P(i > j) = exp(U_i) / (exp(U_i) + exp(U_j)) = sigma(w.(x_i - x_j))

which is exactly what Design 1 of the study observes.

**The extension: Plackett-Luce.** Design 2 asks for a full ranking
i_1 > i_2 > ... > i_n, so the model has to give a probability to an ordering
rather than to a single comparison. The extension used here reads the ranking as
a sequence of choices: the participant first picks their favourite out of all n
films, then their favourite out of the remaining n-1, and so on. Each of those
choices is a softmax over the utilities of the films still available, giving

    P(i_1 > ... > i_n) = prod_{k=1..n} exp(U_{i_k}) / sum_{j>=k} exp(U_{i_j})

**Why this extension and not another.** Three properties recommend it.

1. It *reduces to Bradley-Terry exactly* when n = 2. Both study designs therefore
   estimate the same w under the same likelihood, and the comparison between the
   two interfaces measures the interfaces rather than two different models. This
   is the decisive property: with a different ranking model, any difference
   between designs would confound interface with likelihood.
2. It is consistent with Luce's choice axiom, so the implied pairwise
   probabilities of a ranking agree with the pairwise model.
3. Its log-likelihood is concave in w, so the fit has one optimum and no
   restarts, seeds or local minima to report.

The alternative of expanding a ranking of ten films into all 45 implied pairwise
comparisons and fitting Bradley-Terry to them was rejected: those 45 comparisons
are not independent observations, so the likelihood would count one answer 45
times and report a spuriously tight posterior.

**Estimation.** MAP with a Gaussian prior, i.e. maximise the log-likelihood minus
`alpha * ||w||^2`. The prior is not decoration: with around 30 features and maybe
20 answers the likelihood alone is under-determined, and it is also what picks a
single representative from the directions the model cannot identify (see the note
in `movies.py`). The gradient is analytic and the problem concave, so L-BFGS
converges in a few dozen iterations.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

DEFAULT_ALPHA = 1.0


def _as_blocks(observations, features):
    """Turn observations into the feature rows they reference, best first.

    An observation is a list of movie ids in the order the participant put them,
    most preferred first. A pairwise choice is simply the two-item case.
    """
    blocks = []
    for ordering in observations:
        rows = [int(movie_id) for movie_id in ordering]
        if len(rows) >= 2:
            blocks.append(features[rows])
    return blocks


# note to self: the plackett-luce log-likelihood, and the derivation i should be
# able to give. read a ranking as a sequence of choices: the participant picks
# their favourite out of all n, then their favourite out of the remaining n-1,
# and so on. each stage is a softmax over the utilities still on the table, so
# the log probability of one stage is
#     u_chosen - logsumexp(utilities of everything still available)
# and the block is the sum over stages. the last stage is skipped because with
# one item left the probability is 1 and the term is 0, which is why the loop
# runs to len - 1. at n = 2 this collapses to log sigma(w.(x_i - x_j)), i.e.
# exactly bradley-terry, which is the reason i chose this extension.
def log_likelihood(w, blocks):
    """Plackett-Luce log-likelihood of every observed ordering."""
    total = 0.0
    for block in blocks:
        utilities = block @ w
        # Each stage: the chosen item's utility against those still available.
        # Recomputed per stage, so a block of n items costs O(n^2) exponentials.
        # A backwards cumulative log-sum-exp would make it one pass, but n is 10
        # here and the fit already converges in well under a second.
        for position in range(len(utilities) - 1):
            remaining = utilities[position:]
            total += utilities[position] - _logsumexp(remaining)
    return total


def _logsumexp(values):
    highest = np.max(values)
    return highest + np.log(np.sum(np.exp(values - highest)))


# note to self: the analytic gradient, the second thing i should be able to
# derive on demand. differentiating one stage's
#     u_chosen - logsumexp(remaining)
# with respect to w gives
#     x_chosen - sum_j softmax(remaining)_j * x_j
# that is, the chosen film's features minus the softmax weighted average of the
# features still available, which is the usual "observed minus expected" form.
# the gaussian prior contributes -2 * alpha * w. scipy minimises and this is a
# log posterior i want to maximise, so both the value and the gradient are
# returned negated.
def _negative_objective(w, blocks, alpha):
    """What L-BFGS minimises: negative log posterior, and its gradient."""
    value = 0.0
    gradient = np.zeros_like(w)

    for block in blocks:
        utilities = block @ w
        for position in range(len(utilities) - 1):
            remaining = utilities[position:]
            value += utilities[position] - _logsumexp(remaining)

            # d/dw of [ u_chosen - logsumexp(remaining) ]
            weights = softmax(remaining)
            gradient += block[position] - weights @ block[position:]

    value -= alpha * float(w @ w)
    gradient -= 2.0 * alpha * w

    return -value, -gradient


# note to self: map and not plain maximum likelihood. with 35 features and maybe
# twenty answers the likelihood on its own is under-determined, and the gaussian
# prior is also what picks a single representative out of the directions the
# answers cannot distinguish at all. the log-likelihood is concave and the
# gradient is analytic, so l-bfgs-b converges in a few dozen iterations and
# there are no restarts, seeds or local minima to report.
def fit(observations, features, alpha=DEFAULT_ALPHA):
    """MAP estimate of the preference vector w.

    `observations` is a list of orderings (lists of movie ids, best first);
    `features` is the design matrix. Returns `(w, info)`.
    """
    features = np.asarray(features, dtype=float)
    blocks = _as_blocks(observations, features)
    dimension = features.shape[1]

    if not blocks:
        return np.zeros(dimension), {
            "observations": 0, "comparisons": 0, "converged": False,
            "log_likelihood": 0.0, "message": "no answers yet",
        }

    result = minimize(
        _negative_objective,
        x0=np.zeros(dimension),
        args=(blocks, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )

    w = result.x
    return w, {
        "observations": len(blocks),
        # The number of independent choice events, which is what the fit
        # actually learns from: a ranking of n films contributes n-1 of them.
        "comparisons": int(sum(len(block) - 1 for block in blocks)),
        "converged": bool(result.success),
        "log_likelihood": float(log_likelihood(w, blocks)),
        "iterations": int(result.nit),
        "alpha": alpha,
        "message": str(result.message),
    }


def utilities(w, features):
    return np.asarray(features, dtype=float) @ np.asarray(w, dtype=float)


def probability_preferred(w, features, first, second):
    """Bradley-Terry probability that `first` is preferred to `second`."""
    difference = features[int(first)] - features[int(second)]
    return float(1.0 / (1.0 + np.exp(-(difference @ w))))


def ranking_log_probability(w, features, ordering):
    """Plackett-Luce log-probability of one ordering under w."""
    rows = features[[int(movie_id) for movie_id in ordering]]
    scores = rows @ w
    return float(sum(log_softmax(scores[position:])[0]
                     for position in range(len(scores) - 1)))


# note to self: this is the study's primary outcome measure, so it is worth
# remembering the bug it had. argmax on a tie returns the first index, and the
# first index is the participant's own pick by construction, so a zero weight
# vector scored a perfect 1.000 and the measure was silently broken. ties now
# share credit, which puts chance back at 0.500 where it belongs.
def predictive_accuracy(w, features, held_out):
    """Share of held-out answers whose top choice the fitted w predicts.

    The study's primary outcome measure: how well an elicited w predicts choices
    it was not fitted on. Each observation was recorded favourite-first, so a
    correct prediction means position 0 has the highest utility.

    Ties share the credit rather than being awarded to the first index. That is
    not a nicety: `argmax` on equal utilities returns 0, which is the
    participant's own pick by construction, so an all-zero w would otherwise
    score a perfect 1.0 instead of chance level.
    """
    if not held_out:
        return float("nan")

    total = 0.0
    counted = 0
    for ordering in held_out:
        rows = [int(movie_id) for movie_id in ordering]
        if len(rows) < 2:
            continue
        scores = utilities(w, features[rows])
        winners = np.flatnonzero(scores >= scores.max() - 1e-12)
        if 0 in winners:
            total += 1.0 / len(winners)
        counted += 1

    return total / counted if counted else float("nan")


def top_recommendations(w, features, frame, count=10, exclude=()):
    """The catalogue's highest-utility films under the estimated w."""
    scores = utilities(w, features)
    excluded = {int(movie_id) for movie_id in exclude}
    order = np.argsort(-scores)

    rows = []
    for position in order:
        if int(position) in excluded:
            continue
        rows.append({"movie_id": int(position), "utility": float(scores[position])})
        if len(rows) >= count:
            break
    return rows


def readable_weights(w, names, count=8):
    """The strongest positive and negative weights, for showing the estimate."""
    order = np.argsort(-np.asarray(w))
    strongest = [{"name": names[index], "weight": float(w[index])}
                 for index in order[:count]]
    weakest = [{"name": names[index], "weight": float(w[index])}
               for index in order[-count:][::-1]]
    return strongest, weakest
