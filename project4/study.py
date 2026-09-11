"""The study protocol as executable logic: Project 4, Tasks 3 and 4.

The design this implements, and the reasoning, is set out in full in the PDF
report. In short:

* **Within-subjects.** Every participant uses both interfaces, so each person is
  their own control and between-person variation in film taste -- which is
  enormous -- does not enter the comparison.
* **Counterbalanced.** Half meet Design 1 first, half Design 2, so practice and
  fatigue cannot masquerade as an effect of the interface.
* **A held-out validation block.** After both designs, every participant answers
  the same kind of pairwise questions once more. These answers are never fitted;
  they are the yardstick. The primary outcome is how well the w estimated from
  each design predicts them, which is what makes the two designs comparable at
  all despite collecting different numbers of raw answers.
* **Matched effort, not matched answers.** Design 1 gives 20 pairwise answers and
  Design 2 three rankings of ten, which is 27 independent choice events. The
  designs are deliberately *not* matched on count, because the interesting
  question is what a participant's time buys, and time is recorded per task.

The films shown are drawn uniformly at random from the catalogue, as the task
sheet specifies.
"""

import hashlib
import json
import secrets

import numpy as np

from django.db import IntegrityError, transaction

from . import movies, preference
from .models import (PAIRWISE, PAIRWISE_FIRST, RANKING, RANKING_FIRST,
                     VALIDATION, Answer, StudySession)

PAIRWISE_TASKS = 20
RANKING_TASKS = 3
RANKING_SIZE = 10
VALIDATION_TASKS = 10

BLOCK_SIZES = {
    PAIRWISE: PAIRWISE_TASKS,
    RANKING: RANKING_TASKS,
    VALIDATION: VALIDATION_TASKS,
}

BLOCK_ITEMS = {
    PAIRWISE: 2,
    RANKING: RANKING_SIZE,
    VALIDATION: 2,
}


# note to self: counterbalancing by alternating on the session count rather than
# tossing a coin. random assignment only balances in expectation, and with a
# handful of participants it can easily come out lopsided, which would let
# practice and fatigue look like an effect of the interface.
def new_session():
    """Create a session, assigning the counterbalanced order alternately.

    Alternating on the count rather than tossing a coin keeps the two orders
    balanced even for a small sample, which random assignment does not.
    """
    existing = StudySession.objects.count()
    order = PAIRWISE_FIRST if existing % 2 == 0 else RANKING_FIRST
    return StudySession.objects.create(token=secrets.token_hex(8), order=order)


def block_order(session):
    """The blocks this participant works through, in order."""
    if session.order == RANKING_FIRST:
        return [RANKING, PAIRWISE, VALIDATION]
    return [PAIRWISE, RANKING, VALIDATION]


def progress(session):
    """Which block and position comes next, or None when finished."""
    for block in block_order(session):
        done = session.answers_for(block).count()
        if done < BLOCK_SIZES[block]:
            return block, done
    return None, None


def total_tasks():
    return sum(BLOCK_SIZES.values())


def completed_tasks(session):
    return session.answers.count()


def sample_movies(frame, count, seed):
    """`count` distinct films drawn uniformly at random.

    Seeded from the session token and task position so a reload shows the same
    films rather than quietly resampling, which would let a participant shop for
    an easier question.
    """
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(frame), size=count, replace=False)
    return [int(value) for value in picks]


# note to self: the reason this is a digest and not python's hash is that the
# seed is read in two different requests, once to render the task and once to
# work out what was shown when the answer comes back. hash is salted per
# process, so after a restart the two disagreed and a pairwise answer could be
# stored against a pair that was never displayed together.
def task_seed(session, block, position):
    """A stable seed for one task of one session.

    Deliberately **not** Python's `hash()`. String hashing is salted per process,
    so `hash()` returns a different value after a restart and a different value
    in each worker of a multi-process server. That matters because this seed is
    read twice in two separate requests -- once to render the task and again to
    re-derive what was shown when the answer is submitted. If the two disagree,
    a ranking becomes unanswerable, and a pairwise answer can be stored against a
    pair that was never displayed together.

    `blake2b` is stable across processes and releases. The same approach is used
    for the same reason in `project3/experts.py`.
    """
    digest = hashlib.blake2b(
        f"{session.token}|{block}|{position}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)


def current_task(session, frame):
    """The films to show next, or None when the session is complete."""
    block, position = progress(session)
    if block is None:
        return None

    ids = sample_movies(frame, BLOCK_ITEMS[block],
                        task_seed(session, block, position))
    return {
        "block": block,
        "position": position,
        "size": BLOCK_SIZES[block],
        "movies": movies.display(frame, ids),
        "movie_ids": ids,
    }


def record(session, block, position, shown, ordering, seconds=None):
    """Store one answer. Returns the created row, or None if it already existed."""
    shown = [int(value) for value in shown]
    ordering = [int(value) for value in ordering]

    if sorted(shown) != sorted(ordering):
        raise ValueError("the ordering must be a permutation of the films shown")

    answer = Answer(session=session, block=block, position=position,
                    seconds=seconds)
    answer.set_shown(shown)
    answer.set_ordering(ordering)

    # A double submit hits the unique constraint on (session, block, position).
    # The insert needs its own savepoint: without one, the IntegrityError leaves
    # the surrounding transaction unusable and every later query in the same
    # request fails instead of the duplicate simply being ignored.
    try:
        with transaction.atomic():
            answer.save()
    except IntegrityError:
        return None
    return answer


def observations(session, block):
    """Answers from one block as orderings, ready for `preference.fit`."""
    return [answer.ordered_ids for answer in session.answers_for(block)]


def elapsed(session, block):
    seconds = [answer.seconds for answer in session.answers_for(block)
               if answer.seconds]
    return float(sum(seconds)) if seconds else None


# note to self: this is the comparison the whole study exists for. fit w from
# each design separately, then score both against the held out block that was
# never fitted. that is what makes the two designs comparable at all, because
# they collect different numbers of raw answers, twenty pairwise against
# twenty-seven choice events from three rankings of ten. effort is matched by
# time, which is recorded per task, not by answer count, because the interesting
# question is what a participant's time buys.
def analyse(session, frame, features, names, alpha=preference.DEFAULT_ALPHA):
    """Fit w from each design and score both against the held-out block.

    This is the per-participant half of the analysis the study would run. With
    one participant it is a demonstration; the same function over many sessions
    is the actual comparison.
    """
    held_out = observations(session, VALIDATION)

    results = {}
    for block, label in ((PAIRWISE, "Design 1: pairwise choice"),
                         (RANKING, "Design 2: rank ten")):
        answers = observations(session, block)
        w, info = preference.fit(answers, features, alpha=alpha)
        strongest, weakest = preference.readable_weights(w, names)

        results[block] = {
            "label": label,
            "answers": len(answers),
            "choice_events": info["comparisons"],
            "converged": info["converged"],
            "log_likelihood": info["log_likelihood"],
            "seconds": elapsed(session, block),
            "accuracy": preference.predictive_accuracy(w, features, held_out),
            "likes": strongest,
            "dislikes": weakest,
            "recommendations": movies.display(
                frame,
                [row["movie_id"] for row in
                 preference.top_recommendations(w, features, frame, count=5)],
            ),
            "w": w,
        }

    # Both designs pooled: what the whole session buys together.
    pooled = observations(session, PAIRWISE) + observations(session, RANKING)
    w_all, info_all = preference.fit(pooled, features, alpha=alpha)
    results["combined"] = {
        "label": "Both designs pooled",
        "answers": len(pooled),
        "choice_events": info_all["comparisons"],
        "accuracy": preference.predictive_accuracy(w_all, features, held_out),
        "recommendations": movies.display(
            frame,
            [row["movie_id"] for row in
             preference.top_recommendations(w_all, features, frame, count=5)],
        ),
    }

    results["held_out"] = len(held_out)
    results["chance"] = 0.5
    return results


def export_rows():
    """Every answer in the database, flattened for a CSV export."""
    rows = [["session", "order", "started_at", "block", "position",
             "shown", "ordering", "seconds"]]
    for session in StudySession.objects.all().prefetch_related("answers"):
        for answer in session.answers.all():
            rows.append([
                session.token,
                session.order,
                session.started_at.isoformat(),
                answer.block,
                answer.position,
                json.dumps(answer.shown_ids),
                json.dumps(answer.ordered_ids),
                f"{answer.seconds:.2f}" if answer.seconds else "",
            ])
    return rows
