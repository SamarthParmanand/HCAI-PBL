"""Simulated experts for Project 3, Task 2.

The task sheet asks for experts that are **not** perfect and that "exhibit
expertise in specific regions of the input space". Two are implemented, because
the two natural readings of "region" behave differently and the contrast is worth
showing:

* `TopicExpert` -- a subject specialist. Its region is a set of topics: think of a
  journalist who follows sport and business closely and skims the rest. Inside
  its beat it is near-perfect; outside it guesses badly. The region is defined by
  the article's topic, which is not directly observable but *is* inferable from
  the text, so a deferral rule has to learn the topic boundary to exploit it.

* `KeywordExpert` -- a vocabulary specialist. Its region is defined directly by
  the words present in the article, so it is visible in the input without any
  inference. This makes expert competence much easier to predict, and it is the
  more literal reading of "region of the input space".

Both are deterministic given a seed: the same article always receives the same
answer, which matters because the active-learning loop may query the same point
more than once and must not get a fresh coin flip each time.
"""

import numpy as np

from .data import CLASS_NAMES, N_CLASSES

# Words that mark an article as being inside KeywordExpert's beat. Chosen to sit
# mostly in the Business and Sci/Tech vocabulary without naming the classes.
KEYWORDS = (
    "stocks", "shares", "profit", "revenue", "earnings", "market", "economy",
    "bank", "investors", "oil", "prices", "dollar", "merger", "quarterly",
    "software", "microsoft", "google", "internet", "computer", "chip",
    "wireless", "linux", "server", "browser", "space", "research",
)


class Expert:
    """Common machinery: per-row correctness, then a wrong answer if incorrect."""

    name = "expert"
    description = ""

    def __init__(self, seed=0):
        self.seed = seed

    def competence(self, texts, labels):
        """P(correct) for each row. Implemented by each expert."""
        raise NotImplementedError

    def in_region(self, texts, labels):
        """Boolean mask: is this row inside the expert's area of expertise?"""
        raise NotImplementedError

    def predict(self, texts, labels):
        """The expert's label for each row.

        `labels` are the true labels: a simulated expert is defined by how often
        it agrees with the truth, so it needs them. Nothing downstream is allowed
        to see them except through this call.
        """
        texts = np.asarray(texts, dtype=object)
        labels = np.asarray(labels, dtype=int)

        probabilities = self.competence(texts, labels)

        # Seeded per row by content, so the answer for an article never changes
        # between calls, batches or runs.
        draws = _stable_uniform(texts, labels, self.seed)
        correct = draws < probabilities

        # A wrong expert names one of the other classes, chosen deterministically.
        offset = 1 + (_stable_integers(texts, labels, self.seed) % (N_CLASSES - 1))
        wrong = (labels + offset) % N_CLASSES

        return np.where(correct, labels, wrong).astype(int)

    def report(self, texts, labels):
        """Accuracy overall, inside/outside the region, and per class."""
        predicted = self.predict(texts, labels)
        labels = np.asarray(labels, dtype=int)
        inside = self.in_region(texts, labels)
        hit = predicted == labels

        per_class = {}
        for index, name in enumerate(CLASS_NAMES):
            mask = labels == index
            per_class[name] = {
                "n": int(mask.sum()),
                "accuracy": float(hit[mask].mean()) if mask.any() else float("nan"),
                # How much of this topic falls inside the beat. Read from
                # `in_region` rather than inferred from the accuracy, which only
                # coincides for `TopicExpert`: `KeywordExpert`'s region cuts
                # across topics, so no accuracy threshold identifies it.
                "in_region_share": (float(inside[mask].mean()) if mask.any()
                                    else float("nan")),
            }

        return {
            "name": self.name,
            "description": self.description,
            "accuracy": float(hit.mean()),
            "accuracy_in_region": float(hit[inside].mean()) if inside.any() else float("nan"),
            "accuracy_out_of_region": float(hit[~inside].mean()) if (~inside).any() else float("nan"),
            "region_share": float(inside.mean()),
            "per_class": per_class,
        }


# note to self: the design decision in task 2 is that the expert must be
# imperfect and strong only in a region. mine is also deliberately worse than
# the classifier overall, 0.7046 against 0.9204, because an expert who was
# uniformly better would make the whole deferral question trivial, just always
# defer. being better only somewhere is what forces the system to learn where.
# the topic version is the harder of the two for the system, because its region
# is not directly observable and has to be inferred from the text. one more
# point worth making: the classifier already beats the expert on some topics the
# expert is strong at, so deferring the entire beat would lose accuracy, and the
# rule has to be finer than "is this inside the region".
class TopicExpert(Expert):
    """Near-perfect on a few topics, poor on the rest."""

    name = "Topic specialist"

    def __init__(self, specialties=(1, 2), strong=0.95, weak=0.45, seed=0):
        super().__init__(seed)
        self.specialties = tuple(specialties)
        self.strong = strong
        self.weak = weak
        listed = ", ".join(CLASS_NAMES[index] for index in self.specialties)
        self.description = (
            f"Correct {strong:.0%} of the time on {listed}, and only {weak:.0%} "
            f"elsewhere. Its beat is a set of topics, so a deferral rule has to "
            f"infer the topic from the text before it can exploit the expert.")

    def in_region(self, texts, labels):
        return np.isin(np.asarray(labels, dtype=int), self.specialties)

    def competence(self, texts, labels):
        inside = self.in_region(texts, labels)
        return np.where(inside, self.strong, self.weak)


class KeywordExpert(Expert):
    """Near-perfect on articles containing its vocabulary, poor otherwise."""

    name = "Vocabulary specialist"

    def __init__(self, keywords=KEYWORDS, strong=0.93, weak=0.40, seed=0):
        super().__init__(seed)
        self.keywords = tuple(keywords)
        self.strong = strong
        self.weak = weak
        self.description = (
            f"Correct {strong:.0%} of the time on articles containing any of "
            f"{len(self.keywords)} marker words, and {weak:.0%} otherwise. Its "
            f"region is visible directly in the text, so expert competence is "
            f"easy to predict from the input.")

    def in_region(self, texts, labels):
        lowered = np.asarray([str(text).lower() for text in texts], dtype=object)
        return np.array([any(word in text for word in self.keywords)
                         for text in lowered], dtype=bool)

    def competence(self, texts, labels):
        inside = self.in_region(texts, labels)
        return np.where(inside, self.strong, self.weak)


EXPERTS = {
    "topic": TopicExpert,
    "keyword": KeywordExpert,
}

EXPERT_CHOICES = (
    ("topic", "Topic specialist"),
    ("keyword", "Vocabulary specialist"),
)


def build(kind="topic", seed=0):
    if kind not in EXPERTS:
        raise ValueError(f"unknown expert {kind!r}")
    return EXPERTS[kind](seed=seed)


# note to self: the expert has to give the same article the same answer every
# time, because the active learning loop can query the same point more than once
# and must not receive a fresh coin flip each time. blake2b rather than python's
# hash, because string hashing is salted per process and the answer would change
# after a restart.
def _stable_bytes(texts, labels, seed):
    """A reproducible integer per row, derived from its content."""
    import hashlib
    out = np.empty(len(texts), dtype=np.uint64)
    for index, (text, label) in enumerate(zip(texts, labels)):
        digest = hashlib.blake2b(
            f"{seed}|{label}|{text}".encode("utf-8", "replace"), digest_size=8
        ).digest()
        out[index] = int.from_bytes(digest, "big")
    return out


def _stable_uniform(texts, labels, seed):
    """The same row always draws the same number in [0, 1)."""
    return _stable_bytes(texts, labels, seed) / float(2 ** 64)


def _stable_integers(texts, labels, seed):
    return _stable_bytes(texts, labels, seed + 1).astype(np.int64) % (2 ** 31)
