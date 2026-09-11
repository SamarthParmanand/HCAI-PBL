"""Storage for the user study.

A study that loses its data is not a study, so answers go to the database rather
than the session. The session only remembers *which* session a browser belongs to
and how far through it is; everything a researcher would later analyse is a row
here, and the landing page can export the lot as CSV.
"""

import json

from django.db import models

PAIRWISE = "pairwise"
RANKING = "ranking"
VALIDATION = "validation"

BLOCK_CHOICES = [
    (PAIRWISE, "Design 1: pairwise choice"),
    (RANKING, "Design 2: rank ten"),
    (VALIDATION, "Held-out validation"),
]

PAIRWISE_FIRST = "pairwise_first"
RANKING_FIRST = "ranking_first"

ORDER_CHOICES = [
    (PAIRWISE_FIRST, "Design 1 then Design 2"),
    (RANKING_FIRST, "Design 2 then Design 1"),
]


class StudySession(models.Model):
    """One participant's run through both interfaces."""

    token = models.CharField(max_length=32, unique=True, db_index=True)
    # Counterbalancing: half the participants meet each design first, so a
    # practice or fatigue effect cannot be mistaken for an effect of the design.
    order = models.CharField(max_length=20, choices=ORDER_CHOICES,
                             default=PAIRWISE_FIRST)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):                                    # pragma: no cover
        return f"session {self.token} ({self.order})"

    @property
    def complete(self):
        return self.finished_at is not None

    def answers_for(self, block):
        return self.answers.filter(block=block).order_by("position")


class Answer(models.Model):
    """One elicitation task: what was shown, what came back, how long it took."""

    session = models.ForeignKey(StudySession, on_delete=models.CASCADE,
                                related_name="answers")
    block = models.CharField(max_length=16, choices=BLOCK_CHOICES)
    position = models.IntegerField()

    # JSON lists of movie ids: as presented, and as the participant ordered them
    # (most preferred first). Kept as text so the store needs no JSON field.
    shown = models.TextField()
    ordering = models.TextField()

    seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["session", "block", "position"]
        unique_together = [("session", "block", "position")]

    def __str__(self):                                    # pragma: no cover
        return f"{self.block}#{self.position} of {self.session.token}"

    @property
    def shown_ids(self):
        return json.loads(self.shown)

    @property
    def ordered_ids(self):
        return json.loads(self.ordering)

    def set_shown(self, ids):
        self.shown = json.dumps([int(value) for value in ids])

    def set_ordering(self, ids):
        self.ordering = json.dumps([int(value) for value in ids])
