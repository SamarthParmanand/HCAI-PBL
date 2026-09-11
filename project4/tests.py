"""Tests for Project 4 (preference elicitation).

Run with:  python manage.py test project4
"""

import io
import itertools
import re
import unittest
from unittest import mock

import numpy as np

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:                                       # pragma: no cover
    HAS_PYPDF = False

from django.db import OperationalError
from django.test import TestCase
from django.urls import reverse

from . import movies, preference, study
from .models import PAIRWISE, RANKING, VALIDATION, Answer, StudySession

HAS_DATA = True
try:
    movies.load_raw()
except movies.DatasetMissing:                             # pragma: no cover
    HAS_DATA = False

needs_data = unittest.skipUnless(
    HAS_DATA, "movie_metadata.csv (IMDB 5000) is not present")


@needs_data
class FeatureTests(TestCase):
    """Task 1."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frame = movies.load()
        cls.features, cls.names = movies.build_features(cls.frame)

    def test_the_catalogue_is_usable(self):
        self.assertGreater(len(self.frame), 3000)
        self.assertEqual(len(self.frame), len(self.frame["movie_id"].unique()))

    def test_no_film_is_missing_a_feature(self):
        """A film with a gap could not be shown in a comparison."""
        for column in ("title", "genres", "year", "duration", "imdb_score",
                       "num_voted_users"):
            self.assertFalse(self.frame[column].isna().any(), column)

    def test_titles_are_cleaned(self):
        """The published file leaves a non-breaking space on every title."""
        self.assertFalse(any("\xa0" in title for title in self.frame["title"]))

    def test_the_design_matrix_lines_up_with_the_names(self):
        self.assertEqual(self.features.shape[0], len(self.frame))
        self.assertEqual(self.features.shape[1], len(self.names))

    def test_the_one_hot_blocks_really_are_one_hot(self):
        blocks = movies.block_slices(self.frame)
        for name in ("era", "certificate"):
            block = self.features[:, blocks[name]]
            np.testing.assert_allclose(block.sum(axis=1),
                                       np.ones(len(self.frame)), atol=1e-9)

    def test_every_film_has_at_least_one_genre(self):
        blocks = movies.block_slices(self.frame)
        genre_block = self.features[:, blocks["genre"]]
        self.assertTrue(np.all(genre_block.sum(axis=1) >= 1))

    def test_numeric_features_are_standardised(self):
        """Otherwise a unit of w would mean something different in each."""
        blocks = movies.block_slices(self.frame)
        numeric = self.features[:, blocks["numeric"]]
        np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(numeric.std(axis=0), 1, atol=1e-9)

    def test_only_recognisable_films_are_offered(self):
        self.assertTrue(
            (self.frame["num_voted_users"] >= movies.MIN_VOTES).all())

    def test_display_returns_what_the_interface_needs(self):
        rows = movies.display(self.frame, [0, 1, 2])
        self.assertEqual(len(rows), 3)
        for row in rows:
            for key in ("title", "year", "genres", "score", "duration"):
                self.assertIn(key, row)


@needs_data
class PreferenceModelTests(TestCase):
    """Task 2: Bradley-Terry and its Plackett-Luce extension."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frame = movies.load()
        cls.features, cls.names = movies.build_features(cls.frame)
        cls.dimension = cls.features.shape[1]

    def some_w(self, seed=0):
        return np.random.default_rng(seed).normal(scale=0.3, size=self.dimension)

    def test_it_reduces_to_bradley_terry_at_two_items(self):
        """The decisive property: both designs share one likelihood."""
        w = self.some_w()
        first, second = 10, 200
        blocks = preference._as_blocks([[first, second]], self.features)
        plackett_luce = preference.log_likelihood(w, blocks)

        difference = self.features[first] - self.features[second]
        bradley_terry = -np.log1p(np.exp(-(difference @ w)))
        self.assertAlmostEqual(plackett_luce, bradley_terry, places=12)

    def test_the_analytic_gradient_matches_a_finite_difference(self):
        w = self.some_w(1)
        rng = np.random.default_rng(2)
        observations = [list(rng.choice(len(self.frame), size=size, replace=False))
                        for size in (2, 5, 10)]
        blocks = preference._as_blocks(observations, self.features)

        _value, gradient = preference._negative_objective(w, blocks, alpha=1.0)
        step = 1e-6
        for index in range(0, self.dimension, 4):       # every fourth, for speed
            bump = np.zeros(self.dimension)
            bump[index] = step
            high, _ = preference._negative_objective(w + bump, blocks, alpha=1.0)
            low, _ = preference._negative_objective(w - bump, blocks, alpha=1.0)
            self.assertAlmostEqual((high - low) / (2 * step), gradient[index],
                                   places=5)

    def test_it_is_a_proper_distribution_over_orderings(self):
        """The probabilities of all n! orderings of a set must sum to one."""
        w = self.some_w(3)
        items = [3, 7, 11]
        total = sum(
            np.exp(preference.log_likelihood(
                w, preference._as_blocks([list(ordering)], self.features)))
            for ordering in itertools.permutations(items))
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_a_ranking_yields_n_minus_one_choice_events(self):
        rng = np.random.default_rng(4)
        ranking = [list(rng.choice(len(self.frame), size=10, replace=False))]
        _w, info = preference.fit(ranking, self.features)
        self.assertEqual(info["comparisons"], 9)

        pair = [list(rng.choice(len(self.frame), size=2, replace=False))]
        _w, info = preference.fit(pair, self.features)
        self.assertEqual(info["comparisons"], 1)

    def test_no_answers_gives_a_zero_vector(self):
        w, info = preference.fit([], self.features)
        np.testing.assert_allclose(w, 0)
        self.assertEqual(info["observations"], 0)

    def test_it_recovers_a_known_preference_vector(self):
        true_w = np.random.default_rng(5).normal(scale=0.8, size=self.dimension)
        sample = self._simulate(true_w, 200, 2, seed=6)
        fitted, info = preference.fit(sample, self.features, alpha=0.05)
        self.assertTrue(info["converged"])
        correlation = float(np.corrcoef(fitted, true_w)[0, 1])
        self.assertGreater(correlation, 0.5)

    def test_recovery_improves_with_more_answers(self):
        true_w = np.random.default_rng(7).normal(scale=0.8, size=self.dimension)
        few = preference.fit(self._simulate(true_w, 10, 2, seed=8),
                             self.features, alpha=0.05)[0]
        many = preference.fit(self._simulate(true_w, 400, 2, seed=8),
                              self.features, alpha=0.05)[0]
        self.assertGreater(float(np.corrcoef(many, true_w)[0, 1]),
                           float(np.corrcoef(few, true_w)[0, 1]))

    def test_an_uninformed_vector_scores_exactly_chance(self):
        """`argmax` on ties would otherwise hand a zero w a perfect score."""
        rng = np.random.default_rng(9)
        held_out = [list(rng.choice(len(self.frame), size=2, replace=False))
                    for _ in range(200)]
        accuracy = preference.predictive_accuracy(
            np.zeros(self.dimension), self.features, held_out)
        self.assertAlmostEqual(accuracy, 0.5, places=9)

    def test_the_true_vector_beats_chance_on_held_out_answers(self):
        true_w = np.random.default_rng(10).normal(scale=0.8, size=self.dimension)
        held_out = self._simulate(true_w, 300, 2, seed=11)
        self.assertGreater(
            preference.predictive_accuracy(true_w, self.features, held_out), 0.7)

    def test_recommendations_are_the_highest_utility_films(self):
        w = self.some_w(12)
        top = preference.top_recommendations(w, self.features, self.frame, count=5)
        scores = preference.utilities(w, self.features)
        self.assertEqual(top[0]["movie_id"], int(np.argmax(scores)))
        self.assertEqual([row["movie_id"] for row in top],
                         sorted([row["movie_id"] for row in top],
                                key=lambda index: -scores[index]))

    def test_exclusions_are_respected(self):
        w = self.some_w(13)
        best = int(np.argmax(preference.utilities(w, self.features)))
        top = preference.top_recommendations(w, self.features, self.frame,
                                             count=3, exclude=[best])
        self.assertNotIn(best, [row["movie_id"] for row in top])

    def test_readable_weights_name_real_features(self):
        strongest, weakest = preference.readable_weights(
            self.some_w(14), self.names, count=4)
        self.assertEqual(len(strongest), 4)
        for row in strongest + weakest:
            self.assertIn(row["name"], self.names)

    def _simulate(self, true_w, count, size, seed):
        """Draw orderings from the Plackett-Luce model itself."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(count):
            candidates = list(rng.choice(len(self.frame), size=size,
                                         replace=False))
            scores = list(self.features[candidates] @ true_w)
            ordering = []
            while len(candidates) > 1:
                probabilities = np.exp(scores - np.max(scores))
                probabilities /= probabilities.sum()
                pick = int(rng.choice(len(candidates), p=probabilities))
                ordering.append(candidates.pop(pick))
                scores.pop(pick)
            ordering.append(candidates[0])
            out.append(ordering)
        return out


@needs_data
class StudyProtocolTests(TestCase):
    """Task 3's protocol, as the code implements it."""

    def test_the_order_is_counterbalanced_across_sessions(self):
        orders = [study.new_session().order for _ in range(6)]
        self.assertEqual(orders.count("pairwise_first"), 3)
        self.assertEqual(orders.count("ranking_first"), 3)

    def test_both_orders_cover_the_same_blocks(self):
        for _ in range(2):
            session = study.new_session()
            self.assertEqual(sorted(study.block_order(session)),
                             sorted([PAIRWISE, RANKING, VALIDATION]))

    def test_validation_always_comes_last(self):
        """It is the yardstick, so it must not be answered before either design."""
        for _ in range(2):
            session = study.new_session()
            self.assertEqual(study.block_order(session)[-1], VALIDATION)

    def test_progress_walks_through_every_block(self):
        session = study.new_session()
        seen = []
        while True:
            block, position = study.progress(session)
            if block is None:
                break
            shown = study.sample_movies(
                movies.load(), study.BLOCK_ITEMS[block],
                study.task_seed(session, block, position))
            study.record(session, block, position, shown, shown, seconds=1.0)
            seen.append(block)
        self.assertEqual(len(seen), study.total_tasks())
        self.assertEqual(seen.count(PAIRWISE), study.PAIRWISE_TASKS)
        self.assertEqual(seen.count(RANKING), study.RANKING_TASKS)
        self.assertEqual(seen.count(VALIDATION), study.VALIDATION_TASKS)

    def test_the_task_seed_is_pinned_and_so_cannot_become_process_dependent(self):
        """The seed is derived in one request and re-derived in the next.

        It was once `hash()`, which is salted per process: a restart, or simply a
        different worker of a multi-process server, produced a different seed and
        so a different pair of films for the same task. Pinning the value here is
        what makes that regression impossible to reintroduce quietly -- a
        same-process round trip cannot detect it.
        """
        session = StudySession(token="fixed-token-for-the-test")
        self.assertEqual(study.task_seed(session, PAIRWISE, 0), 1874854411)

    def test_the_same_task_always_shows_the_same_films(self):
        """A reload must not resample, or a participant could shop for an easier one."""
        session = study.new_session()
        frame = movies.load()
        first = study.sample_movies(frame, 2, study.task_seed(session, PAIRWISE, 0))
        second = study.sample_movies(frame, 2, study.task_seed(session, PAIRWISE, 0))
        self.assertEqual(first, second)

    def test_different_tasks_show_different_films(self):
        session = study.new_session()
        frame = movies.load()
        first = study.sample_movies(frame, 2, study.task_seed(session, PAIRWISE, 0))
        second = study.sample_movies(frame, 2, study.task_seed(session, PAIRWISE, 1))
        self.assertNotEqual(first, second)

    def test_a_ranking_shows_ten_distinct_films(self):
        session = study.new_session()
        shown = study.sample_movies(movies.load(), study.BLOCK_ITEMS[RANKING],
                                    study.task_seed(session, RANKING, 0))
        self.assertEqual(len(shown), study.RANKING_SIZE)
        self.assertEqual(len(set(shown)), study.RANKING_SIZE)

    def test_an_ordering_must_be_a_permutation_of_what_was_shown(self):
        session = study.new_session()
        with self.assertRaises(ValueError):
            study.record(session, PAIRWISE, 0, [1, 2], [1, 3])

    def test_the_same_task_cannot_be_answered_twice(self):
        session = study.new_session()
        self.assertIsNotNone(study.record(session, PAIRWISE, 0, [1, 2], [2, 1]))
        self.assertIsNone(study.record(session, PAIRWISE, 0, [1, 2], [1, 2]))
        self.assertEqual(session.answers.count(), 1)

    def test_an_answer_records_its_ordering(self):
        session = study.new_session()
        study.record(session, PAIRWISE, 0, [7, 9], [9, 7], seconds=2.5)
        answer = session.answers.first()
        self.assertEqual(answer.shown_ids, [7, 9])
        self.assertEqual(answer.ordered_ids, [9, 7])
        self.assertAlmostEqual(answer.seconds, 2.5)

    def test_the_export_includes_every_answer(self):
        session = study.new_session()
        study.record(session, PAIRWISE, 0, [1, 2], [2, 1], seconds=1.0)
        study.record(session, PAIRWISE, 1, [3, 4], [3, 4], seconds=1.0)
        rows = study.export_rows()
        self.assertEqual(len(rows), 3)                  # header plus two answers
        self.assertEqual(rows[0][0], "session")


@needs_data
class InterfaceTests(TestCase):
    """Task 4: the participant's interface."""

    def test_the_landing_page_offers_both_things_the_sheet_asks_for(self):
        page = self.client.get(reverse("project4:index"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, reverse("project4:report"))
        self.assertContains(page, reverse("project4:start"))

    def test_the_landing_page_survives_a_missing_database(self):
        """`db.sqlite3` is gitignored, so a fresh clone has no study tables.

        The page used to raise, giving a 500 with no hint that `migrate` was the
        answer. Project 3 already handles its own missing cache this way.
        """
        with mock.patch("project4.views.StudySession.objects") as manager:
            manager.count.side_effect = OperationalError("no such table")
            manager.filter.side_effect = OperationalError("no such table")
            page = self.client.get(reverse("project4:index"))

        self.assertEqual(page.status_code, 503)
        self.assertContains(page, "manage.py migrate", status_code=503)

    def test_a_long_table_cell_wraps_instead_of_overflowing(self):
        """reportlab does not wrap a plain string in a table cell.

        The feature table's "why" column held whole sentences as plain strings
        in a 9.4cm column, so they were drawn on one line and ran straight off
        the right edge of the page, by 11.6cm in the worst row. Over-long cells
        are promoted to Paragraphs, which wrap inside the column.
        """
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph

        from .report import _wrap_long_cells

        widths = [2.6 * cm, 9.4 * cm]
        sentence = ("Taste in period is real and not monotone: somebody may love "
                    "the 1970s and dislike both the 1950s and the 2010s.")
        rows = [["Block", "Why"], ["Era", sentence], ["Acclaim", "IMDb score."]]

        wrapped = _wrap_long_cells(rows, widths)

        self.assertIsInstance(wrapped[1][1], Paragraph)
        # short cells stay plain strings, so the common case is untouched
        self.assertEqual(wrapped[1][0], "Era")
        self.assertEqual(wrapped[2][1], "IMDb score.")

    @unittest.skipUnless(HAS_PYPDF, "pypdf is not installed")
    def test_no_report_text_runs_past_the_right_margin(self):
        """The whole point of the wrapping: nothing leaves the text column."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfbase.pdfmetrics import stringWidth

        response = self.client.get(reverse("project4:report"))
        reader = PdfReader(io.BytesIO(response.content))
        limit = A4[0] - 2.2 * cm

        widest = 0.0
        for page in reader.pages:
            runs = []

            def visitor(text, cm_matrix, tm_matrix, font_dict, size, runs=runs):
                if not text.strip():
                    return
                x = float(tm_matrix[4]) + float(cm_matrix[4])
                runs.append((round(float(tm_matrix[5]) + float(cm_matrix[5]), 1),
                             round(x, 1), text.strip(), float(size or 10)))

            page.extract_text(visitor_text=visitor)

            # segments sharing a baseline and an x are one drawn run
            extents = {}
            for baseline, x, text, size in runs:
                key = (baseline, x)
                extents[key] = extents.get(key, 0.0) + stringWidth(
                    text, "Helvetica", size)
            for (_baseline, x), width in extents.items():
                widest = max(widest, x + width)

        self.assertLess(widest, limit + 1,
                        f"text reaches x={widest:.0f}, margin is at {limit:.0f}")

    def test_the_report_downloads_as_a_pdf(self):
        response = self.client.get(reverse("project4:report"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertTrue(response.content.startswith(b"%PDF"))
        self.assertGreater(len(response.content), 8000)

    def test_a_task_cannot_be_reached_without_starting(self):
        self.assertRedirects(self.client.get(reverse("project4:task")),
                             reverse("project4:index"))

    def test_starting_gives_the_first_task(self):
        self.client.post(reverse("project4:start"))
        page = self.client.get(reverse("project4:task"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, 'name="block"')

    def test_a_ranking_form_offers_every_rank(self):
        self.client.post(reverse("project4:start"))
        session = StudySession.objects.first()
        session.order = "ranking_first"
        session.save()

        page = self.client.get(reverse("project4:task")).content.decode()
        self.assertEqual(len(re.findall(r'name="rank_(\d+)"', page)),
                         study.RANKING_SIZE)

    def test_duplicate_ranks_are_refused(self):
        self.client.post(reverse("project4:start"))
        session = StudySession.objects.first()
        session.order = "ranking_first"
        session.save()

        page = self.client.get(reverse("project4:task")).content.decode()
        ids = re.findall(r'name="rank_(\d+)"', page)
        payload = {"block": RANKING, "position": 0}
        for movie_id in ids:
            payload[f"rank_{movie_id}"] = "1"

        response = self.client.post(reverse("project4:task"), payload)
        self.assertContains(response, "exactly once")
        self.assertEqual(Answer.objects.count(), 0)

    def test_a_valid_ranking_is_stored_in_the_order_given(self):
        self.client.post(reverse("project4:start"))
        session = StudySession.objects.first()
        session.order = "ranking_first"
        session.save()

        page = self.client.get(reverse("project4:task")).content.decode()
        ids = re.findall(r'name="rank_(\d+)"', page)
        payload = {"block": RANKING, "position": 0}
        # Reverse the presented order, so the stored ordering is not the identity.
        for rank, movie_id in enumerate(reversed(ids), start=1):
            payload[f"rank_{movie_id}"] = str(rank)

        self.client.post(reverse("project4:task"), payload)
        answer = Answer.objects.get()
        self.assertEqual(answer.ordered_ids, [int(value) for value in reversed(ids)])

    def test_a_choice_outside_the_pair_is_refused(self):
        self.client.post(reverse("project4:start"))
        page = self.client.get(reverse("project4:task")).content.decode()
        block = re.search(r'name="block" value="([^"]+)"', page).group(1)
        response = self.client.post(reverse("project4:task"),
                                    {"block": block, "position": 0,
                                     "choice": "99999999"})
        self.assertContains(response, "choose one")
        self.assertEqual(Answer.objects.count(), 0)

    def test_a_pairwise_answer_puts_the_chosen_film_first(self):
        self.client.post(reverse("project4:start"))
        page = self.client.get(reverse("project4:task")).content.decode()
        ids = re.findall(r'name="choice"\s+value="(\d+)"', page)
        self.assertEqual(len(ids), 2)
        self.client.post(reverse("project4:task"),
                         {"block": PAIRWISE, "position": 0, "choice": ids[1]})
        answer = Answer.objects.get()
        self.assertEqual(answer.ordered_ids[0], int(ids[1]))

    def test_results_are_refused_until_the_study_is_finished(self):
        self.client.post(reverse("project4:start"))
        self.assertRedirects(self.client.get(reverse("project4:results")),
                             reverse("project4:task"))

    def test_a_whole_session_can_be_completed_and_analysed(self):
        self.client.post(reverse("project4:start"))

        for _ in range(study.total_tasks() + 5):
            page = self.client.get(reverse("project4:task"))
            if page.status_code != 200:
                break
            html = page.content.decode()
            block = re.search(r'name="block" value="([^"]+)"', html).group(1)
            position = int(re.search(r'name="position" value="(\d+)"',
                                     html).group(1))

            if block == RANKING:
                ids = re.findall(r'name="rank_(\d+)"', html)
                payload = {"block": block, "position": position}
                for rank, movie_id in enumerate(ids, start=1):
                    payload[f"rank_{movie_id}"] = str(rank)
            else:
                ids = re.findall(r'name="choice"\s+value="(\d+)"', html)
                payload = {"block": block, "position": position,
                           "choice": ids[0]}
            self.client.post(reverse("project4:task"), payload)

        session = StudySession.objects.get()
        self.assertEqual(session.answers.count(), study.total_tasks())
        self.assertIsNotNone(session.finished_at)

        page = self.client.get(reverse("project4:results"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Design 1: pairwise choice")
        self.assertContains(page, "Design 2: rank ten")
        self.assertContains(page, "Held-out accuracy")

    def test_the_analysis_scores_both_designs_on_held_out_answers(self):
        frame = movies.load()
        features, names = movies.build_features(frame)
        session = study.new_session()

        for block in study.block_order(session):
            for position in range(study.BLOCK_SIZES[block]):
                shown = study.sample_movies(
                    frame, study.BLOCK_ITEMS[block],
                    study.task_seed(session, block, position))
                study.record(session, block, position, shown, shown, seconds=1.0)

        analysis = study.analyse(session, frame, features, names)
        self.assertEqual(analysis["held_out"], study.VALIDATION_TASKS)
        for block in (PAIRWISE, RANKING):
            self.assertEqual(analysis[block]["answers"],
                             study.BLOCK_SIZES[block])
            self.assertGreaterEqual(analysis[block]["accuracy"], 0.0)
            self.assertLessEqual(analysis[block]["accuracy"], 1.0)
        self.assertEqual(analysis[RANKING]["choice_events"],
                         study.RANKING_TASKS * (study.RANKING_SIZE - 1))

    def test_the_csv_export_downloads(self):
        response = self.client.get(reverse("project4:data"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("attachment", response["Content-Disposition"])

    def test_no_template_syntax_leaks(self):
        self.client.post(reverse("project4:start"))
        for name in ("project4:index", "project4:task"):
            page = self.client.get(reverse(name)).content.decode()
            for leaked in ("{#", "#}", "{%", "%}", "{{", "}}"):
                self.assertNotIn(leaked, page, name)
