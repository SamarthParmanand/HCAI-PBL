"""Tests for Project 3 (learning to defer, active learning).

Run with:  python manage.py test project3

Nothing here fits on the full 120,000 articles: the components are tested
directly, and the views are tested against a small bundle built once for the
test run. MEDIA_ROOT is redirected to a temporary directory.
"""

import copy
import io
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from django.test import TestCase, override_settings
from django.urls import reverse

from . import active, classifier, data, defer, experts

TEMP_MEDIA = tempfile.mkdtemp(prefix="project3-tests-")

# Small enough to keep the suite quick, large enough for a real fit.
SMALL = 2000

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:                                       # pragma: no cover
    HAS_PYPDF = False


def _pdf_text(content):
    """The text of a generated PDF, for asserting on what the reader sees."""
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


HAS_DATA = data.is_cached()
needs_data = unittest.skipUnless(
    HAS_DATA, "AG News is not cached; run: python manage.py build_project3")


@needs_data
class DatasetTests(TestCase):

    def test_the_test_set_matches_the_published_dataset(self):
        _train, test = data.load(limit=SMALL)
        self.assertEqual(len(test), 7600)
        self.assertEqual(len(data.CLASS_NAMES), 4)
        self.assertEqual(sorted(test["label"].unique()), [0, 1, 2, 3])

    def test_the_test_set_is_balanced(self):
        _train, test = data.load(limit=SMALL)
        counts = test["label"].value_counts()
        self.assertEqual(set(counts), {1900})

    def test_the_subsample_is_stratified_and_keeps_its_columns(self):
        train, _test = data.load(limit=SMALL)
        self.assertEqual(list(train.columns), ["text", "label"])
        self.assertEqual(len(train), SMALL)
        self.assertEqual(set(train["label"].value_counts()), {SMALL // 4})

    def test_the_subsample_is_reproducible(self):
        first, _ = data.load(limit=SMALL, seed=3)
        second, _ = data.load(limit=SMALL, seed=3)
        self.assertEqual(first["text"].tolist(), second["text"].tolist())


class ExpertTests(TestCase):
    """Task 2."""

    def setUp(self):
        self.texts = np.array([
            "stocks fell as investors weighed earnings",
            "the midfielder scored twice before halftime",
            "a new linux server release shipped today",
            "the president met regional leaders",
        ] * 25, dtype=object)
        self.labels = np.array([2, 1, 3, 0] * 25)

    def test_the_topic_expert_is_strong_only_on_its_beat(self):
        expert = experts.TopicExpert(specialties=(1, 2), strong=0.95, weak=0.45)
        report = expert.report(self.texts, self.labels)
        self.assertGreater(report["accuracy_in_region"],
                           report["accuracy_out_of_region"])
        self.assertAlmostEqual(report["region_share"], 0.5, places=6)

    def test_the_keyword_expert_region_is_read_from_the_text(self):
        expert = experts.KeywordExpert()
        inside = expert.in_region(self.texts, self.labels)
        # "stocks"/"investors"/"earnings" and "linux"/"server" are markers;
        # the sport and politics lines are not.
        self.assertTrue(inside[0])
        self.assertTrue(inside[2])
        self.assertFalse(inside[1])
        self.assertFalse(inside[3])

    def test_the_expert_is_deterministic(self):
        """The active-learning loop may query a point twice; it must not re-roll."""
        expert = experts.build("topic", seed=1)
        first = expert.predict(self.texts, self.labels)
        second = expert.predict(self.texts, self.labels)
        np.testing.assert_array_equal(first, second)

    def test_predictions_are_always_valid_labels(self):
        for kind in ("topic", "keyword"):
            predicted = experts.build(kind).predict(self.texts, self.labels)
            self.assertTrue(set(predicted).issubset(set(range(4))), kind)

    def test_a_wrong_expert_names_a_different_class(self):
        expert = experts.TopicExpert(specialties=(), strong=0.0, weak=0.0)
        predicted = expert.predict(self.texts, self.labels)
        self.assertFalse(np.any(predicted == self.labels))

    def test_an_expert_is_not_perfect(self):
        report = experts.build("topic").report(self.texts, self.labels)
        self.assertLess(report["accuracy"], 1.0)

    def test_an_unknown_expert_is_refused(self):
        with self.assertRaises(ValueError):
            experts.build("nobody")


class DeferralArithmeticTests(TestCase):
    """Task 3: the parts that are pure bookkeeping, checked by hand."""

    def setUp(self):
        # Four cases, one of each kind.
        self.classifier_correct = np.array([True, True, False, False])
        self.expert_correct = np.array([True, False, True, False])

    def test_the_oracle_defers_only_where_it_helps(self):
        oracle = defer.oracle_decisions(self.classifier_correct,
                                        self.expert_correct)
        np.testing.assert_array_equal(oracle, [False, False, True, False])

    def test_never_deferring_scores_the_classifier(self):
        scores = defer.evaluate_policy(np.zeros(4, dtype=bool),
                                       self.classifier_correct,
                                       self.expert_correct)
        self.assertAlmostEqual(scores["team_accuracy"], 0.5)
        self.assertAlmostEqual(scores["coverage"], 1.0)

    def test_always_deferring_scores_the_expert(self):
        scores = defer.evaluate_policy(np.ones(4, dtype=bool),
                                       self.classifier_correct,
                                       self.expert_correct)
        self.assertAlmostEqual(scores["team_accuracy"], 0.5)
        self.assertAlmostEqual(scores["deferral_rate"], 1.0)

    def test_the_oracle_policy_reaches_the_ceiling(self):
        oracle = defer.oracle_decisions(self.classifier_correct,
                                        self.expert_correct)
        scores = defer.evaluate_policy(oracle, self.classifier_correct,
                                       self.expert_correct)
        self.assertAlmostEqual(scores["team_accuracy"], 0.75)
        self.assertEqual(scores["rescued"], 1)
        self.assertEqual(scores["spoiled"], 0)
        self.assertAlmostEqual(scores["deferral_f1"], 1.0)

    def test_rescued_and_spoiled_are_counted_separately(self):
        # Defer everything: one rescue, one answer thrown away.
        scores = defer.evaluate_policy(np.ones(4, dtype=bool),
                                       self.classifier_correct,
                                       self.expert_correct)
        self.assertEqual(scores["rescued"], 1)
        self.assertEqual(scores["spoiled"], 1)
        self.assertEqual(scores["net_gain"], 0)

    def test_the_confidence_policy_defers_the_unsure_cases(self):
        mask = defer.confidence_policy([0.9, 0.4, 0.6], 0.5)
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_the_competence_policy_needs_a_margin_to_be_beaten(self):
        expert = np.array([0.8, 0.8])
        classifier_ = np.array([0.7, 0.7])
        np.testing.assert_array_equal(
            defer.competence_policy(expert, classifier_, margin=0.0),
            [True, True])
        np.testing.assert_array_equal(
            defer.competence_policy(expert, classifier_, margin=0.2),
            [False, False])

    def test_a_sweep_covers_the_whole_range(self):
        rows = defer.sweep_confidence([0.2, 0.5, 0.9], [True, False, True],
                                      [False, True, False])
        self.assertEqual(len(rows), 51)
        self.assertAlmostEqual(rows[0]["deferral_rate"], 0.0)
        self.assertAlmostEqual(rows[-1]["deferral_rate"], 1.0)

    def test_best_by_team_accuracy_picks_the_maximum(self):
        rows = [{"team_accuracy": 0.5, "coverage": 1.0},
                {"team_accuracy": 0.9, "coverage": 0.5},
                {"team_accuracy": 0.7, "coverage": 0.9}]
        self.assertEqual(defer.best_by_team_accuracy(rows)["team_accuracy"], 0.9)

    def test_a_single_class_target_cannot_be_fitted(self):
        self.assertIsNone(defer.fit_competence(["a", "b"], [1, 1]))

    def test_a_missing_model_falls_back_to_the_base_rate(self):
        probability = defer.competence_probability(None, ["a", "b"], 0.42)
        np.testing.assert_allclose(probability, [0.42, 0.42])


class CalibrationTests(TestCase):
    """Turning a softmax maximum into a probability of being right."""

    def setUp(self):
        rng = np.random.default_rng(0)
        # Confidence that genuinely predicts correctness, but overstated.
        self.confidence = rng.uniform(0.4, 1.0, size=2000)
        self.correct = rng.uniform(size=2000) < (self.confidence * 0.8)

    def test_calibration_improves_the_probability_scale(self):
        model = defer.fit_confidence_calibration(self.confidence, self.correct)
        calibrated = defer.calibrated_probability(model, self.confidence, 0.5)

        # Ranking is preserved, so the AUC should not fall...
        raw_auc = defer.competence_quality(self.confidence, self.correct)
        new_auc = defer.competence_quality(calibrated, self.correct)
        self.assertAlmostEqual(raw_auc, new_auc, places=6)

        # ...but the mean should now match the actual rate of being correct.
        self.assertLess(abs(calibrated.mean() - self.correct.mean()), 0.02)
        self.assertGreater(abs(self.confidence.mean() - self.correct.mean()), 0.05)

    def test_calibration_is_monotone_in_the_confidence(self):
        model = defer.fit_confidence_calibration(self.confidence, self.correct)
        grid = np.linspace(0.3, 0.99, 40)
        values = defer.calibrated_probability(model, grid, 0.5)
        self.assertTrue(np.all(np.diff(values) >= -1e-9))

    def test_a_single_class_target_cannot_be_calibrated(self):
        self.assertIsNone(
            defer.fit_confidence_calibration([0.5, 0.9], [1, 1]))


class ActiveLearningTests(TestCase):
    """Task 4."""

    # Topic words, not digits: scikit-learn's default token pattern needs two or
    # more word characters, so a single-digit topic marker would be dropped and
    # the competence would be unlearnable for reasons that have nothing to do
    # with the strategy under test.
    TOPICS = ["finance", "athletics", "hardware", "diplomacy",
              "cinema", "weather", "shipping"]

    def setUp(self):
        rng = np.random.default_rng(0)
        self.pool_texts = np.array(
            [f"an article concerning {self.TOPICS[index % 7]} and other matters"
             for index in range(400)], dtype=object)
        # Expert competence depends on the topic, so it is learnable from the text.
        self.pool_correct = np.array([(index % 7) < 3 for index in range(400)])
        self.pool_probability = rng.uniform(0.5, 1.0, size=400)

        self.test_texts = np.array(
            [f"a report concerning {self.TOPICS[index % 7]} and further details"
             for index in range(200)], dtype=object)
        self.test_expert_correct = np.array([(index % 7) < 3
                                             for index in range(200)])
        self.test_classifier_correct = rng.uniform(size=200) < 0.8
        self.test_probability = rng.uniform(0.5, 1.0, size=200)

    def run_strategy(self, strategy, rounds=3, batch=30):
        return active.run(
            strategy,
            pool_texts=self.pool_texts,
            pool_expert_correct=self.pool_correct,
            pool_classifier_probability=self.pool_probability,
            test_texts=self.test_texts,
            test_classifier_probability=self.test_probability,
            test_classifier_correct=self.test_classifier_correct,
            test_expert_correct=self.test_expert_correct,
            rounds=rounds, batch=batch,
        )

    def test_every_strategy_runs_and_spends_its_budget(self):
        for strategy in active.STRATEGIES:
            history = self.run_strategy(strategy)
            self.assertEqual(len(history), 3, strategy)
            self.assertEqual([row["queries"] for row in history], [30, 60, 90],
                             strategy)

    def test_the_hybrid_mixes_exploration_into_every_batch(self):
        self.assertGreater(active.EXPLORE_FRACTION["hybrid_boundary"], 0.0)
        self.assertEqual(active.EXPLORE_FRACTION["decision_boundary"], 0.0)
        self.assertEqual(active.EXPLORE_FRACTION["random"], 1.0)

    def test_queries_never_repeat_a_point(self):
        """A budget is only spent once; re-querying would inflate the count."""
        for strategy in active.STRATEGIES:
            rng = np.random.default_rng(0)
            remaining = np.arange(50)
            picks = active._select(strategy, 10, remaining, rng,
                                   self.pool_probability[:50], None)
            self.assertEqual(len(set(picks.tolist())), len(picks), strategy)

    def test_the_expert_model_learns_a_learnable_competence(self):
        history = self.run_strategy("random", rounds=4, batch=60)
        self.assertGreater(history[-1]["expert_model_auc"], 0.7)

    def test_queries_to_reach_finds_the_first_round_over_target(self):
        history = [{"queries": 10, "team_accuracy": 0.5},
                   {"queries": 20, "team_accuracy": 0.8},
                   {"queries": 30, "team_accuracy": 0.9}]
        self.assertEqual(active.queries_to_reach(history, 0.75), 20)
        self.assertIsNone(active.queries_to_reach(history, 0.99))

    def test_summarise_reports_every_strategy(self):
        runs = {strategy: self.run_strategy(strategy)
                for strategy in ("random", "hybrid_boundary")}
        summary = active.summarise(runs, target=0.5)
        self.assertEqual(set(summary), {"random", "hybrid_boundary"})
        for row in summary.values():
            self.assertIn("expert_model_auc", row)
            self.assertIn("label", row)

    def test_an_unknown_strategy_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_strategy("telepathy")


@needs_data
class ClassifierTests(TestCase):
    """Task 1, on a subsample so the suite stays quick."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.train, cls.test = data.load(limit=SMALL)
        cls.model = classifier.fit(cls.train["text"], cls.train["label"])
        cls.small_test = cls.test.sample(n=1500, random_state=0)

    def test_it_beats_chance_by_a_wide_margin(self):
        scores = classifier.evaluate(self.model, self.small_test["text"],
                                     self.small_test["label"])
        self.assertGreater(scores["accuracy"], 0.8)

    def test_the_confusion_matrix_totals_the_rows(self):
        scores = classifier.evaluate(self.model, self.small_test["text"],
                                     self.small_test["label"])
        total = sum(sum(row) for row in scores["matrix"])
        self.assertEqual(total, len(self.small_test))

    def test_confidence_is_a_probability(self):
        values = classifier.confidence(self.model, self.small_test["text"][:200])
        self.assertTrue(np.all(values > 0.24))       # at least chance for 4 classes
        self.assertTrue(np.all(values <= 1.0))

    def test_it_reports_every_class(self):
        scores = classifier.evaluate(self.model, self.small_test["text"],
                                     self.small_test["label"])
        self.assertEqual(set(scores["per_class"]), set(data.CLASS_NAMES))


@needs_data
@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class InterfaceTests(TestCase):
    """The pages, against a small bundle rather than the full experiment."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from . import artifacts
        cls.bundle = artifacts.build(limit=SMALL, rounds=2, batch=25,
                                     pool_size=200, save=False, log=lambda _m: None)

    def setUp(self):
        patcher = mock.patch("project3.artifacts.load", return_value=self.bundle)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_results_page_renders(self):
        page = self.client.get(reverse("project3:index"))
        self.assertEqual(page.status_code, 200)
        for expected in ("Task 1", "Task 2", "Task 3", "Task 4",
                         "learning-to-defer"):
            self.assertContains(page, expected)

    def test_it_reports_all_four_tasks_numerically(self):
        page = self.client.get(reverse("project3:index")).content.decode()
        self.assertIn("Oracle ceiling", page)
        self.assertIn("Expert model AUC", page)
        self.assertIn("Calibrated confidence", page)

    def test_the_report_downloads_as_a_pdf(self):
        response = self.client.get(reverse("project3:report"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertTrue(response.content.startswith(b"%PDF"))
        self.assertGreater(len(response.content), 20000)

    @unittest.skipUnless(HAS_PYPDF, "pypdf is not installed")
    def test_the_report_never_prints_a_literal_none(self):
        """`None` on the page means a value reached the text unformatted.

        `active.queries_to_reach` returns None for a strategy that never hits the
        target, and that value used to be interpolated straight into a sentence
        as "after None labels".
        """
        response = self.client.get(reverse("project3:report"))
        text = _pdf_text(response.content)
        self.assertGreater(len(text), 2000)
        self.assertNotIn("None", text)

    @unittest.skipUnless(HAS_PYPDF, "pypdf is not installed")
    def test_the_report_survives_a_strategy_that_never_reaches_the_target(self):
        """The case that produced the bug, forced rather than waited for."""
        bundle = copy.deepcopy(self.bundle)
        for entry in bundle["active_summary"].values():
            entry["queries_to_target"] = None

        with mock.patch("project3.artifacts.load", return_value=bundle):
            response = self.client.get(reverse("project3:report"))
        self.assertEqual(response.status_code, 200)
        text = _pdf_text(response.content)
        self.assertNotIn("None", text)
        self.assertIn("not within the budget tried", text)

    def test_the_expert_page_asks_a_question_and_scores_it(self):
        page = self.client.get(reverse("project3:expert"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Which topic is this?")

        # Answer every article in the round.
        for _ in range(20):
            page = self.client.get(reverse("project3:expert"))
            if "Which topic is this?" not in page.content.decode():
                break
            self.client.post(reverse("project3:expert"), {"answer": "0"})

        page = self.client.get(reverse("project3:expert"))
        self.assertContains(page, "What everyone said")

    def test_a_nonsense_answer_is_ignored(self):
        before = self.client.get(reverse("project3:expert")).content.decode()
        self.client.post(reverse("project3:expert"), {"answer": "99"})
        after = self.client.get(reverse("project3:expert")).content.decode()
        # Still on the same question rather than having advanced.
        self.assertEqual("Which topic is this?" in before,
                         "Which topic is this?" in after)

    def test_no_template_syntax_leaks(self):
        for name in ("project3:index", "project3:expert"):
            page = self.client.get(reverse(name)).content.decode()
            for leaked in ("{#", "#}", "{%", "%}", "{{", "}}"):
                self.assertNotIn(leaked, page, name)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)


class NotBuiltTests(TestCase):
    """The page must explain itself rather than hang or crash."""

    def test_it_tells_you_the_command(self):
        with mock.patch("project3.artifacts.load", return_value=None):
            page = self.client.get(reverse("project3:index"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "manage.py build_project3")
