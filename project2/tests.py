"""Tests for Project 2 (Explainability).

Run with:  python manage.py test project2

MEDIA_ROOT is redirected to a temporary directory throughout, so running the
tests never writes into the real media/ directory.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from django.test import TestCase, override_settings
from django.urls import reverse

from . import counterfactuals as cf
from . import effects, learning, penguins

TEMP_MEDIA = tempfile.mkdtemp(prefix="project2-tests-")


class DatasetTests(TestCase):

    def test_the_dataset_matches_the_task_sheet(self):
        frame = penguins.load()
        self.assertEqual(penguins.TARGET, "species")
        self.assertEqual(sorted(frame["species"].unique()),
                         ["Adelie", "Chinstrap", "Gentoo"])
        self.assertEqual(len(penguins.NUMERIC_FEATURES), 4)
        self.assertEqual(len(penguins.FEATURES), 7)

    def test_incomplete_rows_are_dropped_and_counted(self):
        frame = penguins.load()
        self.assertFalse(frame.isna().any().any())
        self.assertEqual(frame.attrs["total_rows"] - frame.attrs["dropped_rows"],
                         len(frame))
        self.assertEqual(len(frame), 333)

    def test_year_is_treated_as_a_label(self):
        """2007..2009 is not a quantity; leaving it numeric would imply an order."""
        frame = penguins.load()
        self.assertIn("year", penguins.CATEGORICAL_FEATURES)
        self.assertNotIn("year", penguins.NUMERIC_FEATURES)
        self.assertFalse(pd.api.types.is_numeric_dtype(frame["year"]))

    def test_mad_is_positive_for_every_measurement(self):
        deviations = penguins.median_absolute_deviations(penguins.load())
        for name in penguins.NUMERIC_FEATURES:
            self.assertGreater(deviations[name], 0, name)

    def test_species_order_is_fixed(self):
        frame = penguins.load()
        self.assertEqual(penguins.class_names(frame),
                         ["Adelie", "Chinstrap", "Gentoo"])


class ModelFamilyTests(TestCase):
    """Tasks 1 to 3."""

    def test_tree_family_spans_the_leaf_grid(self):
        family = learning.fit_family(learning.TREE)["family"]
        self.assertEqual(len(family), len(learning.LEAF_LIMITS))
        for model in family:
            self.assertLessEqual(model.omega, model.setting)
            self.assertGreaterEqual(model.omega, 2)

    def test_tree_omega_is_the_realised_leaf_count(self):
        """The cap is an upper bound; the achieved count can be lower."""
        family = learning.fit_family(learning.TREE)["family"]
        for model in family:
            self.assertEqual(model.omega, model.estimator.get_n_leaves())

    def test_logistic_family_gets_sparser_as_the_penalty_grows(self):
        family = learning.fit_family(learning.LOGREG)["family"]
        by_strength = sorted(family, key=lambda model: model.setting)
        self.assertLess(by_strength[0].omega, by_strength[-1].omega)

    def test_logistic_omega_counts_non_zero_coefficients(self):
        family = learning.fit_family(learning.LOGREG)["family"]
        for model in family:
            self.assertEqual(model.omega,
                             int(np.count_nonzero(model.estimator.coef_)))

    def test_a_useful_tree_is_found(self):
        family = learning.fit_family(learning.TREE)["family"]
        self.assertGreater(max(model.accuracy_test for model in family), 0.9)

    def test_a_useful_logistic_model_is_found(self):
        family = learning.fit_family(learning.LOGREG)["family"]
        self.assertGreater(max(model.accuracy_test for model in family), 0.9)

    def test_the_family_is_cached_so_the_slider_never_refits(self):
        first = learning.fit_family(learning.TREE)
        second = learning.fit_family(learning.TREE)
        self.assertIs(first, second)

    def test_numeric_columns_come_first_in_the_transformed_matrix(self):
        """The exact-gradient code depends on this mapping."""
        model = learning.fit_family(learning.LOGREG)["family"][-1]
        for position, name in enumerate(penguins.NUMERIC_FEATURES):
            self.assertEqual(
                learning.numeric_column_index(model.pipeline, name), position)

    def test_an_unknown_model_class_is_refused(self):
        with self.assertRaises(ValueError):
            learning.fit_family("nonsense")


class TaskOneTreeTests(TestCase):
    """Task 1 asks for a decision tree, its test accuracy and its leaf count.

    It must not be answered by a member of the Task 2 family: every one of those
    is capped by `max_leaf_nodes`, so none of them is the unconstrained tree the
    task describes.
    """

    def test_the_task_one_tree_is_unconstrained(self):
        tree = learning.fit_default_tree()
        self.assertIsNone(tree.estimator.max_leaf_nodes)
        self.assertEqual(tree.setting, None)

    def test_it_is_grown_to_purity(self):
        tree = learning.fit_default_tree()
        self.assertEqual(tree.accuracy_train, 1.0)

    def test_its_omega_is_the_realised_leaf_count(self):
        tree = learning.fit_default_tree()
        self.assertEqual(tree.omega, tree.estimator.get_n_leaves())
        self.assertGreater(tree.omega, max(learning.LEAF_LIMITS[:1]))

    def test_the_page_shows_its_accuracy_and_leaf_count(self):
        tree = learning.fit_default_tree()
        page = self.client.get(reverse("project2:index"))
        self.assertContains(page, "Task 1")
        self.assertContains(page, f"{tree.accuracy_test:.3f}")
        self.assertContains(page, f">{tree.omega}<")

    def test_it_does_not_move_with_lambda(self):
        """It belongs to Task 1, so Task 2's slider must not change it."""
        low = self.client.get(reverse("project2:index"), {"lam": "0.0"})
        high = self.client.get(reverse("project2:index"),
                               {"lam": f"{learning.LAMBDA_MAX}"})
        tree = learning.fit_default_tree()
        for page in (low, high):
            self.assertContains(page, f"{tree.accuracy_train:.3f}")
            self.assertContains(page, f">{tree.omega}<")


class LambdaSelectionTests(TestCase):
    """Task 2: the slider maximises acc_test - lambda * Omega."""

    def setUp(self):
        self.family = learning.fit_family(learning.TREE)["family"]

    def test_zero_lambda_picks_the_most_accurate_model(self):
        chosen = learning.select(self.family, 0.0)
        self.assertEqual(chosen.accuracy_test,
                         max(model.accuracy_test for model in self.family))

    def test_a_large_lambda_picks_a_far_simpler_model(self):
        """The top of the slider need not reach the degenerate model.

        At lambda = 0.05 a three-leaf tree still beats a two-leaf one, because the
        accuracy it buys (about 0.14) outweighs the extra leaf (0.05). Reaching the
        two-leaf tree would need lambda above 0.14; the range is deliberately kept
        where the useful trade-offs are instead.
        """
        loose = learning.select(self.family, 0.0)
        tight = learning.select(self.family, learning.LAMBDA_MAX)
        self.assertLess(tight.omega, loose.omega / 2)
        self.assertLessEqual(tight.omega, 4)

    def test_the_selected_model_really_maximises_the_objective(self):
        for lam in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
            chosen = learning.select(self.family, lam)
            best = max(model.objective(lam) for model in self.family)
            self.assertAlmostEqual(chosen.objective(lam), best, places=12,
                                   msg=f"lambda={lam}")

    def test_ties_go_to_the_simpler_model(self):
        for lam in (0.0, 0.003, 0.01):
            chosen = learning.select(self.family, lam)
            best = chosen.objective(lam)
            for model in self.family:
                if abs(model.objective(lam) - best) < 1e-12:
                    self.assertLessEqual(chosen.omega, model.omega)

    def test_complexity_never_increases_as_lambda_grows(self):
        """Paying more per leaf can never buy a bigger tree."""
        omegas = [learning.select(self.family, lam / 1000).omega
                  for lam in range(0, 51)]
        self.assertEqual(omegas, sorted(omegas, reverse=True))

    def test_both_families_have_a_real_trade_off_to_show(self):
        for key, _ in learning.MODEL_CLASSES:
            family = learning.fit_family(key)["family"]
            points = learning.switch_points(family)
            self.assertGreater(len(points), 1, f"{key} never changes its mind")


class ExactGradientTests(TestCase):
    """Task 5: the derivative question."""

    def setUp(self):
        data = learning.fit_family(learning.LOGREG)
        self.frame = data["frame"]
        self.classes = data["classes"]
        self.model = learning.select(data["family"], 0.0)

    def test_the_closed_form_gradient_matches_a_finite_difference(self):
        """This is what makes 'exact' an actual claim rather than a hope."""
        for feature in penguins.NUMERIC_FEATURES:
            exact = effects.exact_gradient(self.model.pipeline, self.frame,
                                           feature, self.classes)
            numeric = effects.numerical_gradient(self.model.pipeline, self.frame,
                                                 feature, self.classes)
            self.assertLess(float(np.max(np.abs(exact - numeric))), 1e-6, feature)

    def test_gradients_sum_to_zero_across_classes(self):
        """The three probabilities always sum to one, so their slopes cancel."""
        gradients = effects.exact_gradient(self.model.pipeline, self.frame,
                                           "bill_length_mm", self.classes)
        self.assertLess(float(np.max(np.abs(gradients.sum(axis=1)))), 1e-10)

    def test_a_tree_has_no_usable_derivative(self):
        """Why the tree must be discretised, measured rather than asserted."""
        data = learning.fit_family(learning.TREE)
        tree = learning.select(data["family"], 0.0)
        gradients = effects.numerical_gradient(tree.pipeline, self.frame,
                                               "bill_length_mm", self.classes)
        self.assertEqual(float(np.max(np.abs(gradients))), 0.0)


class PartialDependenceTests(TestCase):

    def setUp(self):
        data = learning.fit_family(learning.TREE)
        self.frame = data["frame"]
        self.classes = data["classes"]
        self.model = learning.select(data["family"], 0.0)

    def test_there_is_one_curve_per_species(self):
        pdp = effects.partial_dependence(self.model.pipeline, self.frame,
                                         "bill_length_mm", self.classes)
        self.assertEqual(len(pdp["curves"]), 3)
        for name in self.classes:
            self.assertEqual(len(pdp["curves"][name]), len(pdp["x"]))

    def test_probabilities_sum_to_one_at_every_grid_point(self):
        pdp = effects.partial_dependence(self.model.pipeline, self.frame,
                                         "flipper_length_mm", self.classes)
        totals = np.sum([pdp["curves"][name] for name in self.classes], axis=0)
        self.assertTrue(np.allclose(totals, 1.0))

    def test_the_grid_covers_the_observed_range(self):
        pdp = effects.partial_dependence(self.model.pipeline, self.frame,
                                         "body_mass_g", self.classes)
        self.assertAlmostEqual(min(pdp["x"]), float(self.frame["body_mass_g"].min()))
        self.assertAlmostEqual(max(pdp["x"]), float(self.frame["body_mass_g"].max()))

    def test_a_categorical_feature_is_refused(self):
        with self.assertRaises(ValueError):
            effects.partial_dependence(self.model.pipeline, self.frame, "island",
                                       self.classes)


class AccumulatedLocalEffectsTests(TestCase):

    def setUp(self):
        self.tree_data = learning.fit_family(learning.TREE)
        self.logreg_data = learning.fit_family(learning.LOGREG)
        self.frame = self.tree_data["frame"]
        self.classes = self.tree_data["classes"]
        self.tree = learning.select(self.tree_data["family"], 0.0)
        self.logreg = learning.select(self.logreg_data["family"], 0.0)

    def curves(self, result):
        return np.array([result["curves"][name] for name in self.classes])

    def test_the_curve_is_centred_on_the_data(self):
        for model, exact in ((self.tree, False), (self.logreg, True)):
            result = effects.accumulated_local_effects(
                model.pipeline, self.frame, "bill_length_mm", self.classes,
                exact=exact)
            curves = self.curves(result)
            counts = np.array(result["counts"])
            midpoints = (curves[:, :-1] + curves[:, 1:]) / 2.0
            mean = (midpoints * counts).sum(axis=1) / counts.sum()
            self.assertLess(float(np.max(np.abs(mean))), 1e-9)

    def test_effects_cancel_across_the_three_species(self):
        result = effects.accumulated_local_effects(
            self.tree.pipeline, self.frame, "bill_depth_mm", self.classes)
        self.assertLess(float(np.max(np.abs(self.curves(result).sum(axis=0)))), 1e-9)

    def test_every_bin_contains_data(self):
        """Quantile edges are chosen precisely so no bin is estimated from nothing."""
        result = effects.accumulated_local_effects(
            self.tree.pipeline, self.frame, "bill_length_mm", self.classes)
        self.assertTrue(all(count > 0 for count in result["counts"]))

    def test_the_curve_has_one_point_per_bin_edge(self):
        result = effects.accumulated_local_effects(
            self.tree.pipeline, self.frame, "body_mass_g", self.classes)
        self.assertEqual(len(result["x"]), len(result["counts"]) + 1)
        for name in self.classes:
            self.assertEqual(len(result["curves"][name]), len(result["x"]))

    def test_exact_and_discrete_agree_for_logistic_regression(self):
        """Two independent routes to the same curve: each checks the other."""
        for feature in penguins.NUMERIC_FEATURES:
            exact = effects.accumulated_local_effects(
                self.logreg.pipeline, self.frame, feature, self.classes, exact=True)
            discrete = effects.accumulated_local_effects(
                self.logreg.pipeline, self.frame, feature, self.classes, exact=False)
            difference = np.max(np.abs(self.curves(exact) - self.curves(discrete)))
            self.assertLess(float(difference), 0.02, feature)

    def test_the_method_is_reported(self):
        exact = effects.accumulated_local_effects(
            self.logreg.pipeline, self.frame, "bill_length_mm", self.classes,
            exact=True)
        discrete = effects.accumulated_local_effects(
            self.tree.pipeline, self.frame, "bill_length_mm", self.classes)
        self.assertIn("derivative", exact["method"])
        self.assertIn("finite difference", discrete["method"])


class CounterfactualTests(TestCase):
    """Task 4."""

    def setUp(self):
        data = learning.fit_family(learning.TREE)
        self.frame = data["frame"]
        self.classes = data["classes"]
        self.model = learning.select(data["family"], learning.DEFAULT_LAMBDA)
        self.deviations = penguins.median_absolute_deviations(self.frame)

    def example(self, index=0):
        return self.frame.loc[index, penguins.FEATURES]

    def test_distance_to_itself_is_zero(self):
        example = self.example()
        same = pd.DataFrame([example], columns=penguins.FEATURES)
        self.assertAlmostEqual(float(cf.distances(example, same, self.deviations)[0]), 0.0)

    def test_one_mad_of_a_measurement_costs_one(self):
        example = self.example()
        moved = pd.DataFrame([example], columns=penguins.FEATURES)
        moved.loc[:, "bill_length_mm"] = (float(example["bill_length_mm"])
                                          + self.deviations["bill_length_mm"])
        self.assertAlmostEqual(
            float(cf.distances(example, moved, self.deviations)[0]), 1.0)

    def test_one_changed_label_costs_one(self):
        example = self.example()
        flipped = pd.DataFrame([example], columns=penguins.FEATURES)
        others = [value for value in penguins.categories(self.frame)["island"]
                  if value != example["island"]]
        flipped.loc[:, "island"] = others[0]
        self.assertAlmostEqual(
            float(cf.distances(example, flipped, self.deviations)[0]), 1.0)

    def test_counterfactuals_are_predicted_as_the_target(self):
        example = self.example()
        rows, _ = cf.generate(self.model.pipeline, self.frame, example,
                              "Chinstrap", k=3)
        self.assertTrue(rows)
        for row in rows:
            frame = pd.DataFrame([row["values"]], columns=penguins.FEATURES)
            self.assertEqual(self.model.pipeline.predict(frame)[0], "Chinstrap")

    def test_counterfactuals_are_ranked_by_distance(self):
        rows, _ = cf.generate(self.model.pipeline, self.frame, self.example(),
                              "Chinstrap", k=5)
        self.assertEqual([row["distance"] for row in rows],
                         sorted(row["distance"] for row in rows))

    def test_k_is_respected(self):
        rows, _ = cf.generate(self.model.pipeline, self.frame, self.example(),
                              "Chinstrap", k=2)
        self.assertLessEqual(len(rows), 2)

    def test_asking_for_the_current_prediction_is_refused(self):
        example = self.example()
        current = self.model.pipeline.predict(
            pd.DataFrame([example], columns=penguins.FEATURES))[0]
        with self.assertRaises(cf.NoCounterfactual) as caught:
            cf.generate(self.model.pipeline, self.frame, example, current)
        self.assertIn("already predicted", str(caught.exception))

    def test_an_unknown_target_is_refused(self):
        with self.assertRaises(cf.NoCounterfactual):
            cf.generate(self.model.pipeline, self.frame, self.example(), "Penguin")

    def test_measurements_stay_inside_the_observed_range(self):
        """A counterfactual with a negative body mass would not be an explanation."""
        rows, _ = cf.generate(self.model.pipeline, self.frame, self.example(),
                              "Gentoo", k=5)
        for row in rows:
            for name in penguins.NUMERIC_FEATURES:
                self.assertGreaterEqual(float(row["values"][name]),
                                        float(self.frame[name].min()))
                self.assertLessEqual(float(row["values"][name]),
                                     float(self.frame[name].max()))

    def test_categorical_values_stay_in_the_observed_set(self):
        rows, _ = cf.generate(self.model.pipeline, self.frame, self.example(),
                              "Chinstrap", k=5)
        options = penguins.categories(self.frame)
        for row in rows:
            for name in penguins.CATEGORICAL_FEATURES:
                self.assertIn(row["values"][name], options[name])

    def test_the_changes_listed_are_the_ones_that_actually_differ(self):
        example = self.example()
        rows, _ = cf.generate(self.model.pipeline, self.frame, example,
                              "Chinstrap", k=1)
        for change in rows[0]["changes"]:
            name = change["feature"]
            self.assertNotEqual(str(rows[0]["values"][name]), str(example[name]))

    def test_it_is_reproducible(self):
        first, _ = cf.generate(self.model.pipeline, self.frame, self.example(),
                               "Chinstrap", k=3, seed=7)
        second, _ = cf.generate(self.model.pipeline, self.frame, self.example(),
                                "Chinstrap", k=3, seed=7)
        self.assertEqual([row["distance"] for row in first],
                         [row["distance"] for row in second])

    def test_counterfactuals_work_for_the_logistic_model_too(self):
        data = learning.fit_family(learning.LOGREG)
        model = learning.select(data["family"], learning.DEFAULT_LAMBDA)
        rows, _ = cf.generate(model.pipeline, self.frame, self.example(),
                              "Chinstrap", k=3)
        self.assertTrue(rows)


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class InterfaceTests(TestCase):

    def get(self, **params):
        return self.client.get(reverse("project2:index"), params)

    def test_the_page_renders(self):
        page = self.get()
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Explainability")
        self.assertContains(page, "Counterfactuals")
        self.assertContains(page, "Feature effect plots")

    def test_task_1_shows_the_tree_its_accuracy_and_its_leaves(self):
        page = self.get(model="tree")
        self.assertContains(page, "tree_")           # the drawing
        self.assertContains(page, "Test accuracy")
        self.assertContains(page, "leaves")

    def test_the_model_class_can_be_switched(self):
        page = self.get(model="logreg")
        self.assertContains(page, 'value="logreg" selected')
        self.assertContains(page, "coefficients_")
        self.assertContains(page, "non-zero coefficients")

    def test_lambda_changes_the_selected_model(self):
        loose = self.get(model="tree", lam="0.0").content.decode()
        tight = self.get(model="tree", lam="0.05").content.decode()
        self.assertNotEqual(loose, tight)

    def test_a_large_lambda_selects_a_simpler_model(self):
        family = learning.fit_family(learning.TREE)["family"]
        tight = learning.select(family, learning.LAMBDA_MAX).omega
        loose = learning.select(family, 0.0).omega
        self.assertLess(tight, loose)
        page = self.get(model="tree", lam="0.05")
        self.assertContains(page, f'<p class="stat-value">{tight}</p>')

    def test_out_of_range_lambda_is_clamped(self):
        self.assertEqual(self.get(lam="999").status_code, 200)
        self.assertEqual(self.get(lam="-5").status_code, 200)
        self.assertEqual(self.get(lam="not-a-number").status_code, 200)

    def test_every_numerical_feature_can_be_plotted(self):
        for feature in penguins.NUMERIC_FEATURES:
            page = self.get(feature=feature)
            self.assertEqual(page.status_code, 200, feature)
            self.assertContains(page, f"pdp_")
            self.assertContains(page, f"ale_")

    def test_a_bad_feature_falls_back_rather_than_failing(self):
        page = self.get(feature="island")
        self.assertEqual(page.status_code, 200)

    def test_the_ale_method_is_named_and_depends_on_the_model(self):
        tree = self.get(model="tree").content.decode()
        logreg = self.get(model="logreg").content.decode()
        self.assertIn("finite difference", tree)
        self.assertIn("closed-form derivative", logreg)

    def test_counterfactuals_appear_for_a_chosen_penguin(self):
        page = self.get(model="tree", cf_index="0", cf_target="Chinstrap", cf_k="3")
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "What would have to change")

    def test_a_bad_counterfactual_index_falls_back(self):
        self.assertEqual(self.get(cf_index="99999").status_code, 200)
        self.assertEqual(self.get(cf_index="abc").status_code, 200)

    def test_a_bad_counterfactual_target_falls_back(self):
        self.assertEqual(self.get(cf_target="Dragon").status_code, 200)

    def test_the_regions_all_read_the_same_model(self):
        """Task 4 and 5 must be linked to the model chosen in Tasks 1 to 3."""
        page = self.get(model="logreg", lam="0.0").content.decode()
        model = learning.select(learning.fit_family(learning.LOGREG)["family"], 0.0)
        # The figures are named after the selected model, so one stamp everywhere.
        import re
        stamps = set(re.findall(r"(?:pdp|ale|tradeoff|coefficients)_([0-9a-f]{10})",
                                page))
        self.assertEqual(len(stamps), 1, stamps)

    def test_the_slider_offers_jumps_to_each_switch_point(self):
        page = self.get(model="tree").content.decode()
        self.assertIn('id="lam-ticks"', page)
        self.assertIn("jump straight to one", page)

        family = learning.fit_family(learning.TREE)["family"]
        for point in learning.switch_points(family):
            self.assertIn(f"lam={point['lam']:.4f}".replace("=", "%3D")
                          if False else f"{point['lam']:.4f}", page)

    def test_a_jump_link_selects_that_lambda(self):
        family = learning.fit_family(learning.TREE)["family"]
        points = learning.switch_points(family)
        self.assertGreater(len(points), 1)
        target = points[-1]
        page = self.get(model="tree", lam=f"{target['lam']:.4f}")
        self.assertContains(page, f'<p class="stat-value">{target["omega"]}</p>')

    def test_no_template_syntax_leaks(self):
        page = self.get().content.decode()
        for leaked in ("{#", "#}", "{%", "%}", "{{", "}}"):
            self.assertNotIn(leaked, page)

    def test_select_elements_contain_only_options(self):
        page = self.get().content.decode()
        for block in page.split("<select")[1:]:
            inner = block.split("</select>")[0]
            for forbidden in ("<p", "<div", "<span", "<form"):
                self.assertNotIn(forbidden, inner)

    def test_the_report_downloads_as_a_pdf(self):
        response = self.client.get(reverse("project2:report"),
                                   {"model": "tree", "lam": "0.002"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertTrue(response.content.startswith(b"%PDF"))
        self.assertGreater(len(response.content), 20000)

    def test_the_report_describes_the_selected_model(self):
        for model_class in ("tree", "logreg"):
            response = self.client.get(reverse("project2:report"),
                                       {"model": model_class, "lam": "0.002"})
            self.assertEqual(response.status_code, 200, model_class)
            self.assertTrue(response.content.startswith(b"%PDF"))

    def test_the_page_links_to_the_report(self):
        self.assertContains(self.get(), reverse("project2:report"))

    def test_project2_is_reachable_from_the_home_page(self):
        page = self.client.get(reverse("home:index"))
        self.assertContains(page, reverse("project2:index"))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)
