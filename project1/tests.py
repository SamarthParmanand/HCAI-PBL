"""Tests for the Project 1 supervised learning interface (Task 3).

Run with:  python manage.py test project1

MEDIA_ROOT is redirected to a temporary directory throughout, so running the
tests never writes into the real media/ directory.
"""

import io
import os
import re
import shutil
import tempfile

from django.test import TestCase, override_settings
from django.urls import reverse

from .dataset import (CLASSIFICATION, REGRESSION, Dataset, _format_cell,
                      detect_problem_type, find_id_columns, load_dataset,
                      read_csv)
from . import plots, training

IRIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iris.csv")

TEMP_MEDIA = tempfile.mkdtemp(prefix="project1-tests-")


def csv_file(text, name="data.csv"):
    """An uploadable file object holding `text`."""
    handle = io.BytesIO(text.encode("utf-8"))
    handle.name = name
    return handle


def spread(i, step=7, modulus=53):
    """A feature value that is never an unbroken run of integers.

    Fixtures using a plain `i` produce a row counter, which the app correctly
    filters out as an id column - leaving the test measuring something else.
    """
    return (i * step) % modulus


# A continuous target, plus an id column and a sorted integer feature that must
# NOT be mistaken for one.
HOUSING = "id,area,rooms,price\n" + "\n".join(
    f"{i},{50 + i * 3},{1 + i % 4},{100000 + i * 1234.5}" for i in range(1, 41))

# Seven classes - more than the plot can colour individually. Neither feature may
# be an unbroken integer run, or it would be filtered out as a row id.
MANY_CLASSES = "a,b,label\n" + "\n".join(
    f"{(i * 7) % 53},{i * 2},class{i % 7}" for i in range(60))


class ProblemTypeDetectionTests(TestCase):

    def test_iris_is_classification_with_string_labels(self):
        dataset = load_dataset(IRIS)
        self.assertEqual(dataset.problem_type, CLASSIFICATION)
        self.assertEqual(dataset.target_name, "variety")
        self.assertEqual(dataset.class_names, ["Setosa", "Versicolor", "Virginica"])
        self.assertEqual(dataset.n_rows, 150)

    def test_continuous_target_is_regression(self):
        dataset = load_dataset(io.StringIO(HOUSING))
        self.assertEqual(dataset.problem_type, REGRESSION)

    def test_few_whole_number_levels_are_class_labels(self):
        # The spec's own Iris example uses numeric species labels.
        numeric = "a,b,species\n" + "\n".join(f"{i},{i * 2},{i % 3}" for i in range(30))
        self.assertEqual(load_dataset(io.StringIO(numeric)).problem_type, CLASSIFICATION)

    def test_two_level_numeric_target_is_classification(self):
        binary = "a,b,y\n" + "\n".join(f"{i},{i * 2},{i % 2}" for i in range(30))
        self.assertEqual(load_dataset(io.StringIO(binary)).problem_type, CLASSIFICATION)

    def test_boolean_target_is_classification(self):
        frame = read_csv(io.StringIO("a,flag\n1,True\n2,False\n3,True\n"))
        self.assertEqual(detect_problem_type(frame["flag"]), CLASSIFICATION)


class IdColumnTests(TestCase):

    def test_named_id_column_is_dropped(self):
        dataset = load_dataset(io.StringIO(HOUSING))
        self.assertEqual(dataset.dropped_columns, ["id"])
        self.assertNotIn("id", dataset.feature_names)

    def test_sorted_feature_is_not_mistaken_for_an_id(self):
        """`area` climbs in steps of 3 - unique and sorted, but not a row counter."""
        dataset = load_dataset(io.StringIO(HOUSING))
        self.assertIn("area", dataset.feature_names)

    def test_unbroken_integer_run_is_dropped(self):
        counter = "n,value,y\n" + "\n".join(f"{i},{i * 7 % 13},{i % 2}" for i in range(20))
        self.assertEqual(find_id_columns(read_csv(io.StringIO(counter))), ["n"])

    def test_target_column_is_never_treated_as_an_id(self):
        # Both columns count upwards, but the last one is the target and must
        # survive; only the feature column may be flagged.
        frame = read_csv(io.StringIO("a,b\n5,0\n6,1\n7,2\n"))
        flagged = find_id_columns(frame)
        self.assertNotIn("b", flagged)
        self.assertEqual(flagged, ["a"])

    def test_id_columns_can_be_kept_on_request(self):
        dataset = load_dataset(io.StringIO(HOUSING), keep_id_columns=True)
        self.assertEqual(dataset.dropped_columns, [])
        self.assertIn("id", dataset.feature_names)


class OverrideTests(TestCase):

    def test_user_override_wins_over_detection(self):
        numeric = "a,b,y\n" + "\n".join(f"{i},{i * 2},{i % 3}" for i in range(30))
        dataset = load_dataset(io.StringIO(numeric), problem_type=REGRESSION)
        self.assertEqual(dataset.problem_type, REGRESSION)
        self.assertTrue(dataset.was_overridden)

    def test_regression_on_a_string_target_is_refused(self):
        dataset = load_dataset(IRIS, problem_type=REGRESSION)
        self.assertEqual(dataset.problem_type, CLASSIFICATION)
        self.assertEqual(dataset.rejected_problem_type, REGRESSION)

    def test_unknown_problem_type_falls_back_to_detection(self):
        dataset = load_dataset(IRIS, problem_type="nonsense")
        self.assertEqual(dataset.problem_type, CLASSIFICATION)
        self.assertIsNone(dataset.rejected_problem_type)

    def test_target_is_an_axis_option_only_for_regression(self):
        regression = load_dataset(io.StringIO(HOUSING))
        self.assertIn("price", regression.axis_choices())

        classification = load_dataset(IRIS)
        self.assertNotIn("variety", classification.axis_choices())


class MalformedInputTests(TestCase):

    def assert_rejected(self, text_or_file):
        source = text_or_file if hasattr(text_or_file, "read") else io.StringIO(text_or_file)
        with self.assertRaises(ValueError):
            load_dataset(source)

    def test_empty_file_is_rejected(self):
        self.assert_rejected("")

    def test_header_without_rows_is_rejected(self):
        self.assert_rejected("a,b,c\n")

    def test_single_column_is_rejected(self):
        self.assert_rejected("only\n1\n2\n")

    def test_dataset_of_only_an_id_and_a_target_is_rejected(self):
        self.assert_rejected("id,y\n1,a\n2,b\n3,c\n")


class FormattingTests(TestCase):

    def test_large_numbers_avoid_scientific_notation(self):
        self.assertEqual(_format_cell(199876.5), "199,876.5")
        self.assertEqual(_format_cell(100000.0), "100,000")

    def test_compact_form_for_summary_tiles(self):
        self.assertEqual(_format_cell(199876.5, compact=True), "199.9K")
        self.assertEqual(_format_cell(1e9, compact=True), "1B")

    def test_whole_floats_lose_their_decimal(self):
        self.assertEqual(_format_cell(3.0), "3")

    def test_non_numeric_values_pass_through(self):
        self.assertEqual(_format_cell("Setosa"), "Setosa")


class PreviewTests(TestCase):

    def test_preview_returns_plain_lists_not_dicts(self):
        """Dotted column names would break attribute lookup in a template."""
        columns, rows = load_dataset(IRIS).preview()
        self.assertIn("sepal.length", columns)
        self.assertIsInstance(rows[0], list)

    def test_summary_marks_exactly_one_target(self):
        summary = load_dataset(IRIS).summary()
        roles = [column["role"] for column in summary]
        self.assertEqual(roles.count("target"), 1)
        self.assertEqual(summary[-1]["role"], "target")


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class PlotTests(TestCase):

    def test_scatter_is_written_to_media(self):
        dataset = load_dataset(IRIS)
        result = plots.save_scatter(dataset, "sepal.length", "sepal.width", "p1/plot.png")
        self.assertTrue(os.path.exists(os.path.join(TEMP_MEDIA, "p1", "plot.png")))
        self.assertTrue(result.url.endswith("p1/plot.png"))
        self.assertEqual(result.folded, [])

    def test_extra_classes_fold_into_other(self):
        dataset = load_dataset(io.StringIO(MANY_CLASSES))
        result = plots.save_scatter(dataset, "a", "b", "p1/many.png")
        self.assertEqual(len(result.folded), 7 - plots.MAX_SERIES)

    def test_regression_plot_with_target_on_an_axis(self):
        dataset = load_dataset(io.StringIO(HOUSING))
        result = plots.save_scatter(dataset, "area", "price", "p1/reg.png")
        self.assertTrue(os.path.exists(os.path.join(TEMP_MEDIA, "p1", "reg.png")))
        self.assertEqual(result.folded, [])

    def test_palette_is_never_cycled(self):
        """More classes than colours must reuse nothing but the Other colour."""
        self.assertEqual(len(set(plots.SERIES_COLOURS)), plots.MAX_SERIES)
        self.assertNotIn(plots.OTHER_COLOUR, plots.SERIES_COLOURS)


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class InterfaceTests(TestCase):
    """The pages a user actually clicks through."""

    def upload(self, text, name="data.csv"):
        return self.client.post(reverse("project1:index"), {"file": csv_file(text, name)})

    def test_no_template_syntax_leaks_into_the_page(self):
        """A `{# #}` comment spanning two lines is not a comment - it renders."""
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        for url in (reverse("project1:index"), reverse("project1:explore")):
            page = self.client.get(url).content.decode()
            for leaked in ("{#", "#}", "{%", "%}", "{{", "}}"):
                self.assertNotIn(leaked, page, f"{leaked} leaked into {url}")

    def test_select_elements_contain_only_options(self):
        """A stray block inside a <select> renders as broken, unusable markup."""
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        page = self.client.get(reverse("project1:explore")).content.decode()
        for block in page.split("<select")[1:]:
            inner = block.split("</select>")[0]
            for forbidden in ("<p", "<div", "<span", "<form"):
                self.assertNotIn(forbidden, inner,
                                 f"{forbidden} found inside a <select>")

    def test_same_column_on_both_axes_is_explained(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        # variety only becomes selectable as an axis once it is not the target.
        self.post(target="petal.length")
        page = self.post(x="variety", y="variety")
        self.assertIn("Both axes show", page)

    def test_different_axes_show_no_such_warning(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        page = self.post(x="sepal.length", y="sepal.width")
        self.assertNotIn("Both axes show", page)

    def post(self, **data):
        data.setdefault("controls", "1")
        return self.client.post(reverse("project1:explore"), data).content.decode()

    def test_swap_axes_exchanges_them(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        page = self.post(x="sepal.length", y="petal.width")
        self.assertIn('value="sepal.length" selected', page)

        swapped = self.post(swap="1")
        block_x = swapped.split('name="x"')[1].split("</select>")[0]
        block_y = swapped.split('name="y"')[1].split("</select>")[0]
        self.assertIn('value="petal.width" selected', block_x)
        self.assertIn('value="sepal.length" selected', block_y)

    def test_upload_page_renders(self):
        response = self.client.get(reverse("project1:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Load a dataset")

    def test_explore_without_a_dataset_redirects_to_upload(self):
        response = self.client.get(reverse("project1:explore"))
        self.assertRedirects(response, reverse("project1:index"))

    def test_upload_then_explore(self):
        response = self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        self.assertRedirects(response, reverse("project1:explore"))

        page = self.client.get(reverse("project1:explore"))
        self.assertEqual(page.status_code, 200)
        for expected in ["iris.csv", "variety", "Setosa", "classification", "sepal.length"]:
            self.assertContains(page, expected)

    def test_sample_button_loads_iris(self):
        response = self.client.post(reverse("project1:index"), {"use_sample": "1"})
        self.assertRedirects(response, reverse("project1:explore"))
        self.assertContains(self.client.get(reverse("project1:explore")), "variety")

    def test_sample_button_is_not_blocked_by_the_required_file_input(self):
        """The browser refuses to submit a form whose required file input is empty.

        The test client ignores HTML5 validation, so this checks the structure the
        browser actually enforces: the sample button must not belong to the form
        that carries the required <input type="file">.
        """
        page = self.client.get(reverse("project1:index")).content.decode()

        self.assertIn('required', page)                    # the file input still is
        self.assertIn('form="sample-form"', page)          # button posts elsewhere
        self.assertIn('id="sample-form"', page)

        # The sample form must not contain a file input of its own.
        sample_form = page.split('id="sample-form"')[1].split("</form>")[0]
        self.assertNotIn("type=\"file\"", sample_form)

    def test_axis_choice_round_trips(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        page = self.client.post(reverse("project1:explore"),
                                {"x": "petal.width", "y": "sepal.width",
                                 "problem_type": "classification"})
        self.assertContains(page, 'value="petal.width" selected')

    def test_impossible_override_is_explained_not_crashed(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        page = self.client.post(reverse("project1:explore"),
                                {"x": "petal.width", "y": "sepal.width",
                                 "problem_type": "regression"})
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "is not numeric")

    def test_keep_ids_checkbox_restores_the_column(self):
        self.upload(HOUSING, "housing.csv")
        # `controls` is the hidden marker the real control panel submits.
        page = self.client.post(reverse("project1:explore"),
                                {"controls": "1", "keep_ids": "on"})
        self.assertContains(page, "Keep id-like columns")
        self.assertNotContains(page, "Filtered out as row identifier")

    def test_wide_dataset_scrolls_instead_of_overflowing(self):
        columns = [f"f{i}" for i in range(30)]
        wide = ",".join(columns) + ",label\n" + "\n".join(
            ",".join(str((i * (j + 2)) % 97) for j in range(30)) + f",class{i % 7}"
            for i in range(60))
        self.upload(wide, "wide.csv")
        page = self.client.get(reverse("project1:explore"))
        self.assertContains(page, "table-responsive")
        self.assertContains(page, "Other")

    def test_dataset_of_only_categorical_features_still_plots(self):
        """Categorical-only data used to be unplottable; it now draws a count heatmap."""
        categorical = "colour,shape,label\n" + "\n".join(
            f"{'red' if i % 2 else 'blue'},{'round' if i % 3 else 'flat'},{i % 2}"
            for i in range(20))
        self.upload(categorical, "categorical.csv")
        page = self.client.get(reverse("project1:explore"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "chart_")            # an image was produced
        self.assertContains(page, 'value="colour"')    # and offered as an axis

    def test_unreadable_upload_shows_an_error_rather_than_failing(self):
        broken = io.BytesIO(b"\x00\x01\x02 not a csv \xff\xfe")
        broken.name = "junk.bin"
        response = self.client.post(reverse("project1:index"), {"file": broken})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "alert-danger")

    def test_uploaded_filename_cannot_escape_the_media_directory(self):
        self.upload("a,b,y\n1,2,x\n3,4,y\n", "../../evil.csv")
        stored = os.listdir(os.path.join(TEMP_MEDIA, "project1"))
        self.assertTrue(any(name.startswith("evil") for name in stored), stored)
        self.assertFalse(os.path.exists(os.path.join(TEMP_MEDIA, "..", "evil.csv")))

    def test_a_second_upload_replaces_the_first(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        self.upload(HOUSING, "housing.csv")
        page = self.client.get(reverse("project1:explore"))
        self.assertContains(page, "housing.csv")
        self.assertNotContains(page, "variety")

    def test_chart_type_can_be_switched(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        for chart, _ in plots.CHART_TYPES:
            page = self.client.post(reverse("project1:explore"), {"chart": chart})
            self.assertEqual(page.status_code, 200, chart)
            self.assertContains(page, f'value="{chart}" selected')

    def test_axis_survives_a_chart_switch(self):
        """The disabled y-select submits nothing; the choice must not be lost."""
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        self.client.post(reverse("project1:explore"),
                         {"chart": "scatter", "x": "petal.length", "y": "petal.width"})
        self.client.post(reverse("project1:explore"), {"chart": "distribution"})
        page = self.client.post(reverse("project1:explore"), {"chart": "scatter"})
        self.assertContains(page, 'value="petal.width" selected')

    def test_correlation_is_hidden_when_there_is_nothing_to_correlate(self):
        self.upload("colour,label\nred,a\nblue,b\ngreen,a\n", "one.csv")
        page = self.client.get(reverse("project1:explore"))
        self.assertNotContains(page, 'value="correlation"')

    def test_explanations_are_shown(self):
        self.upload(open(IRIS, encoding="utf-8").read(), "iris.csv")
        page = self.client.get(reverse("project1:explore"))
        self.assertContains(page, "How this was read")
        self.assertContains(page, "read as class labels")

    def test_audit_reports_imbalance(self):
        skewed = "a,b,y\n" + "\n".join(f"{i},{i * 2},{'rare' if i < 3 else 'common'}"
                                       for i in range(60))
        self.upload(skewed, "skewed.csv")
        page = self.client.get(reverse("project1:explore"))
        self.assertContains(page, "Imbalanced classes")

    def test_skipped_rows_are_reported(self):
        gappy = "a,b,y\n" + "\n".join(
            (f"{spread(i)},,{i % 2}" if i % 10 == 0
             else f"{spread(i)},{spread(i, 11, 37)},{i % 2}") for i in range(40))
        self.upload(gappy, "gappy.csv")
        page = self.client.get(reverse("project1:explore"))
        self.assertContains(page, "skipped because a plotted column is empty")

    def test_download_returns_the_parsed_csv(self):
        self.upload(HOUSING, "housing.csv")
        response = self.client.get(reverse("project1:download"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("attachment", response["Content-Disposition"])

        body = response.content.decode()
        self.assertTrue(body.startswith("area,rooms,price"), body[:40])
        self.assertNotIn("id,", body.splitlines()[0])   # the id column was filtered out

    def test_download_without_a_dataset_redirects(self):
        self.assertRedirects(self.client.get(reverse("project1:download")),
                             reverse("project1:index"))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)


CONTINUOUS = "area,rooms,price\n" + "\n".join(
    f"{(i * 7) % 53},{1 + i % 4},{100000 + i * 1234.5}" for i in range(60))

# Last column is boolean, like the real airbus export: classification only.
BOOLEAN_TARGET = "hours,altitude,incident\n" + "\n".join(
    f"{(i * 7) % 53},{30000 + i * 100},{'TRUE' if i % 3 else 'FALSE'}" for i in range(60))

FEW_INTS = "a,b,stars\n" + "\n".join(
    f"{(i * 7) % 53},{(i * 11) % 37},{i % 5}" for i in range(60))


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class ProblemTypePersistenceTests(TestCase):
    """The problem type must be the user's choice, and must survive other edits."""

    def upload(self, text):
        handle = csv_file(text)
        self.client.post(reverse("project1:index"), {"file": handle})

    def shown(self):
        page = self.client.get(reverse("project1:explore")).content.decode()
        return self.type_of(page)

    @staticmethod
    def type_of(page):
        block = page.split('name="problem_type"')[1].split("</select>")[0]
        match = re.search(r'value="(\w+)"[^>]*\sselected', block)
        return match.group(1) if match else "none"

    def post(self, **data):
        data.setdefault("controls", "1")
        return self.client.post(reverse("project1:explore"), data).content.decode()

    def test_continuous_target_starts_as_regression(self):
        self.upload(CONTINUOUS)
        self.assertEqual(self.shown(), "regression")

    def test_override_to_classification_sticks(self):
        self.upload(CONTINUOUS)
        self.assertEqual(self.type_of(self.post(problem_type="classification")),
                         "classification")

    def test_override_survives_a_chart_change(self):
        """A POST without problem_type must not silently revert to detection."""
        self.upload(FEW_INTS)
        self.assertEqual(self.shown(), "classification")           # detected
        self.assertEqual(self.type_of(self.post(problem_type="regression")), "regression")
        # Changing the chart submits no problem_type when the control is absent.
        page = self.client.post(reverse("project1:explore"), {"chart": "distribution"})
        self.assertEqual(self.type_of(page.content.decode()), "regression")

    def test_override_survives_a_plain_reload(self):
        self.upload(FEW_INTS)
        self.post(problem_type="regression")
        self.assertEqual(self.shown(), "regression")

    def test_keep_ids_is_not_reset_by_an_unrelated_post(self):
        self.upload(HOUSING)
        self.post(keep_ids="on")
        page = self.client.post(reverse("project1:explore"), {"chart": "distribution"})
        self.assertNotContains(page, "Filtered out as row identifier")

    def test_regression_is_offered_for_a_numeric_target(self):
        self.upload(CONTINUOUS)
        page = self.client.get(reverse("project1:explore")).content.decode()
        block = page.split('name="problem_type"')[1].split("</select>")[0]
        self.assertNotIn("disabled", block)

    def test_regression_is_disabled_for_a_boolean_target(self):
        """The airbus case: the control must show why it cannot be changed."""
        self.upload(BOOLEAN_TARGET)
        page = self.client.get(reverse("project1:explore")).content.decode()
        block = page.split('name="problem_type"')[1].split("</select>")[0]
        self.assertIn("disabled", block)
        self.assertIn("can only be a classification problem", page)


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class TargetSelectionTests(TestCase):
    """Choosing the target is what makes the problem type a real choice."""

    def upload(self, text):
        self.client.post(reverse("project1:index"), {"file": csv_file(text)})

    def post(self, **data):
        data.setdefault("controls", "1")
        return self.client.post(reverse("project1:explore"), data).content.decode()

    def test_last_column_is_the_default_target(self):
        self.upload(BOOLEAN_TARGET)
        page = self.client.get(reverse("project1:explore")).content.decode()
        self.assertIn('<option value="incident" selected>', page)

    def test_choosing_a_numeric_target_enables_regression(self):
        """The fix for the airbus dataset: point at a quantity instead of a flag."""
        self.upload(BOOLEAN_TARGET)
        page = self.post(target="altitude")
        self.assertIn("Target: altitude", page)
        self.assertIn("regression", page)
        block = page.split('name="problem_type"')[1].split("</select>")[0]
        self.assertNotIn("disabled", block)

    def test_chosen_target_becomes_a_feature_again_when_changed_back(self):
        self.upload(BOOLEAN_TARGET)
        self.post(target="altitude")
        page = self.post(target="incident")
        self.assertIn("Target: incident", page)

    def test_changing_the_target_clears_a_stale_override(self):
        self.upload(FEW_INTS)
        self.post(problem_type="regression")
        page = self.post(target="a")
        # 'a' has many distinct values, so detection should say regression on its
        # own merits rather than because of the old override.
        self.assertIn("Detected automatically", page)

    def test_choosing_the_target_is_explained(self):
        self.upload(BOOLEAN_TARGET)
        page = self.post(target="altitude")
        self.assertIn("You chose", page)
        self.assertIn("would have been the last column", page)

    def test_target_is_never_filtered_out_as_an_id(self):
        counter = "value,rowid\n" + "\n".join(f"{(i * 7) % 53},{i}" for i in range(30))
        dataset = load_dataset(io.StringIO(counter), target_name="rowid")
        self.assertEqual(dataset.target_name, "rowid")
        self.assertEqual(dataset.dropped_columns, [])

    def test_an_unknown_target_falls_back_to_the_last_column(self):
        dataset = load_dataset(io.StringIO(CONTINUOUS), target_name="nonexistent")
        self.assertEqual(dataset.target_name, "price")
        self.assertFalse(dataset.target_was_chosen)

    def test_features_exclude_the_chosen_target(self):
        dataset = load_dataset(io.StringIO(CONTINUOUS), target_name="rooms")
        self.assertNotIn("rooms", dataset.feature_names)
        self.assertIn("price", dataset.feature_names)


class CategoricalPlottingTests(TestCase):

    def test_categorical_features_are_offered_as_axes(self):
        mixed = "size,colour,y\n" + "\n".join(
            f"{i},{'red' if i % 2 else 'blue'},{i % 2}" for i in range(20))
        dataset = load_dataset(io.StringIO(mixed))
        self.assertIn("colour", dataset.axis_choices())
        self.assertEqual(dataset.categorical_feature_names, ["colour"])

    def test_constant_columns_are_not_chosen_as_default_axes(self):
        """Opening on a column that never varies draws a flat line and teaches nothing."""
        messy = "constant,score,width,y\n" + "\n".join(
            f"7,{spread(i)},{spread(i, 11, 37)},{i % 2}" for i in range(40))
        dataset = load_dataset(io.StringIO(messy))
        self.assertEqual(dataset.default_axes(), ("score", "width"))
        self.assertIn("constant", dataset.axis_choices())   # still selectable by hand

    def test_informative_category_beats_a_constant_number(self):
        """With only one varying number, pair it with a category rather than a flat column."""
        messy = "constant,score,region,y\n" + "\n".join(
            f"7,{spread(i)},{['n', 's', 'e'][i % 3]},{i % 2}" for i in range(40))
        dataset = load_dataset(io.StringIO(messy))
        self.assertEqual(dataset.default_axes(), ("score", "region"))

    def test_single_categorical_feature_is_described_in_the_singular(self):
        mixed = "size,colour,y\n" + "\n".join(
            f"{spread(i)},{'red' if i % 2 else 'blue'},{i % 2}" for i in range(20))
        reasons = " ".join(load_dataset(io.StringIO(mixed)).explanations())
        self.assertIn("“colour” holds labels", reasons)

    def test_numeric_axes_are_preferred_by_default(self):
        mixed = "colour,size,weight,y\n" + "\n".join(
            f"{'red' if i % 2 else 'blue'},{spread(i)},{spread(i, 11, 37)},{i % 2}"
            for i in range(20))
        dataset = load_dataset(io.StringIO(mixed))
        self.assertEqual(dataset.default_axes(), ("size", "weight"))


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class ChartRenderingTests(TestCase):
    """Every chart must render for every dataset shape it is offered for."""

    MIXED = "colour,size,y\n" + "\n".join(
        f"{'red' if i % 2 else 'blue'},{i % 23},{'a' if i % 3 else 'b'}" for i in range(40))

    def render(self, dataset, chart, x, y, name):
        result = plots.save_chart(dataset, chart, x, y, f"charts/{name}.png")
        self.assertTrue(os.path.exists(os.path.join(TEMP_MEDIA, "charts", f"{name}.png")))
        return result

    def test_distribution_of_a_numeric_feature(self):
        dataset = load_dataset(IRIS)
        self.render(dataset, plots.DISTRIBUTION, "sepal.length", "sepal.width", "dist_num")

    def test_distribution_of_a_categorical_feature(self):
        dataset = load_dataset(io.StringIO(self.MIXED))
        self.render(dataset, plots.DISTRIBUTION, "colour", "size", "dist_cat")

    def test_correlation_matrix(self):
        self.render(load_dataset(IRIS), plots.CORRELATION, "sepal.length", "sepal.width", "corr")

    def test_scatter_with_one_categorical_axis(self):
        dataset = load_dataset(io.StringIO(self.MIXED))
        self.render(dataset, plots.SCATTER, "colour", "size", "strip")

    def test_scatter_with_two_categorical_axes(self):
        both = "colour,shape,y\n" + "\n".join(
            f"{'red' if i % 2 else 'blue'},{'round' if i % 3 else 'flat'},{i % 2}"
            for i in range(30))
        dataset = load_dataset(io.StringIO(both))
        self.render(dataset, plots.SCATTER, "colour", "shape", "heatmap")

    def test_distribution_with_missing_values(self):
        gappy = "a,b,y\n" + "\n".join(
            (f",{i},{i % 2}" if i % 7 == 0 else f"{i},{i},{i % 2}") for i in range(40))
        dataset = load_dataset(io.StringIO(gappy))
        self.render(dataset, plots.DISTRIBUTION, "a", "b", "dist_gappy")


class AuditTests(TestCase):

    def titles(self, dataset):
        return [finding["title"] for finding in dataset.audit()]

    def test_clean_data_produces_no_findings(self):
        clean = "width,height,y\n" + "\n".join(
            f"{spread(i)},{spread(i, 11, 37)},{'a' if i % 2 else 'b'}" for i in range(60))
        self.assertEqual(load_dataset(io.StringIO(clean)).audit(), [])

    def test_iris_duplicate_row_is_reported(self):
        """iris.csv genuinely contains one repeated row - the audit must say so."""
        findings = load_dataset(IRIS).audit()
        self.assertEqual([finding["title"] for finding in findings], ["Duplicate rows"])

    def test_a_unique_column_is_not_reported_as_leakage(self):
        """Every value appearing once implies one class trivially, not by leaking."""
        unique = "code,size,y\n" + "\n".join(
            f"u{i},{spread(i)},{'a' if i % 2 else 'b'}" for i in range(40))
        self.assertNotIn("Possible target leakage",
                         self.titles(load_dataset(io.StringIO(unique))))

    def test_constant_feature_is_flagged(self):
        constant = "a,b,y\n" + "\n".join(f"{i},7,{i % 2}" for i in range(20))
        self.assertIn("Constant feature(s)", self.titles(load_dataset(io.StringIO(constant))))

    def test_duplicate_rows_are_flagged(self):
        duplicated = "a,b,y\n" + "\n".join("1,2,x" for _ in range(10))
        self.assertIn("Duplicate rows", self.titles(load_dataset(io.StringIO(duplicated))))

    def test_imbalance_is_flagged(self):
        skewed = "a,b,y\n" + "\n".join(
            f"{i},{i * 2},{'rare' if i < 3 else 'common'}" for i in range(60))
        self.assertIn("Imbalanced classes", self.titles(load_dataset(io.StringIO(skewed))))

    def test_missing_heavy_column_is_flagged(self):
        gappy = "a,b,y\n" + "\n".join(
            (f"{i},,{i % 2}" if i % 2 else f"{i},{i},{i % 2}") for i in range(40))
        self.assertIn("Column(s) missing a lot of data",
                      self.titles(load_dataset(io.StringIO(gappy))))

    def test_numeric_leakage_is_flagged(self):
        leaky = "a,copy,price\n" + "\n".join(
            f"{i},{i * 2 + 1},{i * 2.0 + 1}" for i in range(40))
        self.assertIn("Possible target leakage", self.titles(load_dataset(io.StringIO(leaky))))

    def test_categorical_leakage_is_flagged(self):
        leaky = "size,code,y\n" + "\n".join(
            f"{i},{'c' if i % 2 else 'd'},{'a' if i % 2 else 'b'}" for i in range(30))
        self.assertIn("Possible target leakage", self.titles(load_dataset(io.StringIO(leaky))))

    def test_missing_target_values_are_flagged(self):
        gappy = "a,b,y\n" + "\n".join(
            (f"{spread(i)},{spread(i, 11, 37)}," if i % 9 == 0
             else f"{spread(i)},{spread(i, 11, 37)},{i % 2}") for i in range(40))
        self.assertIn("Rows with no target value",
                      self.titles(load_dataset(io.StringIO(gappy))))


class RowsPlottedTests(TestCase):

    def test_rows_with_gaps_are_counted_out(self):
        gappy = "a,b,y\n" + "\n".join(
            (f"{spread(i)},,{i % 2}" if i % 10 == 0
             else f"{spread(i)},{spread(i, 11, 37)},{i % 2}") for i in range(40))
        dataset = load_dataset(io.StringIO(gappy))
        self.assertIn("a", dataset.feature_names)      # not filtered out as an id
        kept, skipped = dataset.rows_plotted(["a", "b"])
        self.assertEqual(skipped, 4)
        self.assertEqual(kept, 36)

    def test_complete_data_skips_nothing(self):
        dataset = load_dataset(IRIS)
        kept, skipped = dataset.rows_plotted(["sepal.length", "sepal.width"])
        self.assertEqual((kept, skipped), (150, 0))


class ExplanationTests(TestCase):

    def test_string_target_is_explained(self):
        reasons = " ".join(load_dataset(IRIS).explanations())
        self.assertIn("variety", reasons)
        self.assertIn("class labels", reasons)

    def test_continuous_target_is_explained(self):
        reasons = " ".join(load_dataset(io.StringIO(HOUSING)).explanations())
        self.assertIn("quantity to predict", reasons)

    def test_dropped_id_column_is_explained(self):
        reasons = " ".join(load_dataset(io.StringIO(HOUSING)).explanations())
        self.assertIn("row number", reasons)

    def test_override_is_explained(self):
        numeric = "a,b,y\n" + "\n".join(f"{i},{i * 2},{i % 3}" for i in range(30))
        dataset = load_dataset(io.StringIO(numeric), problem_type=REGRESSION)
        self.assertIn("You overrode that", " ".join(dataset.explanations()))


# A target that genuinely depends on the features, so a model can actually learn it.
LEARNABLE = "area,rooms,price\n" + "\n".join(
    f"{50 + (i * 7) % 53},{1 + i % 4},{3000 * (50 + (i * 7) % 53) + 12000 * (1 + i % 4)}"
    for i in range(120))


class TrainingPipelineTests(TestCase):
    """Task 4: split, sweep, score."""

    def test_every_classification_model_trains_on_iris(self):
        dataset = load_dataset(IRIS)
        for key, label in training.model_choices("classification"):
            result = training.train(dataset, key)
            self.assertGreater(result["best"]["test_score"], 0.8, label)
            self.assertEqual(result["score_label"], "Accuracy")

    def test_every_regression_model_trains(self):
        dataset = load_dataset(io.StringIO(LEARNABLE))
        self.assertEqual(dataset.problem_type, REGRESSION)
        for key, label in training.model_choices("regression"):
            result = training.train(dataset, key)
            self.assertEqual(result["score_label"], "R\u00b2")
            self.assertIn("r2", result["detail"])

    def test_a_learnable_target_is_actually_learned(self):
        """price is a linear function of the features, so ridge should nail it."""
        dataset = load_dataset(io.StringIO(LEARNABLE))
        result = training.train(dataset, "ridge")
        self.assertGreater(result["best"]["test_score"], 0.95)

    def test_the_sweep_covers_every_hyperparameter_value(self):
        dataset = load_dataset(IRIS)
        result = training.train(dataset, "tree")
        self.assertEqual(len(result["results"]), len(training.DEPTHS))
        self.assertIn(result["best"]["value"], training.DEPTHS)

    def test_best_is_the_highest_test_score(self):
        dataset = load_dataset(IRIS)
        result = training.train(dataset, "tree")
        scored = [row["test_score"] for row in result["results"] if row["error"] is None]
        self.assertEqual(result["best"]["test_score"], max(scored))

    def test_test_size_controls_the_split(self):
        dataset = load_dataset(IRIS)
        small = training.train(dataset, "tree", test_size=0.1)
        large = training.train(dataset, "tree", test_size=0.5)
        self.assertLess(small["n_test"], large["n_test"])
        self.assertEqual(small["n_train"] + small["n_test"], 150)

    def test_the_split_is_stratified_when_it_can_be(self):
        self.assertTrue(training.train(load_dataset(IRIS), "tree")["stratified"])

    def test_confusion_matrix_totals_the_test_rows(self):
        dataset = load_dataset(IRIS)
        result = training.train(dataset, "tree")
        total = sum(sum(row) for row in result["detail"]["matrix"])
        self.assertEqual(total, result["n_test"])

    def test_categorical_features_and_missing_values_are_handled(self):
        mixed = "colour,size,grade\n" + "\n".join(
            (f"{'red' if i % 2 else 'blue'},{'' if i % 9 == 0 else spread(i)},"
             f"{'a' if i % 2 else 'b'}")
            for i in range(80))
        dataset = load_dataset(io.StringIO(mixed))
        result = training.train(dataset, "tree")
        self.assertEqual(result["n_categorical"], 1)
        self.assertGreater(result["best"]["test_score"], 0.5)

    def test_feature_importance_is_read_off_the_fitted_model(self):
        dataset = load_dataset(IRIS)
        importance = training.train(dataset, "tree")["importance"]
        self.assertIsNotNone(importance)
        self.assertIn("impurity", importance["kind"])
        names = [row["name"] for row in importance["rows"]]
        # Petal measurements are what actually separates the iris species.
        self.assertIn("petal.width", names[:2])
        for row in importance["rows"]:
            self.assertIn(row["name"], dataset.feature_names)

    def test_importance_shares_are_a_fraction_of_the_whole(self):
        importance = training.train(load_dataset(IRIS), "forest")["importance"]
        total = sum(row["share"] for row in importance["rows"])
        self.assertLessEqual(total, 1.0 + 1e-9)
        self.assertGreater(total, 0.5)

    def test_a_linear_model_reports_its_coefficients(self):
        importance = training.train(load_dataset(IRIS), "logistic")["importance"]
        self.assertIn("coefficient", importance["kind"])

    def test_knn_reports_no_importance_rather_than_inventing_one(self):
        self.assertIsNone(training.train(load_dataset(IRIS), "knn")["importance"])

    def test_a_single_class_target_is_refused(self):
        same = "a,b,y\n" + "\n".join(f"{spread(i)},{i % 7},only" for i in range(30))
        with self.assertRaises(training.TrainingError):
            training.train(load_dataset(io.StringIO(same)), "tree")

    def test_too_few_rows_are_refused(self):
        tiny = "a,b,y\n1,2,x\n3,4,y\n5,6,x\n"
        with self.assertRaises(training.TrainingError):
            training.train(load_dataset(io.StringIO(tiny)), "tree")

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(training.TrainingError):
            training.train(load_dataset(IRIS), "does-not-exist")

    def test_model_choices_match_the_problem_type(self):
        classification = dict(training.model_choices("classification"))
        regression = dict(training.model_choices("regression"))
        self.assertIn("logistic", classification)
        self.assertNotIn("logistic", regression)
        self.assertIn("ridge", regression)
        self.assertNotIn("ridge", classification)


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class TrainingInterfaceTests(TestCase):

    def upload(self, text, name="data.csv"):
        return self.client.post(reverse("project1:index"), {"file": csv_file(text, name)})

    def test_training_without_a_dataset_redirects(self):
        self.assertRedirects(self.client.get(reverse("project1:train")),
                             reverse("project1:index"))

    def test_training_page_renders_for_iris(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.get(reverse("project1:train"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Decision tree")
        self.assertContains(page, "Test accuracy")
        self.assertContains(page, "sweep_")

    def test_choosing_a_model_sticks(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.post(reverse("project1:train"), {"model": "knn"})
        self.assertContains(page, 'value="knn" selected')
        self.assertContains(page, "Neighbours (k)")

    def slider_value(self, page):
        """The value the test-size range input is actually rendered with."""
        tag = page.content.decode().split('name="test_size"')[1].split(">")[0]
        return re.search(r'value="(\d+)"', tag).group(1)

    def test_test_size_is_applied_and_clamped(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.post(reverse("project1:train"),
                                {"model": "tree", "test_size": "40"})
        self.assertEqual(self.slider_value(page), "40")
        # 40% of 150 rows, held back.
        self.assertContains(page, "held back for testing (40%)")

        # Out of range values must not produce an unusable split.
        page = self.client.post(reverse("project1:train"),
                                {"model": "tree", "test_size": "999"})
        self.assertEqual(self.slider_value(page), "50")

    def test_a_confusion_matrix_is_drawn_for_classification(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.get(reverse("project1:train"))
        self.assertContains(page, "confusion_")
        self.assertNotContains(page, "predictions_")

    def test_predictions_are_drawn_for_regression(self):
        self.upload(LEARNABLE, "prices.csv")
        page = self.client.get(reverse("project1:train"))
        self.assertContains(page, "predictions_")
        self.assertNotContains(page, "confusion_")
        self.assertContains(page, "RMSE")

    def test_the_importance_panel_is_shown(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.post(reverse("project1:train"), {"model": "tree"})
        self.assertContains(page, "What the model leaned on")
        self.assertContains(page, "petal.width")

    def test_no_importance_panel_for_knn(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.post(reverse("project1:train"), {"model": "knn"})
        self.assertNotContains(page, "What the model leaned on")

    def test_untrainable_data_explains_itself(self):
        same = "a,b,y\n" + "\n".join(f"{spread(i)},{i % 7},only" for i in range(30))
        self.upload(same, "same.csv")
        page = self.client.get(reverse("project1:train"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "nothing to tell apart")

    def test_setup_is_explained(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.get(reverse("project1:train")).content.decode()
        self.assertIn("held back for testing", page)
        self.assertIn("nothing about the test set leaks in", page)

    def test_no_template_syntax_leaks(self):
        self.client.post(reverse("project1:index"), {"use_sample": "1"})
        page = self.client.get(reverse("project1:train")).content.decode()
        for leaked in ("{#", "#}", "{%", "%}", "{{", "}}"):
            self.assertNotIn(leaked, page)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)
