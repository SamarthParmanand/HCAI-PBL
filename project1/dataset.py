"""Reading an uploaded CSV and working out what kind of problem it describes.

Deliberately free of Django imports: the logic here is plain pandas, so it can be
exercised on its own without a request.

Convention from the project description: the first row holds the feature names and
the **last column is the target**. Datasets may carry a row-id column, which we
filter out.
"""

import pandas as pd
from pandas.api import types as ptypes

CLASSIFICATION = "classification"
REGRESSION = "regression"
PROBLEM_TYPES = (CLASSIFICATION, REGRESSION)

# A numeric target with at most this many distinct whole-number values is read as
# a set of class labels rather than a continuous output.
MAX_DISCRETE_LEVELS = 20

# Column names that mean "row number" rather than "feature".
ID_COLUMN_NAMES = {"id", "index", "idx", "no", "nr", "num", "row", "unnamed: 0"}

# Rows shown in the preview table.
PREVIEW_ROWS = 8


def read_csv(file_or_path):
    """Read a CSV into a frame, tidying up whitespace in the header."""
    frame = pd.read_csv(file_or_path)
    frame.columns = [str(name).strip() for name in frame.columns]
    return frame


# note to self: this is the id filter the sheet mentions. the first version
# dropped any column whose values were unique and sorted, and that threw away a
# real feature once, an area column that happened to be ascending. the fix is to
# require an unbroken run, max - min == len - 1, so only an actual row counter
# matches. the name list catches the rest.
def find_id_columns(frame, protect=None):
    """Columns that look like a row identifier rather than a real feature.

    Two signals: a telling name, or an unbroken run of integers - unique values
    spanning exactly as many steps as there are rows, i.e. a permutation of
    `min..min+n-1`. Requiring the run to be unbroken matters: merely being sorted
    and unique is also true of plenty of genuine features, and dropping one of
    those would silently discard real data.

    The target is never considered, so it can never be filtered away.
    """
    id_columns = []
    protect = protect if protect is not None else frame.columns[-1]

    for name in frame.columns:
        if name == protect:
            continue
        if name.lower() in ID_COLUMN_NAMES:
            id_columns.append(name)
            continue

        column = frame[name]
        if len(frame) < 2 or not ptypes.is_integer_dtype(column) or not column.is_unique:
            continue
        if int(column.max()) - int(column.min()) == len(frame) - 1:
            id_columns.append(name)

    return id_columns


# note to self: numeric with more than max_discrete_levels distinct values is
# read as regression, anything else as classification. twenty is a judgement
# call and i should say so rather than pretend it is derived, it is what
# separates a rating out of ten from a price.
def detect_problem_type(target):
    """Guess whether `target` is a set of class labels or a continuous output."""
    # Booleans are numeric as far as pandas is concerned, so test them first.
    if ptypes.is_bool_dtype(target) or not ptypes.is_numeric_dtype(target):
        return CLASSIFICATION

    values = target.dropna()
    levels = values.nunique()

    if levels <= 2:
        return CLASSIFICATION

    whole_numbers = ptypes.is_integer_dtype(values) or bool((values % 1 == 0).all())
    if whole_numbers and levels <= MAX_DISCRETE_LEVELS:
        return CLASSIFICATION

    return REGRESSION


class Dataset:
    """An uploaded table split into features and a target.

    `problem_type` is detected from the target, but the caller may override it -
    the user gets the final say in the interface.
    """

    def __init__(self, frame, dropped_columns=(), problem_type=None, target_name=None):
        self.frame = frame
        self.dropped_columns = list(dropped_columns)

        # The last column is the target by the project's convention, but the user
        # may point at another one - which is what makes the problem type a real
        # choice rather than a property of the file.
        self.default_target_name = frame.columns[-1]
        self.target_name = (target_name if target_name in list(frame.columns)
                            else self.default_target_name)
        self.feature_names = [name for name in frame.columns if name != self.target_name]

        self.detected_problem_type = detect_problem_type(self.target)

        # Honour the user's override, except where it cannot mean anything: you
        # cannot regress onto a column of strings. Say so instead of failing.
        self.rejected_problem_type = None
        requested = problem_type if problem_type in PROBLEM_TYPES else None
        if requested == REGRESSION and not self.target_is_numeric:
            self.rejected_problem_type = requested
            requested = None

        self.problem_type = requested or self.detected_problem_type

    @property
    def target(self):
        return self.frame[self.target_name]

    @property
    def target_is_numeric(self):
        return ptypes.is_numeric_dtype(self.target) and not ptypes.is_bool_dtype(self.target)

    @property
    def is_classification(self):
        return self.problem_type == CLASSIFICATION

    @property
    def too_many_classes(self):
        """A near-continuous column read as labels is usually a regression instead."""
        return self.is_classification and self.target.nunique(dropna=True) > MAX_DISCRETE_LEVELS

    @property
    def was_overridden(self):
        return self.problem_type != self.detected_problem_type

    @property
    def target_was_chosen(self):
        """True when the user picked a target other than the last column."""
        return self.target_name != self.default_target_name

    @property
    def can_regress(self):
        """Whether "regression" is even meaningful for this target."""
        return self.target_is_numeric

    @property
    def n_rows(self):
        return len(self.frame)

    @property
    def n_missing(self):
        return int(self.frame.isna().sum().sum())

    @property
    def target_range(self):
        """Compact "min – max" for a continuous target."""
        if not self.target_is_numeric:
            return "–"
        low = _format_cell(self.target.min(), compact=True)
        high = _format_cell(self.target.max(), compact=True)
        return f"{low} – {high}"

    def class_counts(self):
        """Class labels with their row counts, largest first."""
        counts = self.target.value_counts()
        return [(str(label), int(count)) for label, count in counts.items()]

    @property
    def class_names(self):
        """Sorted distinct target values - only meaningful for classification."""
        return [str(value) for value in sorted(self.target.dropna().unique())]

    @property
    def numeric_feature_names(self):
        """Features held as numbers.

        Booleans count here - they plot perfectly well as 0/1 on an axis - even
        though `target_is_numeric` excludes them, because you cannot regress onto
        a flag. The two rules differ on purpose.
        """
        return [
            name for name in self.feature_names
            if ptypes.is_numeric_dtype(self.frame[name])
        ]

    @property
    def categorical_feature_names(self):
        """Features held as labels rather than numbers."""
        return [name for name in self.feature_names
                if name not in self.numeric_feature_names]

    def is_categorical(self, name):
        return name in self.categorical_feature_names

    def axis_choices(self):
        """Columns offered as plot axes.

        Categorical features are included - they are plotted as a strip of
        categories rather than being silently unavailable. For regression the
        target is offered too: plotting one feature against a continuous output is
        the natural view there.
        """
        choices = list(self.numeric_feature_names) + list(self.categorical_feature_names)
        if not self.is_classification and self.target_is_numeric:
            choices.append(self.target_name)
        return choices

    def default_axes(self):
        """A sensible starting pair of axes.

        Numeric features first, and columns that never vary last: opening on a
        constant column draws a single flat line and tells the user nothing.
        """
        def informative(names):
            return [name for name in names if self.frame[name].nunique(dropna=True) > 1]

        # An informative categorical column beats a numeric one that never varies.
        for candidates in (informative(self.numeric_feature_names),
                           informative(self.axis_choices()),
                           self.numeric_feature_names,
                           self.axis_choices()):
            if len(candidates) >= 2:
                return candidates[0], candidates[1]

        choices = self.axis_choices()
        if len(choices) == 1:
            return choices[0], choices[0]
        return None, None

    def rows_plotted(self, columns):
        """How many rows survive once rows missing any of `columns` are dropped.

        Returns `(kept, dropped)`. Rows with gaps cannot be placed on a chart, and
        dropping them quietly would misrepresent the data.
        """
        wanted = [name for name in dict.fromkeys(columns) if name in self.frame.columns]
        if not wanted:
            return self.n_rows, 0
        kept = int(self.frame[wanted].notna().all(axis=1).sum())
        return kept, self.n_rows - kept

    def explanations(self):
        """Plain-language reasons for each automatic decision.

        The interface can show what was decided; this says *why*, which is the
        part a person actually needs in order to disagree with it.
        """
        reasons = []
        target = self.target
        levels = int(target.nunique(dropna=True))

        if self.target_was_chosen:
            reasons.append(
                f"You chose “{self.target_name}” as the target; by default it would "
                f"have been the last column, “{self.default_target_name}”."
            )

        if not self.target_is_numeric:
            reasons.append(
                f"“{self.target_name}” holds text with {levels} distinct values, "
                f"so it is read as class labels."
            )
        elif self.detected_problem_type == CLASSIFICATION:
            reasons.append(
                f"“{self.target_name}” holds whole numbers with only {levels} distinct "
                f"values, so they are read as class labels rather than measurements."
            )
        else:
            reasons.append(
                f"“{self.target_name}” holds {levels} distinct numeric values across a "
                f"continuous range, so it is read as a quantity to predict."
            )

        if self.was_overridden:
            reasons.append(
                f"You overrode that: the data is being treated as "
                f"{self.problem_type} instead."
            )

        for name in self.dropped_columns:
            if name.lower() in ID_COLUMN_NAMES:
                reasons.append(f"“{name}” was filtered out because its name marks it as a row number.")
            else:
                reasons.append(
                    f"“{name}” was filtered out because it is an unbroken run of "
                    f"integers — a row counter, not a measurement."
                )

        categorical = self.categorical_feature_names
        if categorical:
            listed = ", ".join(f"“{name}”" for name in categorical)
            verb = "holds" if len(categorical) == 1 else "hold"
            reasons.append(f"{listed} {verb} labels, so they are drawn as categories.")

        return reasons

    def audit(self):
        """Problems worth knowing about before training a model on this data.

        Each entry is `{level, title, detail}` where level is "warn" or "info".
        """
        findings = []
        findings.extend(self._audit_columns())
        findings.extend(self._audit_rows())
        findings.extend(self._audit_target())
        findings.extend(self._audit_leakage())
        return findings

    def _audit_columns(self):
        findings = []

        constant = [name for name in self.feature_names
                    if self.frame[name].nunique(dropna=True) <= 1]
        if constant:
            findings.append({
                "level": "warn",
                "title": "Constant feature(s)",
                "detail": f"{', '.join(constant)} never change, so they cannot help a "
                          f"model separate anything. Consider removing them.",
            })

        gappy = [(name, int(self.frame[name].isna().sum())) for name in self.frame.columns]
        gappy = [(name, count) for name, count in gappy if count > 0.2 * self.n_rows]
        if gappy:
            listed = ", ".join(f"{name} ({count} missing)" for name, count in gappy)
            findings.append({
                "level": "warn",
                "title": "Column(s) missing a lot of data",
                "detail": f"{listed}. More than a fifth of the values are absent, so "
                          f"anything learned from them rests on thin evidence.",
            })

        wide = [name for name in self.categorical_feature_names
                if self.frame[name].nunique(dropna=True) > max(20, 0.5 * self.n_rows)]
        if wide:
            findings.append({
                "level": "info",
                "title": "High-cardinality categorical feature(s)",
                "detail": f"{', '.join(wide)} take nearly a different value in every row. "
                          f"Encoding them will produce very many columns.",
            })

        return findings

    def _audit_rows(self):
        duplicates = int(self.frame.duplicated().sum())
        if not duplicates:
            return []
        return [{
            "level": "info",
            "title": "Duplicate rows",
            "detail": f"{duplicates} row(s) are exact copies of an earlier row. If they "
                      f"land on both sides of a train/test split, the test score flatters "
                      f"the model.",
        }]

    def _audit_target(self):
        findings = []

        missing_target = int(self.target.isna().sum())
        if missing_target:
            findings.append({
                "level": "warn",
                "title": "Rows with no target value",
                "detail": f"{missing_target} row(s) have nothing to predict and cannot be "
                          f"used for training.",
            })

        if not self.is_classification:
            return findings

        counts = self.target.value_counts()
        if len(counts) >= 2:
            largest, smallest = int(counts.iloc[0]), int(counts.iloc[-1])
            if smallest * 5 < largest:
                findings.append({
                    "level": "warn",
                    "title": "Imbalanced classes",
                    "detail": f"“{counts.index[-1]}” has {smallest} row(s) against "
                              f"{largest} for “{counts.index[0]}”. Plain accuracy will look "
                              f"high even if the smallest class is never predicted.",
                })

        tiny = [str(label) for label, count in counts.items() if count < 5]
        if tiny:
            findings.append({
                "level": "warn",
                "title": "Very small class(es)",
                "detail": f"{', '.join(tiny)} have fewer than 5 rows, too few to both "
                          f"train on and evaluate.",
            })

        return findings

    def _audit_leakage(self):
        """Features that predict the target almost perfectly - usually a mistake."""
        findings = []
        suspects = []

        for name in self.feature_names:
            column = self.frame[name]

            if self.target_is_numeric and ptypes.is_numeric_dtype(column):
                paired = pd.concat([column, self.target], axis=1).dropna()
                if len(paired) > 2 and paired.iloc[:, 0].nunique() > 1:
                    correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1])
                    if correlation == correlation and abs(correlation) >= 0.98:
                        suspects.append(f"{name} (correlation {correlation:.2f})")

            elif self.is_classification and self.frame[name].nunique(dropna=True) > 1:
                # A feature that maps one-to-one onto the label carries the answer.
                grouped = self.frame.groupby(name, dropna=True)[self.target_name].agg(
                    ["nunique", "size"])
                # Require at least one value shared by several rows: a column with a
                # different value in every row trivially "implies" one class each and
                # would otherwise be reported as leakage every time.
                repeated = bool((grouped["size"] > 1).any())
                if len(grouped) > 1 and repeated and bool((grouped["nunique"] <= 1).all()):
                    suspects.append(f"{name} (each value implies one class)")

        if suspects:
            findings.append({
                "level": "warn",
                "title": "Possible target leakage",
                "detail": f"{', '.join(suspects)} predicts the target almost exactly. That "
                          f"usually means the column encodes the answer, and a model "
                          f"trained on it will not generalise.",
            })

        return findings

    def preview(self, n_rows=PREVIEW_ROWS):
        """Header plus the first rows, as plain lists.

        Lists rather than dicts on purpose: column names in the reference dataset
        contain dots (`sepal.length`), and Django templates read a dot as an
        attribute lookup, so `{{ row.sepal.length }}` could never work.
        """
        head = self.frame.head(n_rows)
        rows = [[_format_cell(value) for value in row] for row in head.itertuples(index=False)]
        return list(head.columns), rows

    def summary(self):
        """Per-column description used by the summary table in the interface."""
        described = []
        for name in self.frame.columns:
            column = self.frame[name]
            is_numeric = ptypes.is_numeric_dtype(column)
            described.append({
                "name": name,
                "role": "target" if name == self.target_name else "feature",
                "dtype": str(column.dtype),
                "missing": int(column.isna().sum()),
                "distinct": int(column.nunique(dropna=True)),
                "minimum": _format_cell(column.min()) if is_numeric else "–",
                "maximum": _format_cell(column.max()) if is_numeric else "–",
            })
        return described


def _format_number(value, compact=False):
    """Readable number: thousands separators, never scientific notation.

    `compact` is for the large standalone values in the summary tiles, where
    "199.9K" reads far better than "1.99e+05".
    """
    number = float(value)

    if number != number:            # NaN
        return "–"

    if compact and abs(number) >= 10_000:
        for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(number) >= divisor:
                scaled = f"{number / divisor:.1f}".rstrip("0").rstrip(".")
                return f"{scaled}{suffix}"

    rounded = round(number, 3)
    if rounded == int(rounded) and abs(rounded) < 1e15:
        return f"{int(rounded):,}"
    return f"{rounded:,}"


def _format_cell(value, compact=False):
    """Render a cell for display."""
    if value is None:
        return "–"
    if isinstance(value, bool):
        return str(value)
    try:
        return _format_number(value, compact=compact)
    except (TypeError, ValueError):
        return str(value)


def load_dataset(file_or_path, problem_type=None, keep_id_columns=False,
                 target_name=None):
    """Build a `Dataset` from an uploaded file or a path.

    Raises `ValueError` with a message fit to show the user.
    """
    try:
        frame = read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        raise ValueError("That file is empty.")
    except UnicodeDecodeError:
        raise ValueError("That file is not text — a plain UTF-8 CSV is expected.")
    except pd.errors.ParserError as error:
        raise ValueError(f"That file could not be parsed as CSV: {error}")

    if frame.empty:
        raise ValueError("That CSV has a header but no data rows.")

    target = target_name if target_name in list(frame.columns) else frame.columns[-1]

    dropped = [] if keep_id_columns else find_id_columns(frame, protect=target)
    if frame.shape[1] - len(dropped) < 2:
        raise ValueError(
            "At least two columns are needed: one feature and a target in the last column."
        )
    if dropped:
        frame = frame.drop(columns=dropped)

    return Dataset(frame, dropped_columns=dropped, problem_type=problem_type,
                   target_name=target)
