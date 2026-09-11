# HCAI-PBL - full project documentation

TUHH, *Human-Centric Artificial Intelligence* (project-based learning). Four projects, one
Django site, one Git repository.

This file is the reference for **what exists, what each part does, and why it was built that
way**. Design decisions are recorded with their reasoning, including the places where a
simpler or more flattering choice was available and deliberately not taken.

Numbers quoted here are from the committed datasets and the cached Project 3 run, measured,
not estimated. Where a number depends on a user's upload (Project 1) or on a participant's
answers (Project 4), that is said explicitly.

---

## 0. Quick start

```bash
python -m pip install -r requirements.txt
python manage.py migrate               # Project 4 stores answers in the database
python manage.py build_project3        # ~110 s; caches the Project 3 experiment
python manage.py runserver             # -> http://127.0.0.1:8000/
```

| Step | Needed for | If skipped |
|---|---|---|
| `pip install -r requirements.txt` | everything | import errors |
| `migrate` | Project 4 only | `/project4/` shows a "run migrate" page (HTTP 503), not a crash |
| `build_project3` | Project 3 only | `/project3/` shows the command to run, not a hang |
| - | Projects 1 and 2 need neither | - |

Bootstrap 5.3.3 is **vendored** under `static/vendor/`, so the site renders with no network
access. `iris.csv`, `penguins.csv` and `movie_metadata.csv` are committed. AG News (19 MB)
downloads on first use into the gitignored `data/`.

Tests: `python manage.py test` - **307 tests**.

---

## 1. Architecture and shared conventions

### 1.1 Layout

```
pbl/            settings.py, urls.py (root URLconf), wsgi/asgi
home/           the hub: group members + links to all four projects
demos/          the skeleton's reference app (file upload + matplotlib rendering)
templates/      project-level templates; base.html lives here
static/         style.css + vendored Bootstrap + favicon
media/          runtime uploads and generated PNGs (served in development)
project1/       supervised-learning interface
project2/       explainability
project3/       active learning for learning-to-defer
project4/       preference elicitation
```

Roughly 10,000 lines of Python across the four apps, of which about 2,500 are tests.

### 1.2 URL map

| URL | View | What it is |
|---|---|---|
| `/` | redirect | to `/home/` - the task sheet tells the reader to open `http://127.0.0.1:8000/` |
| `/home/` | `home.views.index` | hub: names, matriculation numbers, project links |
| `/project1/` | `index` | upload a CSV or load the Iris sample |
| `/project1/explore/` | `explore` | data audit + charts |
| `/project1/train/` | `train` | Task 4 training pipeline |
| `/project1/download/` | `download` | the cleaned dataset as CSV |
| `/project2/` | `index` | the whole explainability interface (one page, state in the query string) |
| `/project2/report/` | `download_report` | PDF |
| `/project3/` | `index` | Tasks 1-4, read from the cache |
| `/project3/expert/` | `be_the_expert` | Task 5: a human answers the deferred articles |
| `/project3/report/` | `download_report` | PDF |
| `/project4/` | `index` | study landing page |
| `/project4/start/`, `/task/`, `/results/` | | the participant interface |
| `/project4/report/`, `/data/` | | PDF, and the answers as CSV |

### 1.3 Conventions the whole repo follows, and why

**Views return `HttpResponse(template.render(context, request))`**, not the `render()`
shortcut. Not a preference: it is what the skeleton's `home/views.py` does, and matching the
existing code was a stated rule.

**Matplotlib never renders into a response.** Every figure is written into `MEDIA_ROOT` and
the template loads it as an `<img src="{{ MEDIA_URL }}...">`. That is the pattern the
`demos` app demonstrates. The backend is forced to `Agg` because there is no GUI on a
server. Figure filenames embed a hash of the state that produced them
(`project2/views.py::_token`), so an unchanged page reuses the file instead of redrawing,
and two different states never overwrite each other's image.

**Logic modules import no Django.** `dataset`, `training`, `learning`, `effects`,
`counterfactuals`, `penguins`, `data`, `experts`, `defer`, `active`, `classifier`, `movies`,
`preference` are plain Python. This is what makes the ML testable without a request cycle,
and it keeps the boundary between "the model" and "the web app" visible. Anything that
touches `settings`, `HttpResponse` or the ORM lives in `views.py`, `plots.py`, `report.py`
or `models.py`.

**One design system.** The categorical palette lives in `project1/plots.py`
(`SERIES_COLOURS`) and Projects 2 and 3 import it, so all four projects draw with the same
colours. The palette was chosen for colour-vision safety: **a scatter caps at four
colours**, because four is the largest set where every *pair* remains distinguishable under
the common forms of colour blindness; a fifth class and beyond fold into a grey "Other"
bucket rather than silently becoming indistinguishable. Identity is never carried by colour
alone - legends, swatches beside table rows, and cell labels repeat the information.

**Bootstrap quirks that are worked around on purpose.** Bootstrap hardcodes `#0d6efd` for
`.form-range` thumbs and `.form-check-input:checked`; those do *not* read `--bs-primary`, so
they are overridden explicitly. And a vendor-prefixed selector (`::-moz-range-thumb`) is
never grouped with an ordinary one, because a browser discards an entire comma-group if it
does not recognise one member - a rule that had silently dropped once already.

---

## 2. Project 1 - Supervised learning interface

**Task sheet:** home page with group members (Task 1); create and register the app (Task 2);
upload a CSV and visualise it (Task 3); a full training pipeline with an explicit decision
about what the user controls (Task 4).

### 2.1 Files

| File | Lines | Responsibility |
|---|---|---|
| `dataset.py` | 526 | read a CSV, work out what it contains, audit quality. No Django. |
| `training.py` | 383 | split, sweep one hyperparameter, score, report. No Django. |
| `plots.py` | 358 | the three exploration charts **and the shared palette for all four projects** |
| `training_plots.py` | 150 | sweep curve, confusion matrix, predicted-vs-actual |
| `views.py` | 384 | three pages plus session handling |
| `tests.py` | 955 | |

### 2.2 Reading a CSV (`dataset.py`)

The task sheet says the last column is the target and there "may be an id column to filter
out". Both are handled as *defaults that the user can override*, because a real upload
breaks both assumptions regularly.

**Problem-type detection** (`detect_problem_type`): a target is treated as regression if it
is numeric with more than `MAX_DISCRETE_LEVELS = 20` distinct values, and classification
otherwise. Twenty is a judgement call, not a theorem - it separates "a rating out of ten"
from "a price".

**Id detection** (`find_id_columns`) is the more interesting one. The first version dropped
any column whose values were unique and sorted, which threw away a genuine feature: in one
dataset `area` happened to be unique and ascending. The rule was tightened to require an
**unbroken run** - `max - min == len - 1` - so only an actual row counter matches. Names in
`ID_COLUMN_NAMES` (`id`, `index`, `unnamed: 0`, ...) also match. A `keep_id_columns` toggle in
the UI lets the user put them back, and the "how this was read" panel states every decision
the app made, so nothing is silently removed.

**Impossible overrides are refused rather than attempted.** Asking for regression on a
string target used to reach matplotlib and raise `Invalid RGBA argument: 'Setosa'`. Now
`Dataset` rejects the override and explains why, the Regression option is disabled in the UI
with the reason shown, and `plots.py` guards independently - the same invariant checked at
both layers, because the second one is what a future caller will hit.

**Iris specifics** the sheet warns about: string labels in `variety` and dotted column names
(`sepal.length`) both work. Detected: 150 rows, 4 numeric features, target `variety`,
classification.

### 2.3 Visualisation (Task 3)

Three chart types, chosen so that every column pairing has something honest to show:

- **Scatter**, coloured by class, with jitter (`JITTER = 0.18`) for discrete axes. Two
  categorical axes fall back to a **count heat-map**, because a scatter of two categories is
  a grid of overprinted dots carrying no information.
- **Distribution** - histogram for a numeric column, bar chart of counts for a categorical
  one.
- **Correlation** - a diverging heat-map over every numeric column, anchored at zero
  (`TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)`) so the neutral midpoint always means "no
  relationship" and the colour cannot mislead.

Cell labels are drawn only while they stay legible: `MAX_LABELLED_CELLS = 144` for counts,
`MAX_CORRELATION_CELLS = 64` for correlations, which are five characters wide.

The caption under each figure describes the figure actually shown. For the correlation grid
it says that each cell uses the rows where *that pair* is present, because `DataFrame.corr`
is pairwise-complete and no single "showing N of M rows" count is true of the whole grid.

### 2.4 The training pipeline (Task 4)

**The division of control - the thing the task sheet asks to be decided explicitly:**

*The user chooses* the model family, the test-set size (10 %-50 %), and the random seed of
the split. *The app decides* preprocessing, which hyperparameter to sweep and over what
grid, and which score to report.

The reasoning: the first three change what the result *means* and need judgement. The rest
is mechanical given the problem type, and getting it wrong - leaking the test set into
scaling, scoring a regression with accuracy - is exactly the class of mistake an interface
should make impossible. The seed is exposed because it is the cheapest way to show a reader
how much of a reported score is sampling noise.

**Four families per problem type**, each sweeping the single hyperparameter that governs how
much structure the model may take on, so the sweep traces the underfit/overfit curve:

| Family | Classification | Regression | Swept | Grid |
|---|---|---|---|---|
| Decision tree | ✔ | ✔ | `max_depth` | 1...20 (11 values) |
| k-nearest neighbours | ✔ | ✔ | `n_neighbors` | 1...31 (9 values) |
| Linear | logistic | ridge | `C` / `alpha` | 0.001...100 |
| Random forest | ✔ | ✔ | `n_estimators` | 10...200 |

**Preprocessing** is inside the `Pipeline`, so it is refitted on the training split alone and
nothing about the test set leaks into the transformation - the single most common silent
error in this kind of app. Numeric columns get median imputation, plus `StandardScaler` only
for the families that need it (k-NN and the linear models; trees and forests are
scale-invariant, so scaling them would be noise in the code). Categorical columns get
most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")` - `ignore` because a
category can appear in the test split and not the training split.

The split is **stratified** when every class has at least two rows, and silently not when it
cannot be (one row of some class). `stratified` is reported in the result so the page can say
which happened.

Scores: accuracy for classification, R² for regression. One failing hyperparameter is
recorded as an error row and does not sink the whole sweep.

**Feature importance** is read off the fitted estimator, not recomputed: `feature_importances_`
for trees and forests, coefficient magnitudes for the linear models, and **`None` for k-NN**,
which has no such notion. Reporting `None` is the point - inventing a number there would be
worse than admitting the model does not have one.

Example (Iris, decision tree, 25 % test): best `max_depth = 4`, test accuracy **0.974**.

### 2.5 Interaction details that came out of real failures

Two bugs here are worth knowing about because both were invisible to the test client:

- The **"Use Iris sample" button was dead**. It sat inside the form whose file input is
  `required`, so HTML5 validation blocked the submit before it reached Django. Fixed by
  splitting into two forms bound with the HTML5 `form` attribute. Django's test client
  ignores HTML5 validation, so there is now an explicit structural test.
- **"Problem type is always classification"** had two causes: a POST without
  `problem_type` overwrote the stored state with `None` (fixed with a hidden `controls`
  marker, so an *unchecked checkbox* still reads as "off" rather than "absent"), and the
  user's target genuinely could not be regressed - which is why the target-column selector
  and the disabled-with-explanation Regression option exist.

---

## 3. Project 2 - Explainability

**Dataset:** Palmer Penguins. **333 usable rows** of 344; the 11 incomplete rows are dropped
and the count is shown, not hidden. **Target: species** (Adelie / Chinstrap / Gentoo).

**The governing objective**, from the lecture:

```
argmin_f  (1/n) Σ loss(f(x_i), y_i) + λ · Ω(f)
```

handled in the interface as the equivalent maximisation the task sheet states:
**`acc_test - λ · Ω(f)`**.

**The single most important point about λ:** it is *not* the fitting-time regularisation
parameter. Each family is fitted **once per fitting-time setting**, producing a menu of
models; λ then picks one off that menu. Moving the slider therefore never refits anything -
which is also why it responds instantly, and why `_fit_family_cached` is `lru_cache`d.

### 3.1 Files

| File | Lines | Responsibility |
|---|---|---|
| `penguins.py` | 139 | load, drop incomplete rows, MADs, category lists |
| `learning.py` | 348 | both model families, Ω, and the λ selection |
| `counterfactuals.py` | 192 | Task 4 |
| `effects.py` | 261 | Task 5 - PDP and ALE, written by hand |
| `plots.py` | 231 | tree drawing, coefficients, trade-off curve, effect curves |
| `views.py` | 377 | one page; all state in the query string |
| `report.py` | 390 | the PDF |
| `tests.py` | 580 | |

### 3.2 Task 1 - one decision tree

A plain `DecisionTreeClassifier()` with **nothing constrained**, grown until its leaves are
pure, fitted by `learning.fit_default_tree()` on the same split as everything else.
Measured: **test accuracy 1.000, 13 leaves, depth 5**, train accuracy 1.000 (exact by
construction - growth only stops at purity).

It is deliberately fitted *outside* the Task 2 grid. Task 1 asks for *a* decision tree with
its accuracy and leaf count and says nothing about a trade-off; every member of the Task 2
family is capped by `max_leaf_nodes`, so none of them is the tree Task 1 describes. The card
also does not move when the λ slider moves, because it does not belong to Task 2.

That 13 leaves already exceeds what a reader will hold in their head is the whole motivation
for Task 2.

### 3.3 Tasks 2 and 3 - the two families and the choice of Ω

| | Fitting-time control | Grid | Ω (complexity) |
|---|---|---|---|
| Decision tree | `max_leaf_nodes` | 2...30 (13 values) | **number of leaves** (as the sheet specifies) |
| Logistic regression | L1 strength `C` | 0.003...10 (11 values) | **number of non-zero coefficients** |

**Why non-zero coefficients for logistic regression.** It is the honest analogue of counting
leaves: both count the pieces of the model a reader has to hold in their head, and L1 is what
actually drives coefficients to zero so the count can fall. A norm such as ‖w‖₂ also measures
"size", but it shrinks coefficients smoothly without ever removing one, so no term ever
disappears and the number would not correspond to anything the reader could skip.

Ω is **read back off the fitted model**, never assumed: a tree capped at 30 leaves may
realise fewer, and `l1_ratio=1.0` does not guarantee how many coefficients survive.
(`l1_ratio=1.0` rather than `penalty="l1"`: scikit-learn 1.8+ deprecated the latter and
mixing them raises an inconsistency warning.)

**Selection** (`learning.select`) maximises `acc_test - λ·Ω`, with **ties going to the
simpler model** - at the λ where a bigger model stops paying for itself, the interpretable
one is the intended answer.

**Switch points.** `learning.switch_points` sweeps λ and returns the values where the
selection changes. They are exposed as `<datalist>` tick marks on the slider and as
one-click jumps, because otherwise the interesting settings can only be found by dragging and
watching. Measured:

| Family | λ = 0 | first switch | second switch |
|---|---|---|---|
| Tree | 12 leaves, acc 0.980 | λ ≥ 0.005 → 4 leaves, 0.940 | λ ≥ 0.010 → 3 leaves, 0.930 |
| Logistic | 12 non-zero, acc 1.000 | λ ≥ 0.0025 → 8, 0.990 | λ ≥ 0.015 → 4, 0.930 |

**Why the slider stops at λ = 0.05.** Every switch that matters happens below about 0.02.
Forcing the *degenerate* model (no split at all, or every coefficient zero) would need λ above
0.14, which would squeeze the whole interesting region into the first tenth of the slider for
no benefit. This is documented in `learning.py` rather than left as an unexplained bound.

One correction worth recording, because it went the other way: it is **not** true that the
top of the range reaches the simplest possible model. At λ = 0.05 a 3-leaf tree
(0.930 - 0.15) still beats a 2-leaf one (0.790 - 0.10). The docstring was corrected rather
than the slider distorted to make the claim true.

### 3.4 Task 4 - counterfactuals

`counterfactuals.generate(pipeline, frame, example, target, k)`:

1. Sample N points **locally** around the chosen example.
2. Keep those the **currently selected model** predicts as the target class.
3. Rank by **MAD-weighted L1 distance** - each feature's difference divided by that
   feature's median absolute deviation, so "1 mm of bill" and "100 g of mass" are comparable
   and the ranking is not dominated by whichever feature happens to have the largest units.
4. Show the best k (user-selectable, 1-10).
5. If none are found, **widen and retry** - increase N and the spread iteratively rather than
   reporting failure on the first attempt.

**Type-aware noising**, as the sheet demands (not "just decimal noise"): numeric features get
Gaussian noise scaled by their MAD; categorical features are **resampled from the observed
categories**; binary features are **flipped** with some probability. Adding 0.3 to "island =
Biscoe" is meaningless, so it is not done.

It is **linked** to the model class and λ of Tasks 1-3 because it reads the same selected
`model.pipeline` - which is the mechanism the sheet asks for, not a coincidence of layout.

### 3.5 Task 5 - PDP and ALE, computed by hand

No library. Both are implemented in `effects.py`, three curves each (one per species).

**PDP** (`partial_dependence`): for each grid value v of the chosen feature, replace that
column with v for *every* row, predict, and average the probabilities. `PDP_GRID` points
across the observed range.

**ALE** (`accumulated_local_effects`): bin the feature, compute the *local* effect inside
each bin, accumulate across bins, then centre the curve on the data. ALE only ever uses rows
that really fall in each bin, which is precisely the difference from PDP: the PDP averages
over feature combinations that **do not occur in the data**, so where the four measurements
are strongly correlated - and in penguins they are - the PDP is the more optimistic of the
two. The two curves therefore agree on the *direction* of an effect and disagree on its
*size*, and the interface says so.

**The partial derivative, which the sheet asks to be exact for one model class and
discretised for the other:**

- **Logistic regression - exact, closed form** (`exact_gradient`). For a softmax over a
  linear function, ∂p_c/∂x_j = p_c · (w_{c,j} - Σ_k p_k · w_{k,j}). The one subtlety is that
  the derivative is with respect to the *original* feature, while the coefficient belongs to
  the *standardised* column, so it is divided by the scaler's scale.
  `learning.numeric_column_index` is the checked way to find where a feature sits in the
  transformed matrix - which is also why `build_preprocessor` puts the numeric block first.
  A test asserts the closed form matches a finite difference, and another asserts the
  gradients sum to zero across the three classes (they must, since the probabilities sum
  to one).
- **Decision tree - finite differences** (`numerical_gradient`, step 1e-4). A tree is
  piecewise constant: its derivative is zero almost everywhere and undefined on the split
  boundaries, so there is no closed form to have. A test asserts exactly that. This is
  visible in the output: the tree's curves move in steps, flat between splits.

---

## 4. Project 3 - Active learning for learning-to-defer

**Dataset:** AG News - 120,000 training and 7,600 test articles, four balanced topics
(World / Sports / Business / Sci-Tech), median 232 characters.

The full experiment takes about **110 seconds**, which is far too long for a page load, so it
is run once by `python manage.py build_project3` and cached to `data/project3/` with joblib.
The views read the cache and, if it is missing, render a page with the command rather than
hanging.

### 4.1 Files

| File | Lines | Responsibility |
|---|---|---|
| `data.py` | 92 | download AG News from Hugging Face, cache, stratified subsample |
| `classifier.py` | 76 | the Task 1 model |
| `experts.py` | 195 | Task 2 - two simulated experts |
| `defer.py` | 216 | Task 3 - competence estimation and the deferral policies |
| `active.py` | 221 | Task 4 - five query strategies |
| `artifacts.py` | 289 | the whole experiment, and the cache |
| `views.py`, `plots.py`, `report.py` | 848 | pages, figures, PDF |
| `management/commands/` | 100 | `build_project3`, `benchmark_classifier` |
| `tests.py` | 467 | |

### 4.2 The split, and why it is not the obvious one

```
120,000 training articles
├── 96,000 (80 %)  fit the classifier
└── 24,000 (20 %)  estimate how often each side is right   <- the "meta" split
     8,000          of those, the active-learning pool
7,600 test articles  evaluation only
```

The deployed classifier is fitted on **80 %**, not on everything. The reason is not
squeamishness about data: on its own training rows the classifier is right almost always, so
a competence or calibration model fitted there would conclude the classifier is **never
wrong**, and the system would never defer. Both accuracies are reported side by side -
baseline on all labels **0.9245**, the deployed 80 % classifier **0.9204** - so the cost of
the decision is visible rather than buried.

### 4.3 Task 1 - the classifier

TF-IDF over word unigrams and bigrams (`sublinear_tf`, `min_df=2`, up to 300,000 features,
English stop words) followed by multinomial logistic regression (`C=4.0`).

**Test accuracy 0.9245, macro F1 0.9243.**

**Why a linear model on n-grams.** Topic in a news snippet is carried almost entirely by
vocabulary - "shares", "midfielder", "Linux" - so this is close to the *right inductive bias*
rather than a compromise. A transformer would score a little higher but would put a GPU
between the reader and the experiment, and every downstream number in this project is about
the interaction between confidence and expert competence, not about the last point of
accuracy.

**Why not LinearSVC**, which is *more* accurate here: it returns a decision-function margin,
not a probability, and every deferral rule compares the classifier's confidence against the
expert's estimated competence - a margin is not on that scale. The comparison is not a
claim from memory: `python manage.py benchmark_classifier` fits both on identical features
and reproduces **logistic 0.9245, LinearSVC 0.9271**.

### 4.4 Task 2 - the simulated experts

Two are implemented, because the two natural readings of "expert in specific regions of the
input space" behave differently and the contrast is the interesting part:

- **`TopicExpert`** (the one the UI shows). Correct 95 % on Sports and Business, 45 %
  elsewhere. Its region is a set of *topics* - inferable from the text but not directly
  observable, so a deferral rule has to learn the topic boundary before it can exploit the
  expert. Measured: overall **0.7046**, inside its beat **0.948**, outside **0.462**, region
  share **0.500**.
- **`KeywordExpert`**. Its region is defined directly by the words present, so competence is
  visible in the input without inference - the more literal reading of "region of the input
  space", and a much easier target to predict.

**The expert is worse than the classifier overall, deliberately.** An expert who was
uniformly better would make the deferral question trivial: always defer. Being better only
*somewhere* is what forces the system to learn where. And the per-topic table shows the
classifier already beating the expert on some topics the expert is strong at, so deferring
the whole beat would *lose* accuracy - the rule has to be finer than "is this in the beat".

**Both experts are deterministic given a seed.** The same article always gets the same
answer, because the active-learning loop can select a point more than once and must not
receive a fresh coin flip each time. This is done with `hashlib.blake2b` over the article
text, *not* Python's `hash()` - string hashing is salted per process, so `hash()` would give
a different answer after a restart and a different answer in each worker of a multi-process
server.

### 4.5 Task 3 - learning to defer

**The rule:** defer ⟺ `P(expert correct | x) > P(classifier correct | x)`.

Both sides have to be *estimated*, and how they are estimated matters more than the rule:

| Signal | Estimates | Test AUC |
|---|---|---|
| Platt-calibrated confidence | P(classifier correct) | **0.845** |
| A text model of classifier correctness | P(classifier correct) | 0.749 |
| Text model of expert correctness | P(expert correct) | 0.760 |

The classifier side uses **calibrated confidence**, not a text model of its own correctness.
Using the weaker signal costs almost the whole benefit - the text-model variant reaches only
**0.9214**, barely above the classifier alone. Platt calibration is a logistic regression on
the confidence (`fit_confidence_calibration`), which maps an over-confident softmax score
onto something that behaves like a probability, so the two sides of the inequality are
finally on the same scale.

**Results** (7,600 test articles):

| Policy | Team accuracy | Deferral rate | Deferral F1 |
|---|---|---|---|
| Classifier alone | 0.9204 | 0 % | - |
| Expert alone | 0.7046 | 100 % | 0.100 |
| **Competence rule, margin 0** | **0.9297** | **8.1 %** | 0.292 |
| Competence rule, text model | 0.9214 | 5.1 % | 0.066 |
| Confidence threshold, tuned in hindsight | 0.9307 | 6.2 % | 0.345 |
| Competence rule, margin tuned in hindsight | 0.9303 | 9.9 % | 0.283 |
| Oracle ceiling | 0.9732 | 5.3 % | 1.000 |

**The headline is 0.9297, the untuned rule** - and this is a deliberate correction. The two
rows marked *"tuned in hindsight"* are the maximum of a 51-point sweep taken **over test-set
accuracy**. They are real numbers but they are upper bounds selected on the evaluation set,
so presenting either as the result would contradict "the test set is for evaluation only".
The reported figure is the theory's own default - margin 0, nothing tuned - which still beats
the classifier by +0.0093. The swept curves are kept as the curve behind the figure and
labelled as hindsight bounds.

Deferral quality is evaluated separately from accuracy, as the sheet asks: `evaluate_policy`
reports **rescued** (the expert fixed a classifier error), **spoiled** (the expert broke a
correct answer), net gain, and F1 against the **oracle** decisions - deferring exactly when
the expert is right and the classifier is wrong. The oracle's 5.3 % deferral rate against the
rule's 8.1 % is the interesting gap: the system defers more than it needs to, because it
cannot tell in advance which of the expert's answers will help.

### 4.6 Task 4 - active learning, and an honest negative result

Now the expert's labels have to be *bought*: 10 rounds of 100 queries from a pool of 8,000.
Five strategies:

| Strategy | Team accuracy | Expert-model AUC | Labels to target |
|---|---|---|---|
| Random (uniform) | 0.9295 | **0.730** | 100 |
| **Boundary + exploration (proposed)** | 0.9297 | 0.706 | 100 |
| Deferral boundary only | 0.9299 | 0.594 | 500 |
| Classifier uncertainty | 0.9304 | 0.586 | 400 |
| Expert-model uncertainty | 0.9268 | 0.646 | 100 |

**The reasoning behind the proposed strategy.** The system does not need a good model of the
expert everywhere; it needs one binary comparison to come out right. So the most valuable
labels are those near the **boundary where the two competences cross**, where a single label
can flip the decision.

**That reasoning is correct and the pure strategy still fails.** Querying only near the
boundary makes the sample unrepresentative of the data the model is later asked about, and the
fitted competence model does not transfer: AUC **0.594** against **0.730** for uniform
sampling. Pure expert-model uncertainty is worse still - it drifts into a skewed corner of
the pool and its estimate of the expert's base rate collapses. This is the classic
sampling-bias failure of greedy active learning, and here it is *measured* rather than
hypothesised.

The proposed fix splits every batch: **half uniform**, to keep the sample representative and
the base rate honest, and half on the boundary. That repairs the mechanism - AUC recovers to
**0.706**.

**The conclusion, stated as it came out:** on this problem active learning does **not** beat
uniform sampling on team accuracy. Two reasons, both visible in the numbers. First, this
expert's competence is defined by topic and topic is easy to read off the text, so a few
hundred labels drawn anywhere already characterise it. Second, the deferral decision only
changes the answer on about 8 % of articles, so team accuracy is insensitive to refinements
in the competence model. Where the targeted strategies show an edge is in the *quality of the
deferral decisions at small budgets*, not in accuracy.

The practical recommendation that follows: use uniform sampling unless the expert's
competence is concentrated in a narrow region random querying would rarely hit - and if a
targeted strategy is used, keep an exploration share in every batch, because the unmixed
version is measurably worse than doing nothing clever at all.

### 4.7 Task 5 (optional) - be the expert

`/project3/expert/` shows a reader the articles **the deferral rule actually handed over** and
asks them to classify them, then scores them against both the simulated expert and the
classifier on those same articles. Same query loop as Task 4 with a human in place of the
simulation.

---

## 5. Project 4 - Preference elicitation

**Dataset:** IMDB 5000 (`movie_metadata.csv`). **35 features over 3,913 films.**

### 5.1 Files

| File | Lines | Responsibility |
|---|---|---|
| `movies.py` | 259 | load, filter, build the interpretable feature matrix |
| `preference.py` | 219 | Plackett-Luce, MAP fit, prediction, recommendations |
| `models.py` | 92 | `StudySession`, `Answer` - the two Django models |
| `study.py` | 245 | the study protocol as executable logic |
| `views.py` | 257 | the participant interface |
| `report.py` | 413 | the PDF (Tasks 1-3 are described there in full) |
| `tests.py` | 506 | |

### 5.2 Task 1 - the feature representation

**The constraint that drives every choice:** a feature has to be something a person could
have a preference *about*, and it has to be present for every film, because a missing feature
cannot be shown in a comparison.

Six blocks, 35 columns:

1. **Genre** - multi-hot, **23 columns**. The primary axis along which people describe their
   own film taste.
2. **Era** - one-hot, 5 buckets. Taste in period is real and *not monotone*: somebody may
   love the 1970s and dislike both the 1950s and the 2010s, which a single "year" number
   could not express.
3. **Certificate** - one-hot, 4 buckets. Stands in for how family-friendly or adult a film is.
4. **Acclaim** - IMDb score, standardised.
5. **Reach** - log number of votes, standardised. The mainstream/obscure axis.
6. **Length** - runtime, standardised.

**Deliberately excluded.** The dataset's popularity columns (Facebook likes for cast and
director, review counts) are well populated but describe a film's *marketing footprint*
rather than anything a viewer has a preference about. Free-text plot keywords would add
thousands of dimensions to estimate from twenty answers.

**The catalogue is filtered, and this is disclosed everywhere it is reported:** 5,043 raw
rows → 4,923 after dropping incomplete ones → 4,026 after requiring ≥ 5,000 votes → **3,913**
after de-duplicating titles. That removes 22.4 %. The task sheet says films "may be selected
uniformly at random from the dataset"; sampling **is** uniform, over a catalogue restricted to
films a participant has a reasonable chance of recognising - which is a precondition for the
answers meaning anything.

### 5.3 Task 2 - the ranking model

**Bradley-Terry.** With utility `U(x) = w·x`, `P(i ≻ j) = σ(w·(x_i - x_j))`. That is exactly
what Design 1 observes.

**The extension: Plackett-Luce.** Design 2 collects a full ranking, so the model must assign a
probability to an *ordering*. Read the ranking as a sequence of choices - favourite out of
n, then favourite out of the remaining n-1, and so on - each a softmax over the utilities
still available:

```
P(i_1 ≻ ... ≻ i_n) = Π_k  exp(U_{i_k}) / Σ_{j≥k} exp(U_{i_j})
```

**Why this extension, which is the question the task sheet actually asks.** Three properties,
the first decisive:

1. It **reduces to Bradley-Terry exactly when n = 2.** Both study designs therefore estimate
   the same `w` under the same likelihood, so the comparison between interfaces measures the
   *interfaces* rather than two different models. Under any other ranking model, a difference
   between designs would confound interface with likelihood.
2. It is consistent with **Luce's choice axiom**, so the pairwise probabilities implied by a
   ranking agree with the pairwise model.
3. Its log-likelihood is **concave** in `w`, so there is one optimum - no restarts, seeds or
   local minima to report.

**The alternative that was rejected:** expanding a ranking of ten into all 45 implied pairwise
comparisons and fitting Bradley-Terry. Those 45 are not independent observations, so the
likelihood would count one answer 45 times and report a spuriously tight posterior.

**Estimation.** MAP with a Gaussian prior - maximise log-likelihood - α‖w‖² with α = 1.0.
The prior is not decoration: with 35 features and perhaps 20 answers the likelihood alone is
under-determined, and the prior is also what picks a single representative from the directions
the data cannot identify. The **gradient is analytic** and the problem is concave, so L-BFGS-B
converges in a few dozen iterations.

### 5.4 Task 3 - the study design

Set out in full in the PDF; implemented in `study.py`:

- **Within-subjects.** Every participant uses both interfaces, so each person is their own
  control and between-person variation in film taste - which is enormous - does not enter the
  comparison.
- **Counterbalanced**, by alternating on the session count rather than tossing a coin, which
  keeps the two orders balanced even for a small sample. Half meet Design 1 first, half
  Design 2, so practice and fatigue cannot masquerade as an effect of the interface.
- **A held-out validation block.** After both designs, everyone answers 10 more pairwise
  questions. These are never fitted; they are the yardstick. **The primary outcome is how
  well the `w` estimated from each design predicts them** - which is what makes the designs
  comparable at all, given that they collect different numbers of raw answers.
- **Matched effort, not matched answers.** Design 1 gives 20 pairwise answers; Design 2 gives
  3 rankings of ten, which is 27 independent choice events. They are deliberately *not*
  matched on count, because the interesting question is what a participant's *time* buys -
  and time is recorded per task.

### 5.5 Task 4 - the participant interface

20 pairwise + 3 rankings of 10 + 10 held-out = **33 tasks**. Answers are stored in the
database (`StudySession`, `Answer`, unique on `(session, block, position)`), with a CSV export
at `/project4/data/`.

**Films are seeded per task, not resampled.** `study.task_seed(session, block, position)`
derives a stable seed, so a reload shows the same films rather than quietly resampling - which
would let a participant shop for an easier question.

That seed is `hashlib.blake2b`, and the reason is worth recording because it was a real bug.
It used to be `abs(hash((token, block, position)))`, and Python salts string hashing per
process: three fresh interpreters returned 2486928632, 415838594 and 1412702808 for identical
inputs. The seed is read in **two separate requests** - once to render the task, once to
re-derive what was shown when the answer arrives - so after a restart, or simply on a
different worker of a multi-process server, the two disagreed. Consequences: rankings became
unanswerable, and for a pairwise task, if the previously chosen film happened to fall in the
newly sampled pair (≈ 2/3913), validation *passed* and the row was stored as a preference
between two films **never shown together**. The permutation check in `study.record` cannot
catch that, because both lists derive from the same new sample. There is now a test pinning
`task_seed` to a constant, because a same-process round trip structurally cannot detect this.

**The analysis** (`study.analyse`) fits `w` from each design separately and from both pooled,
scores each against the held-out block, and reports the strongest and weakest feature weights
plus the top five recommendations.

One measurement bug here is worth knowing about: `predictive_accuracy` returned **1.000 for a
zero weight vector** - `np.argmax` on ties returns index 0, which is the participant's own
pick by construction, so the study's primary outcome measure was silently broken. Fixed with
tie-sharing, which gives the correct **0.500** chance level.

---

## 6. Testing

**307 tests**, `python manage.py test`.

Principles worth stating:

- **Nothing fits on a full dataset.** Project 3's tests build a small bundle once
  (`limit=2000`) and patch `artifacts.load`; components are tested directly.
- `MEDIA_ROOT` is redirected to a temporary directory throughout, so a test run never writes
  into the repository.
- **Django's test client ignores HTML5 validation**, so structural traps have explicit tests:
  a control in the wrong form, a block element inside a `<select>`, leaked template syntax,
  a static reference that does not resolve.
- The stylesheet is **parsed**, not grepped, because a dropped comma-group is invisible to a
  string search.
- Mathematical claims are tested against independent computations: the closed-form gradient
  against a finite difference, gradients summing to zero across classes, the ALE curve
  centred on the data, effects cancelling across the three species.

Some of these tests exist because they caught something real: the missing
`static/favicon/favicon.ico` (a 404 on every page since the skeleton), a patch that landed
inside an `<option>` tag, and a multi-line `{# ... #}` comment rendering as visible page text -
Django's `{# #}` cannot span lines, so `{% comment %}` is used instead.

---

## 7. Traceability - every task and where it lives

| Task | Where | Evidence in the UI |
|---|---|---|
| P1.1 group members | `home/views.py::STUDENTS` | `/home/` |
| P1.2 app created and registered | `project1/`, `pbl/urls.py`, `home/views.py` | link on the hub |
| P1.3 upload and visualise | `dataset.py`, `plots.py` | `/project1/explore/` |
| P1.4 training pipeline | `training.py` | `/project1/train/` |
| P2.1 tree, accuracy, leaves | `learning.fit_default_tree` | Task 1 card |
| P2.2 λ slider, tree Ω = leaves | `learning._fit_trees`, `select` | model + complexity card |
| P2.3 logistic regression, own Ω | `learning._fit_logistic` | same card, model class switch |
| P2.4 counterfactuals | `counterfactuals.generate` | counterfactual region |
| P2.5 PDP + ALE by hand | `effects.py` | feature-effects region |
| P3.1 classifier + test accuracy | `classifier.py` | Task 1 card |
| P3.2 imperfect regional expert | `experts.py` | Task 2 card |
| P3.3 deferral + both evaluations | `defer.py` | Task 3 card |
| P3.4 active learning + justification | `active.py` | Task 4 card |
| P3.5 human expert (optional) | `views.be_the_expert` | `/project3/expert/` |
| P4.1 feature representation | `movies.build_features` | landing page + PDF |
| P4.2 BT extension to rankings | `preference.py` | PDF §2 |
| P4.3 study design | `study.py` + PDF | PDF §3 |
| P4.4 participant interface | `views.task`, `models.py` | `/project4/task/` |

PDF reports are downloadable from Projects 2, 3 and 4. Verified from the task sheets: only
`HCAI-project_03.pdf` and `_04.pdf` actually require one - `_02` does not, and Project 2's
report is extra.

---

## 8. Known limitations, disclosed

Nothing here is a surprise discovered at submission time; all of it is recorded in the code
at the point where it matters.

1. **`home/views.py` still lists a placeholder group member** ("Meow Meow Meow" / "000000"),
   which also appears on the Project 2 PDF cover and in its `author` metadata. Project 1
   Task 1 is not complete until this is real or the entry is removed.
2. **Project 3's `KeywordExpert` is not reachable from the UI.** It is implemented and
   tested, but only `build_project3 --expert keyword` produces its bundle, and that lands at
   a cache path no view loads. The sheet asks for "at least one" expert, so this is not
   non-compliance - but the second expert's analysis is only in the code, not on the page.
3. **Project 3's hindsight rows are upper bounds.** `confidence_best` (0.9307) and
   `competence_best` (0.9303) are argmax over test-set accuracy. They are labelled as such
   and are not the headline. The honest figure is 0.9297.
4. **Project 1 defines no Django models**, despite offering four algorithms. The sheet only
   says one "might have to"; nothing is persisted between requests beyond the session, so
   plain dicts are the right answer.
5. **Project 1's sweep shares one `ColumnTransformer` instance** across every pipeline in the
   sweep (`training.py`). Harmless today - every fit refits it on the same training split -
   but the fitted preprocessors are aliased, so it would bite anyone who later inspected a
   non-winning pipeline.
6. **Project 4's `w` from a single session is a demonstration, not a result.** The analysis
   function is the per-participant half of the comparison; the actual study would run the
   same function over many sessions.
7. **Project 2's λ range cannot reach the degenerate model** (see §3.3) - documented, with the
   reason, in `learning.py`.
8. Some dead code remains: `experts.EXPERT_CHOICES`, `artifacts.is_built`,
   `policies["competence_text_best"]`, `penguins.features_and_target`, a handful of unused
   parameters and imports.

---

## 9. Quick reference - the decisions most worth being able to defend

| Question | Answer |
|---|---|
| Why is λ not the fitting-time parameter? | The family is fitted once per fitting-time setting; λ selects off that menu. Moving it refits nothing. |
| Why Ω = non-zero coefficients for logistic regression? | Same meaning as counting leaves - pieces a reader must hold - and L1 is what lets the count actually fall. ‖w‖₂ never removes a term. |
| Which model's ALE derivative is exact, and why? | Logistic regression: softmax of a linear function has a closed-form derivative. A tree is piecewise constant - derivative zero almost everywhere, undefined on splits - so finite differences. |
| Why does PDP disagree with ALE? | PDP averages over feature combinations that do not occur; ALE only uses rows really in each bin. With correlated features PDP is the optimistic one. |
| Why logistic regression over LinearSVC in P3, when SVC scores higher? | Deferral compares a confidence against a competence. A decision-function margin is not on that scale. |
| Why is the P3 classifier trained on only 80 %? | On its own training rows it is right almost always, so a competence model fitted there would conclude it is never wrong and the system would never defer. |
| Why is the P3 headline 0.9297 and not 0.9307? | 0.9307 is the maximum of a sweep taken over test accuracy - an upper bound selected on the evaluation set. 0.9297 is the untuned default. |
| Why did pure boundary sampling fail? | Sampling bias: the queried sample stops resembling the data the model is asked about, so the competence model does not transfer (AUC 0.594 vs 0.730). |
| Why Plackett-Luce and not 45 pairwise expansions? | It reduces to Bradley-Terry exactly at n = 2, so both designs share one likelihood. The 45 implied comparisons are not independent and would overstate the evidence. |
| Why a Gaussian prior on w? | 35 features, ~20 answers: the likelihood alone is under-determined, and the prior selects one representative from the unidentifiable directions. |
| Why `blake2b` and not `hash()`? | `hash()` is salted per process. The seed is read in two different requests, so a restart or a second worker changed what a participant was shown. |
| Why does a scatter cap at four colours? | Four is the largest set where every *pair* stays distinguishable under common colour-vision deficiencies. Further classes fold into "Other". |
