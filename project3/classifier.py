"""The AG News classifier for Project 3, Task 1.

**Why this model.** TF-IDF over word unigrams and bigrams followed by
multinomial logistic regression. AG News is a topic-classification task on short
news snippets, where topic is carried almost entirely by vocabulary ("shares",
"midfielder", "Linux"), so a linear model on n-gram counts is close to the right
inductive bias rather than a compromise. It also fits in under a minute on the
full 120,000 rows, which matters here because the later tasks need the model's
*calibrated confidence*, not just its label: a transformer would score a little
higher but would put a GPU between the reader and the experiment, and every
downstream number in this project is about the interaction between confidence and
expert competence rather than about the last point of accuracy.

Logistic regression is chosen over the marginally more accurate `LinearSVC`
(measured: 0.9271 against 0.9245 on the same features -- reproduce with
`python manage.py benchmark_classifier`) precisely because it returns
probabilities. The deferral rules in `defer.py` compare the classifier's
confidence against the expert's estimated competence, and a decision-function
margin is not on that scale.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from .data import CLASS_NAMES

SEED = 0


# note to self: the model choice question. tf-idf over unigrams and bigrams plus
# a linear model is close to the right inductive bias here rather than a
# compromise, because topic in a news snippet is carried almost entirely by
# vocabulary, words like shares, midfielder, linux. linearsvc actually scores
# higher on the same features, 0.9271 against 0.9245, and is still not used
# because it returns a decision function margin while every later task needs a
# probability to compare against the expert's competence. manage.py
# benchmark_classifier reproduces both numbers if anyone asks for proof.
def build(max_features=300000, C=4.0, seed=SEED):
    """An unfitted text-classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
        )),
        ("model", LogisticRegression(max_iter=1000, C=C, random_state=seed)),
    ])


def fit(texts, labels, **kwargs):
    pipeline = build(**kwargs)
    pipeline.fit(texts, labels)
    return pipeline


def evaluate(pipeline, texts, labels):
    """Test-set numbers for Task 1."""
    predicted = pipeline.predict(texts)
    matrix = confusion_matrix(labels, predicted, labels=range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "labels": list(CLASS_NAMES),
        "matrix": matrix.tolist(),
        "per_class": {
            CLASS_NAMES[index]: {
                "n": int(matrix[index].sum()),
                "recall": float(matrix[index][index] / matrix[index].sum())
                if matrix[index].sum() else float("nan"),
            }
            for index in range(len(CLASS_NAMES))
        },
    }


def confidence(pipeline, texts):
    """The classifier's own probability for the label it chose."""
    return pipeline.predict_proba(texts).max(axis=1)
