"""Reproduce the model-choice comparison quoted in `project3/classifier.py`.

    python manage.py benchmark_classifier
    python manage.py benchmark_classifier --limit 8000    # quick, lower accuracy

`classifier.py` says logistic regression is chosen over the marginally more
accurate `LinearSVC` because it returns probabilities, and quotes both numbers.
Nothing in the running app ever fits a `LinearSVC`, so without this command the
comparison would be a claim with no code behind it. This fits both models on
exactly the same TF-IDF features and prints the two test accuracies.

Takes a couple of minutes on the full 120,000 articles.
"""

from django.core.management.base import BaseCommand

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from project3 import classifier, data


class Command(BaseCommand):
    help = ("Fit logistic regression and LinearSVC on identical features and "
            "report both test accuracies.")

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=None,
                            help="Subsample the training set (for a quick run).")
        parser.add_argument("--seed", type=int, default=classifier.SEED)

    def handle(self, *args, **options):
        train, test = data.load()
        if options["limit"]:
            train = train.sample(n=min(options["limit"], len(train)),
                                 random_state=options["seed"])

        x_train = train["text"].to_numpy()
        y_train = train["label"].to_numpy()
        x_test = test["text"].to_numpy()
        y_test = test["label"].to_numpy()

        self.stdout.write(f"{len(x_train)} training and {len(x_test)} test articles")

        # The same pipeline the app deploys.
        logistic = classifier.fit(x_train, y_train, seed=options["seed"])
        logistic_accuracy = accuracy_score(y_test, logistic.predict(x_test))

        # Identical features, the vectoriser rebuilt from the same builder so the
        # two models genuinely differ only in the estimator.
        reference = classifier.build(seed=options["seed"])
        svc = Pipeline([
            ("tfidf", reference.named_steps["tfidf"]),
            ("model", LinearSVC(C=1.0, random_state=options["seed"])),
        ])
        svc.fit(x_train, y_train)
        svc_accuracy = accuracy_score(y_test, svc.predict(x_test))

        self.stdout.write(f"logistic regression  {logistic_accuracy:.4f}")
        self.stdout.write(f"LinearSVC            {svc_accuracy:.4f}")
        self.stdout.write(
            "LinearSVC scores higher but returns a decision-function margin, not "
            "a probability, which is what the deferral rules in defer.py compare "
            "against the expert's estimated competence.")
        self.stdout.write(self.style.SUCCESS("done"))
