"""Run the Project 3 experiment and cache it.

    python manage.py build_project3
    python manage.py build_project3 --expert keyword
    python manage.py build_project3 --limit 4000     # quick smoke run
"""

from django.core.management.base import BaseCommand

from project3 import artifacts


class Command(BaseCommand):
    help = "Fit the Project 3 models and cache the results under data/project3/."

    def add_arguments(self, parser):
        parser.add_argument("--expert", default=artifacts.DEFAULT_EXPERT,
                            choices=["topic", "keyword"])
        parser.add_argument("--seed", type=int, default=artifacts.DEFAULT_SEED)
        parser.add_argument("--limit", type=int, default=None,
                            help="Subsample the training set (for a quick run).")
        parser.add_argument("--rounds", type=int, default=artifacts.AL_ROUNDS)
        parser.add_argument("--batch", type=int, default=artifacts.AL_BATCH)

    def handle(self, *args, **options):
        artifacts.build(
            expert_kind=options["expert"],
            seed=options["seed"],
            limit=options["limit"],
            rounds=options["rounds"],
            batch=options["batch"],
            log=lambda message: self.stdout.write(str(message)),
        )
        self.stdout.write(self.style.SUCCESS("done"))
