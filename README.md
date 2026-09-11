# HCAI-PBL

Project-based work for the TUHH course **Human-Centric Artificial Intelligence**. All four
projects live in one Django project (`pbl/`), each as its own app, reachable from a shared
hub at the site root.

| | | |
|---|---|---|
| **Project 1** | Supervised learning interface | `/project1/` |
| **Project 2** | Explainability (Palmer Penguins) | `/project2/` |
| **Project 3** | Active learning for learning-to-defer (AG News) | `/project3/` |
| **Project 4** | Preference elicitation (IMDB 5000) | `/project4/` |

Projects 2, 3 and 4 each offer a downloadable PDF report from their own page.

## Setup

From the project root, in a Python 3.13 virtual environment:

```bash
python -m pip install -r requirements.txt
python manage.py migrate              # Project 4 stores its answers in the database
python manage.py build_project3       # ~2 minutes; downloads AG News (19 MB) on first run
python manage.py runserver            # -> http://127.0.0.1:8000/
```

All four steps are needed for the whole site:

* **`migrate`** creates the Project 4 study tables. `db.sqlite3` is not checked in, so
  without this `/project4/` reports that the tables are missing instead of working.
* **`build_project3`** runs the Project 3 experiment once and caches it under `data/`.
  Fitting on all 120,000 articles is too slow for a page load, so the views read the
  cache; without it `/project3/` shows the command to run rather than hanging. Add
  `--limit 8000` for a fast, lower-accuracy run.

Projects 1 and 2 need neither step. `iris.csv`, `penguins.csv` and `movie_metadata.csv`
are committed; AG News and the Project 3 cache are downloaded and generated into the
gitignored `data/`.

Bootstrap 5.3.3 is vendored under `static/vendor/`, so the site renders with no network
access.

## Tests

```bash
python manage.py test
```

The suite never fits on a full dataset: Project 3's tests build a small bundle and patch
the artifact loader, and `MEDIA_ROOT` is redirected to a temporary directory throughout.
