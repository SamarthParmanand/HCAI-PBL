"""The IMDB 5000 dataset and the movie feature representation: Project 4, Task 1.

Free of Django imports.

**What the features have to be good for.** The utility model is U(x) = w.x with one
w per participant, estimated from a few dozen interactions. That budget drives
every choice here:

* the representation has to be *low-dimensional*, because w has to be pinned down
  from perhaps twenty comparisons;
* it has to be *interpretable*, because the point of eliciting w is to be able to
  say what the person likes, not merely to rank for them;
* every feature has to be present for essentially every film, since a movie with
  missing features cannot be shown in a comparison.

That rules out the dataset's popularity columns as *taste* features even though
they are well populated (Facebook likes for cast and director, review counts):
they describe a film's marketing footprint rather than anything a viewer has a
preference about. It also rules out free-text plot keywords, which would add
thousands of dimensions to estimate from twenty answers.

The six blocks kept, and why:

1. **Genre** (multi-hot, 23 columns) - the primary axis along which people
   describe their own film taste, and complete for every row of the catalogue.
2. **Era** (one-hot, 5 buckets) - taste in period is real and *not* monotone:
   somebody may love the 1970s and dislike both 1950s and 2010s films, which a
   single "year" number could not express.
3. **Certificate** (one-hot, 4 buckets) - stands in for how family-friendly or
   adult a film is.
4. **Acclaim** - IMDb score, standardised.
5. **Reach** - log number of votes, standardised: the mainstream/obscure axis.
6. **Length** - runtime, standardised.

**One identifiability note.** The Plackett-Luce likelihood is invariant to adding
a constant to every utility, and because each film has exactly one era bucket and
exactly one certificate bucket, adding a constant to all five era weights (or all
four certificate weights) leaves every choice probability unchanged. Those two
directions are therefore not identifiable from preference data alone. The L2 prior
in `preference.py` resolves it by selecting the minimum-norm representative, which
is the one whose per-block weights are centred on zero; the estimated weights
should be read as relative within a block, never as absolute levels.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "movie_metadata.csv")

# note to self: era is one-hot buckets rather than a single year number on
# purpose. taste in period is not monotone, somebody can love the seventies and
# dislike both the fifties and the twenty-tens, and one number cannot express
# that.
ERA_BUCKETS = [
    ("Pre-1970", -np.inf, 1970),
    ("1970s-80s", 1970, 1990),
    ("1990s", 1990, 2000),
    ("2000s", 2000, 2010),
    ("2010s", 2010, np.inf),
]

CERTIFICATE_BUCKETS = {
    "G": "Family (G/PG)",
    "PG": "Family (G/PG)",
    "TV-G": "Family (G/PG)",
    "TV-PG": "Family (G/PG)",
    "Approved": "Family (G/PG)",
    "PG-13": "Teen (PG-13)",
    "TV-14": "Teen (PG-13)",
    "R": "Adult (R)",
    "TV-MA": "Adult (R)",
    "NC-17": "Adult (R)",
    "X": "Adult (R)",
}
CERTIFICATE_OTHER = "Unrated / other"

NUMERIC_BLOCK = [
    ("acclaim", "IMDb score"),
    ("reach", "Audience reach (log votes)"),
    ("length", "Runtime"),
]

# note to self: the catalogue is filtered and i should say so before anyone asks.
# 5043 raw rows, then 4923 after dropping incomplete ones, then 4026 after
# requiring five thousand votes, then 3913 after de-duplicating titles, so 22.4
# percent is removed. the sheet says films may be drawn uniformly at random from
# the dataset, and the sampling genuinely is uniform, just over a catalogue
# restricted to films a participant has some chance of recognising, which is a
# precondition for the answers meaning anything at all.

# Films below this many votes are dropped: an elicitation interface should show
# films a participant has some chance of recognising, and the long tail of the
# dataset is largely unknown titles.
MIN_VOTES = 5000


class DatasetMissing(RuntimeError):
    pass


def load_raw():
    if not os.path.exists(CSV):
        raise DatasetMissing(
            f"movie_metadata.csv was not found at {CSV}. It is the IMDB 5000 "
            f"Movie Dataset.")
    return pd.read_csv(CSV)


def _clean_title(value):
    # Titles in this file carry a trailing non-breaking space.
    return str(value).replace("\xa0", "").strip()


def _era(year):
    for name, low, high in ERA_BUCKETS:
        if low <= year < high:
            return name
    return ERA_BUCKETS[-1][0]


def load():
    """Usable films with the columns the interface and the model need."""
    frame = load_raw()

    frame = frame.rename(columns={"movie_title": "title"})
    frame["title"] = frame["title"].map(_clean_title)

    needed = ["title", "genres", "title_year", "duration", "imdb_score",
              "num_voted_users"]
    frame = frame.dropna(subset=needed)
    frame = frame[frame["num_voted_users"] >= MIN_VOTES]

    # One row per film: the dataset has a handful of duplicated titles.
    frame = frame.drop_duplicates(subset=["title", "title_year"])

    frame["year"] = frame["title_year"].astype(int)
    frame["era"] = frame["year"].map(_era)
    frame["certificate"] = (frame["content_rating"]
                            .map(CERTIFICATE_BUCKETS)
                            .fillna(CERTIFICATE_OTHER))
    frame["genre_list"] = frame["genres"].map(
        lambda value: [part for part in str(value).split("|") if part])
    frame["director"] = frame["director_name"].fillna("Unknown")

    frame = frame.reset_index(drop=True)
    frame["movie_id"] = frame.index.astype(int)
    return frame


def genre_vocabulary(frame):
    genres = set()
    for row in frame["genre_list"]:
        genres.update(row)
    return sorted(genres)


def feature_names(frame):
    """Human-readable name for every column of the design matrix, in order."""
    names = [f"Genre: {genre}" for genre in genre_vocabulary(frame)]
    names += [f"Era: {name}" for name, _low, _high in ERA_BUCKETS]
    names += [f"Certificate: {name}" for name in _certificate_levels()]
    names += [label for _key, label in NUMERIC_BLOCK]
    return names


def _certificate_levels():
    seen = []
    for name in CERTIFICATE_BUCKETS.values():
        if name not in seen:
            seen.append(name)
    seen.append(CERTIFICATE_OTHER)
    return seen


# note to self: task 1 is a judgement call rather than a coding problem, so the
# justification is the answer. the rule i used is that a feature has to be
# something a person could actually hold a preference about, and it has to be
# present for every film, because a missing feature cannot be shown in a
# comparison. that is why the facebook like counts are left out even though they
# are well populated, they describe a film's marketing footprint rather than
# anything a viewer has a taste about, and why plot keywords are out, thousands
# of dimensions to estimate from twenty answers.
def build_features(frame):
    """The design matrix X and the name of every column.

    Numeric columns are standardised over the whole catalogue so that a unit of w
    means the same thing in each: without that, the runtime weight would be
    thousands of times smaller than the IMDb-score weight and the L2 prior would
    penalise them incomparably.
    """
    genres = genre_vocabulary(frame)
    eras = [name for name, _low, _high in ERA_BUCKETS]
    certificates = _certificate_levels()

    blocks = []

    genre_matrix = np.zeros((len(frame), len(genres)))
    index = {genre: position for position, genre in enumerate(genres)}
    for row, values in enumerate(frame["genre_list"]):
        for genre in values:
            genre_matrix[row, index[genre]] = 1.0
    blocks.append(genre_matrix)

    era_matrix = np.zeros((len(frame), len(eras)))
    era_index = {name: position for position, name in enumerate(eras)}
    for row, value in enumerate(frame["era"]):
        era_matrix[row, era_index[value]] = 1.0
    blocks.append(era_matrix)

    certificate_matrix = np.zeros((len(frame), len(certificates)))
    certificate_index = {name: position for position, name
                         in enumerate(certificates)}
    for row, value in enumerate(frame["certificate"]):
        certificate_matrix[row, certificate_index[value]] = 1.0
    blocks.append(certificate_matrix)

    numeric = np.column_stack([
        frame["imdb_score"].to_numpy(dtype=float),
        np.log10(frame["num_voted_users"].to_numpy(dtype=float)),
        frame["duration"].to_numpy(dtype=float),
    ])
    numeric = (numeric - numeric.mean(axis=0)) / numeric.std(axis=0)
    blocks.append(numeric)

    matrix = np.hstack(blocks)
    return matrix, feature_names(frame)


def block_slices(frame):
    """Where each block sits in the design matrix, for reading w back out."""
    genres = genre_vocabulary(frame)
    eras = [name for name, _low, _high in ERA_BUCKETS]
    certificates = _certificate_levels()

    start = 0
    slices = {}
    for name, size in (("genre", len(genres)), ("era", len(eras)),
                       ("certificate", len(certificates)),
                       ("numeric", len(NUMERIC_BLOCK))):
        slices[name] = slice(start, start + size)
        start += size
    return slices


def describe(frame):
    """Counts for the interface and the report."""
    raw = load_raw()
    return {
        "n_raw": int(len(raw)),
        "n_usable": int(len(frame)),
        "min_votes": MIN_VOTES,
        "n_genres": len(genre_vocabulary(frame)),
        "n_features": len(feature_names(frame)),
        "year_range": (int(frame["year"].min()), int(frame["year"].max())),
        "eras": [name for name, _low, _high in ERA_BUCKETS],
        "certificates": _certificate_levels(),
        "genres": genre_vocabulary(frame),
    }


def display(frame, ids):
    """The fields the interface shows for a set of films."""
    rows = []
    for movie_id in ids:
        row = frame.loc[int(movie_id)]
        rows.append({
            "movie_id": int(movie_id),
            "title": row["title"],
            "year": int(row["year"]),
            "genres": ", ".join(row["genre_list"][:3]),
            "director": row["director"],
            "score": float(row["imdb_score"]),
            "duration": int(row["duration"]),
            "certificate": row["certificate"],
        })
    return rows
