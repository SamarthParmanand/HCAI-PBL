"""Project 4: preference elicitation.

The landing page does the two things the task sheet asks of it: download the PDF
covering the feature representation, the ranking model and the study design
(Tasks 1 to 3), and start the study itself (Task 4).
"""

import csv
import time
from functools import lru_cache

from django.db import DatabaseError
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template import loader
from django.utils import timezone

from . import movies, preference, report, study
from .models import PAIRWISE, RANKING, VALIDATION, StudySession

SESSION_KEY = "project4_token"
SHOWN_KEY = "project4_shown_at"


@lru_cache(maxsize=1)
def catalogue():
    """The films and their design matrix, built once per process.

    Reading and encoding 5,000 rows takes a moment and never changes, so it is
    cached rather than rebuilt on every request.
    """
    frame = movies.load()
    features, names = movies.build_features(frame)
    return frame, features, names


def _session(request, create=False):
    """The participant's session, if any."""
    token = request.session.get(SESSION_KEY)
    if token:
        found = StudySession.objects.filter(token=token).first()
        if found is not None:
            return found

    if not create:
        return None

    created = study.new_session()
    request.session[SESSION_KEY] = created.token
    return created


def _not_migrated(request, error):
    """`migrate` has not been run, so there is nowhere to put answers.

    `db.sqlite3` is gitignored, so this is exactly what a fresh clone hits. A
    500 would leave the reader guessing; Project 3 already handles its own
    missing cache the same way.
    """
    template = loader.get_template("project4/not_migrated.html")
    context = {"command": "python manage.py migrate", "error": error}
    return HttpResponse(template.render(context, request), status=503)


def index(request):
    """The landing page."""
    frame, features, _names = catalogue()

    try:
        session = _session(request)
        completed_sessions = StudySession.objects.filter(
            finished_at__isnull=False).count()
        total_sessions = StudySession.objects.count()
    except DatabaseError as failure:
        return _not_migrated(request, failure)
    template = loader.get_template("project4/index.html")
    context = {
        "dataset": movies.describe(frame),
        "n_features": features.shape[1],
        "pairwise_tasks": study.PAIRWISE_TASKS,
        "ranking_tasks": study.RANKING_TASKS,
        "ranking_size": study.RANKING_SIZE,
        "validation_tasks": study.VALIDATION_TASKS,
        "total_tasks": study.total_tasks(),
        "session": session,
        "in_progress": session is not None and not session.complete,
        "completed_sessions": completed_sessions,
        "total_sessions": total_sessions,
    }
    return HttpResponse(template.render(context, request))


def start(request):
    """Begin (or restart) a run through the study."""
    if request.method != "POST":
        return redirect("project4:index")

    if "restart" in request.POST:
        request.session.pop(SESSION_KEY, None)

    _session(request, create=True)
    return redirect("project4:task")


def task(request):
    """One elicitation task: a pair to choose from, or ten films to rank."""
    frame, features, names = catalogue()

    session = _session(request)
    if session is None:
        return redirect("project4:index")

    if request.method == "POST":
        error = _accept(request, session, frame)
        if error is None:
            return redirect("project4:task")
    else:
        error = None

    current = study.current_task(session, frame)
    if current is None:
        if session.finished_at is None:
            session.finished_at = timezone.now()
            session.save(update_fields=["finished_at"])
        return redirect("project4:results")

    # Recorded so the time each answer took can be reported per design.
    request.session[SHOWN_KEY] = time.time()

    template = loader.get_template("project4/task.html")
    context = {
        "session": session,
        "task": current,
        "is_ranking": current["block"] == RANKING,
        "is_validation": current["block"] == VALIDATION,
        "ranks": list(range(1, len(current["movies"]) + 1)),
        "done": study.completed_tasks(session),
        "total": study.total_tasks(),
        "percent": int(100 * study.completed_tasks(session) / study.total_tasks()),
        "error": error,
    }
    return HttpResponse(template.render(context, request))


def _accept(request, session, frame):
    """Validate and store one submitted answer. Returns an error message or None."""
    block = request.POST.get("block")
    try:
        position = int(request.POST.get("position", ""))
    except ValueError:
        return "That answer could not be read. Please try again."

    if block not in (PAIRWISE, RANKING, VALIDATION):
        return "That answer could not be read. Please try again."

    expected_block, expected_position = study.progress(session)
    if (block, position) != (expected_block, expected_position):
        # A double submit or a stale form: the next task is already the right one.
        return None

    shown = study.sample_movies(
        frame, study.BLOCK_ITEMS[block], study.task_seed(session, block, position))

    if block == RANKING:
        ordering, error = _read_ranking(request, shown)
        if error:
            return error
    else:
        try:
            chosen = int(request.POST.get("choice", ""))
        except ValueError:
            return "Please choose one of the two films."
        if chosen not in shown:
            return "Please choose one of the two films."
        other = [value for value in shown if value != chosen]
        ordering = [chosen] + other

    started = request.session.get(SHOWN_KEY)
    seconds = (time.time() - started) if started else None

    try:
        study.record(session, block, position, shown, ordering, seconds=seconds)
    except ValueError as invalid:
        return str(invalid)
    return None


def _read_ranking(request, shown):
    """Read the rank each film was given, and check it is a real ranking."""
    assigned = {}
    for movie_id in shown:
        raw = request.POST.get(f"rank_{movie_id}", "")
        try:
            rank = int(raw)
        except ValueError:
            return None, "Give every film a rank from 1 to %d." % len(shown)
        if not 1 <= rank <= len(shown):
            return None, "Ranks must be between 1 and %d." % len(shown)
        assigned[movie_id] = rank

    if sorted(assigned.values()) != list(range(1, len(shown) + 1)):
        return None, ("Each rank from 1 to %d must be used exactly once - two "
                      "films currently share a rank." % len(shown))

    ordered = sorted(shown, key=lambda movie_id: assigned[movie_id])
    return ordered, None


def results(request):
    """What the session produced: an estimate of w from each design."""
    frame, features, names = catalogue()

    session = _session(request)
    if session is None:
        return redirect("project4:index")

    block, _position = study.progress(session)
    if block is not None:
        return redirect("project4:task")

    analysis = study.analyse(session, frame, features, names)

    template = loader.get_template("project4/results.html")
    context = {
        "session": session,
        "analysis": analysis,
        "designs": [analysis[PAIRWISE], analysis[RANKING]],
        "combined": analysis["combined"],
        "held_out": analysis["held_out"],
    }
    return HttpResponse(template.render(context, request))


def download_report(request):
    """The PDF covering Tasks 1 to 3."""
    frame, features, names = catalogue()
    pdf = report.build(frame, features, names)
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        'attachment; filename="HCAI-project4-report.pdf"')
    return response


def download_data(request):
    """Every answer collected so far, as CSV.

    A real study needs its data out of the application, so this is the export a
    researcher would actually use.
    """
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        'attachment; filename="project4-responses.csv"')

    writer = csv.writer(response)
    for row in study.export_rows():
        writer.writerow(row)
    return response
