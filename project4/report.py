"""The downloadable PDF for Project 4.

The task sheet asks the landing page to offer a PDF explaining the implemented
method (Tasks 1 and 2) and the design of the user study (Task 3). The study is
explicitly not to be run, so this document is the deliverable for Task 3: the
protocol that would be followed if it were.
"""

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from home.views import STUDENTS

from . import movies, study

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#cbd5e1")
BAND = colors.HexColor("#f1f5f9")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontName="Helvetica-Bold", fontSize=20,
                                leading=24, textColor=INK, alignment=TA_LEFT,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontName="Helvetica", fontSize=10.5,
                                   leading=14, textColor=MUTED, spaceAfter=14),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontName="Helvetica-Bold", fontSize=13, leading=16,
                             textColor=INK, spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Heading3"],
                             fontName="Helvetica-Bold", fontSize=11, leading=14,
                             textColor=INK, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontName="Helvetica", fontSize=9.5, leading=13.5,
                               textColor=INK, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
                                 fontName="Helvetica", fontSize=9.5,
                                 leading=13.5, textColor=INK, spaceAfter=2),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
                                  fontName="Helvetica-Oblique", fontSize=8.5,
                                  leading=11, textColor=MUTED, spaceAfter=10),
    }


CELL_PADDING = 6


def _cell_style(name, font, size, colour):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=size + 2,
                          textColor=colour)


def _wrap_long_cells(rows, widths):
    """Let an over-long cell wrap instead of running off the page.

    A plain string in a Table cell is drawn on a single line: reportlab does not
    wrap it, so a long sentence overflows its column, prints over the next one
    and can run clean past the right margin. Measuring every cell against its
    column and promoting the over-long ones to Paragraphs is what makes them
    wrap inside the column instead. Short cells are left as plain strings so
    the common case is untouched.
    """
    header = _cell_style("cell-head", "Helvetica-Bold", 8.5, MUTED)
    body = _cell_style("cell-body", "Helvetica", 8.5, INK)

    wrapped = []
    for index, row in enumerate(rows):
        style = header if index == 0 else body
        cells = []
        for column, cell in enumerate(row):
            limit = widths[column] - 2 * CELL_PADDING
            if (isinstance(cell, str)
                    and stringWidth(cell, style.fontName, style.fontSize) > limit):
                cells.append(Paragraph(cell, style))
            else:
                cells.append(cell)
        wrapped.append(cells)
    return wrapped


def _table(rows, widths):
    table = Table(_wrap_long_cells(rows, widths), colWidths=widths,
                  hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _bullets(items, styles):
    return ListFlowable(
        [ListItem(Paragraph(text, styles["bullet"]), leftIndent=12)
         for text in items],
        bulletType="bullet", start="circle", leftIndent=12,
    )


def build(frame, features, names):
    """Render the report. Returns PDF bytes."""
    styles = _styles()
    story = []
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="HCAI Project 4 - Preference elicitation",
        author=", ".join(student["name"] for student in STUDENTS),
    )

    described = movies.describe(frame)

    story.append(Paragraph("Project 4: Preference elicitation", styles["title"]))
    story.append(Paragraph(
        f"Human-Centric Artificial Intelligence &middot; IMDB 5000 Movie Dataset "
        f"&middot; generated {date.today().isoformat()}", styles["subtitle"]))

    story.append(_table(
        [["Group member", "Matriculation"]] +
        [[s["name"], s["matriculation"]] for s in STUDENTS],
        [9 * cm, 5 * cm]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "This document covers the feature representation (Task 1), the extension "
        "of Bradley-Terry to rankings (Task 2) and the design of the user study "
        "(Task 3). The study itself is implemented as an interactive interface and "
        "is reachable from the project's landing page; as the task sheet states, it "
        "is not run here.", styles["body"]))

    # --- Task 1 --------------------------------------------------------
    story.append(Paragraph("1. Task 1: the feature representation", styles["h2"]))
    story.append(Paragraph(
        f"The utility of a film to a participant is U(x) = w&middot;x, with one w "
        f"per person estimated from a few dozen interactions. That budget drives "
        f"every choice below: the representation has to be low-dimensional, because "
        f"w must be pinned down from perhaps twenty answers; interpretable, because "
        f"the point of eliciting w is to be able to say what somebody likes; and "
        f"complete, because a film with a missing feature cannot be shown in a "
        f"comparison.", styles["body"]))

    story.append(Paragraph("What was excluded, and why", styles["h3"]))
    story.append(Paragraph(
        "The dataset's popularity columns - Facebook likes for the cast and "
        "director, critic and user review counts - are well populated but describe "
        "a film's marketing footprint rather than anything a viewer holds a "
        "preference about, so they are not taste features. Free-text plot keywords "
        "were also excluded: they would add thousands of dimensions to estimate "
        "from twenty answers.", styles["body"]))

    story.append(Paragraph("The six blocks", styles["h3"]))
    story.append(_table([
        ["Block", "Columns", "Encoding", "Why"],
        ["Genre", str(described["n_genres"]), "multi-hot",
         "The axis along which people describe their own film taste, and complete "
         "for every row."],
        ["Era", str(len(described["eras"])), "one-hot",
         "Taste in period is real and not monotone: somebody may love the 1970s and "
         "dislike both the 1950s and the 2010s, which a single year number cannot "
         "express."],
        ["Certificate", str(len(described["certificates"])), "one-hot",
         "Stands in for how family-friendly or adult a film is."],
        ["Acclaim", "1", "standardised", "IMDb score."],
        ["Reach", "1", "standardised",
         "Log number of votes: the mainstream-versus-obscure axis."],
        ["Length", "1", "standardised", "Runtime."],
    ], [2.6 * cm, 1.6 * cm, 2.4 * cm, 9.4 * cm]))

    story.append(Paragraph(
        f"That gives {described['n_features']} features over "
        f"{described['n_usable']:,} usable films, drawn from the "
        f"{described['n_raw']:,} rows of the dataset. Films with fewer than "
        f"{described['min_votes']:,} votes are dropped: an elicitation interface "
        f"should show films a participant has some chance of recognising, and the "
        f"long tail of the dataset is largely unknown titles. Rows missing a title, "
        f"genre, year, runtime or score are dropped for the same reason. The "
        f"surviving films span {described['year_range'][0]} to "
        f"{described['year_range'][1]}.", styles["body"]))

    story.append(Paragraph(
        "The three numeric features are standardised across the catalogue. Without "
        "that, a unit of w would mean something wildly different in each - runtime "
        "in minutes against a score out of ten - and the Gaussian prior would "
        "penalise them incomparably.", styles["body"]))

    story.append(Paragraph("An identifiability note", styles["h3"]))
    story.append(Paragraph(
        "The likelihood below is invariant to adding a constant to every utility. "
        "Because each film carries exactly one era bucket and exactly one "
        "certificate bucket, adding a constant to all era weights - or all "
        "certificate weights - leaves every choice probability unchanged, so those "
        "two directions are not identifiable from preference data at all. The prior "
        "resolves it by selecting the minimum-norm representative, whose weights "
        "are centred within each block. Estimated weights should therefore be read "
        "as relative within a block, never as absolute levels.", styles["body"]))

    # --- Task 2 --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("2. Task 2: from pairwise choice to rankings",
                           styles["h2"]))
    story.append(Paragraph(
        "Bradley-Terry gives the probability that film i is preferred to film j as "
        "the logistic function of the utility difference:", styles["body"]))
    story.append(Paragraph(
        "P(i &gt; j) = exp(U<sub>i</sub>) / (exp(U<sub>i</sub>) + "
        "exp(U<sub>j</sub>)) = &sigma;(w&middot;(x<sub>i</sub> &minus; "
        "x<sub>j</sub>))", styles["body"]))
    story.append(Paragraph(
        "which is exactly what Design 1 observes. Design 2 asks for a full ranking "
        "of ten films, so the model must assign a probability to an ordering.",
        styles["body"]))

    story.append(Paragraph("The proposed extension: Plackett-Luce", styles["h3"]))
    story.append(Paragraph(
        "Read the ranking as a sequence of choices. The participant first picks "
        "their favourite from all n films, then their favourite from the remaining "
        "n&minus;1, and so on. Each stage is a softmax over the utilities of the "
        "films still available:", styles["body"]))
    story.append(Paragraph(
        "P(i<sub>1</sub> &gt; i<sub>2</sub> &gt; &hellip; &gt; i<sub>n</sub>) = "
        "&prod;<sub>k=1..n</sub> exp(U<sub>i<sub>k</sub></sub>) / "
        "&sum;<sub>j&ge;k</sub> exp(U<sub>i<sub>j</sub></sub>)", styles["body"]))

    story.append(Paragraph("Why this extension", styles["h3"]))
    story.append(_bullets([
        "<b>It reduces to Bradley-Terry exactly at n = 2.</b> This is the decisive "
        "property. Both designs then estimate the same w under the same likelihood, "
        "so the study measures the two <i>interfaces</i> rather than two different "
        "models. Under any other ranking model, a difference between designs would "
        "confound the interface with the likelihood used to analyse it.",
        "It satisfies Luce's choice axiom, so the pairwise probabilities implied by "
        "a ranking agree with the pairwise model.",
        "Its log-likelihood is concave in w, so the fit has a single optimum and "
        "there are no restarts or local minima to report.",
    ], styles))

    story.append(Paragraph("The alternative that was rejected", styles["h3"]))
    story.append(Paragraph(
        "A ranking of ten films could be expanded into all 45 implied pairwise "
        "comparisons and fitted with Bradley-Terry. That was rejected because those "
        "45 comparisons are not independent observations: the likelihood would count "
        "a single answer 45 times and report a spuriously confident estimate. "
        "Plackett-Luce extracts n&minus;1 = 9 genuinely independent choice events "
        "from the same answer.", styles["body"]))

    story.append(Paragraph("Estimation", styles["h3"]))
    story.append(Paragraph(
        "MAP with a Gaussian prior: maximise the log-likelihood less "
        "&alpha;&middot;||w||&sup2;. The prior is load-bearing rather than "
        f"decorative - with {described['n_features']} features and around twenty "
        "answers the likelihood alone is under-determined, and the prior is also "
        "what selects a representative from the unidentifiable directions noted "
        "above. The gradient is analytic and the problem concave, so L-BFGS "
        "converges in a few dozen iterations. The implementation is checked against "
        "a finite-difference gradient, against the closed-form Bradley-Terry "
        "likelihood at n = 2, and by confirming that the probabilities it assigns "
        "to all orderings of a set sum to one.", styles["body"]))

    # --- Task 3 --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Task 3: the user study", styles["h2"]))

    story.append(Paragraph("Research question and hypotheses", styles["h3"]))
    story.append(Paragraph(
        "Both interfaces elicit preferences, but they spend a participant's "
        "attention differently: Design 1 asks many cheap questions, Design 2 asks "
        "few expensive ones. The question is which buys a better estimate of w for "
        "the effort, and at what cost in perceived burden.", styles["body"]))
    story.append(_bullets([
        "<b>H1 (accuracy per answer).</b> A ranking of ten yields a more accurate "
        "w than a single pairwise comparison, because it contributes nine choice "
        "events rather than one.",
        "<b>H2 (accuracy per minute).</b> The advantage shrinks or reverses once "
        "time is accounted for, because ranking ten films takes far longer than one "
        "binary choice. This is the hypothesis the study is really for: H1 is "
        "close to arithmetic, H2 is not.",
        "<b>H3 (workload).</b> Design 2 imposes a higher subjective workload, "
        "measured by NASA-TLX.",
        "<b>H4 (consistency).</b> Rankings of ten contain more internal "
        "inconsistency than pairwise choices, measurable as the rate at which the "
        "fitted model fails to reproduce the participant's own middle placements, "
        "where discrimination is hardest.",
    ], styles))

    story.append(Paragraph("Design", styles["h3"]))
    story.append(Paragraph(
        f"<b>Within-subjects</b>, both designs per participant. Film taste varies "
        f"enormously between people, and a between-subjects design would put that "
        f"variance directly into the comparison; within-subjects makes each person "
        f"their own control. <b>Counterbalanced</b>: participants are assigned "
        f"alternately so that half meet Design 1 first and half Design 2, which "
        f"prevents practice and fatigue from being mistaken for an effect of the "
        f"interface. The implementation assigns the order by alternating on the "
        f"session count rather than by coin flip, which keeps the two arms balanced "
        f"even in a small sample.", styles["body"]))

    story.append(Paragraph(
        f"<b>Blocks.</b> {study.PAIRWISE_TASKS} pairwise comparisons "
        f"(Design 1); {study.RANKING_TASKS} rankings of {study.RANKING_SIZE} films "
        f"(Design 2), which is {study.RANKING_TASKS * (study.RANKING_SIZE - 1)} "
        f"choice events; then a <b>held-out validation block</b> of "
        f"{study.VALIDATION_TASKS} further pairwise comparisons. The blocks are "
        f"deliberately <i>not</i> matched on answer count, because the interesting "
        f"question is what a participant's time buys; time per task is recorded "
        f"throughout.", styles["body"]))

    story.append(Paragraph("The primary measure", styles["h3"]))
    story.append(Paragraph(
        "The validation answers are never fitted. w is estimated from each design's "
        "block separately, and each estimate is scored by how well it predicts the "
        "held-out answers. This is what makes two designs that collect different "
        "numbers of different-shaped answers comparable at all. Chance is 50% for "
        "pairwise validation items, and an all-zero w scores exactly chance, which "
        "is the floor the measure is checked against.", styles["body"]))

    story.append(_table([
        ["Measure", "Type", "How"],
        ["Held-out predictive accuracy", "primary",
         "Share of the validation comparisons the w from each design predicts "
         "correctly."],
        ["Accuracy per minute of elicitation", "primary",
         "The same, divided by the time that design's block took. Tests H2."],
        ["Time per task", "secondary", "Recorded server-side per answer."],
        ["NASA-TLX", "secondary",
         "Six sub-scales after each block. Tests H3."],
        ["Forced-choice preference", "secondary",
         "Which interface the participant would rather use again, plus a free-text "
         "reason."],
        ["Internal consistency", "secondary",
         "Log-likelihood of each participant's own answers under their fitted w. "
         "Tests H4."],
    ], [5.2 * cm, 2.2 * cm, 8.6 * cm]))

    story.append(Paragraph("Participants and recruitment", styles["h3"]))
    story.append(_bullets([
        "<b>Target n = 40</b>, from a power analysis: a paired comparison with "
        "&alpha; = 0.05 and power 0.8 detects a medium within-subject effect "
        "(Cohen's d = 0.5) at n = 34; 40 allows for exclusions.",
        "<b>Recruitment</b> through the university's participant pool and a "
        "crowd-working platform, quota-balanced on age band and gender so the "
        "sample is not exclusively students.",
        "<b>Inclusion</b>: fluent enough in English to read film titles and "
        "genres, and self-reported familiarity with at least some mainstream "
        "cinema; the interface shows only films above "
        f"{described['min_votes']:,} votes to support this.",
        "<b>Exclusion</b> before analysis, on criteria fixed in advance: a median "
        "response time under two seconds per pairwise task, a ranking submitted in "
        "the order presented in every ranking block, or a failed attention check.",
        "<b>Compensation</b> at the local minimum hourly wage, pro-rated to the "
        "expected 20 minutes, paid regardless of exclusion.",
    ], styles))

    story.append(Paragraph("Procedure", styles["h3"]))
    story.append(_bullets([
        "Information sheet and consent; explanation that no personal data beyond "
        "the answers and timings is collected.",
        "Two practice tasks per design, discarded, so the learning curve does not "
        "land in the data.",
        "First block, in the assigned order, then NASA-TLX.",
        "Second block, then NASA-TLX again.",
        "Validation block.",
        "Forced-choice preference question and free-text comment.",
        "Debrief, including the estimated w and the recommendations it produces, "
        "which the interface shows.",
    ], styles))

    story.append(Paragraph("Analysis plan, fixed in advance", styles["h3"]))
    story.append(Paragraph(
        "H1 and H2: paired t-test on held-out accuracy and on accuracy per minute, "
        "with a Wilcoxon signed-rank test as the non-parametric fallback if "
        "normality is rejected. H3: paired test on TLX totals. Order is included as "
        "a between-subjects factor in a mixed-effects model, with participant as a "
        "random intercept, to confirm counterbalancing worked and to absorb any "
        "residual order effect. Effect sizes with 95% confidence intervals are "
        "reported alongside every p-value, and the four hypotheses are corrected "
        "for multiplicity with Holm-Bonferroni.", styles["body"]))

    story.append(Paragraph("Threats to validity", styles["h3"]))
    story.append(_table([
        ["Threat", "Mitigation"],
        ["The two designs differ in answer count as well as shape, so an effect "
         "could be either.",
         "Report accuracy per answer and per minute separately, and treat the "
         "per-minute result as the primary one."],
        ["Films drawn uniformly at random are often unknown to the participant.",
         "Restrict the catalogue by vote count; record 'I don't know either film' "
         "as an explicit option so ignorance is not recorded as preference."],
        ["The validation block itself uses Design 1's format, which may favour "
         "Design 1.",
         "A limitation to state openly. The alternative - validating with rankings "
         "- would simply reverse the bias. A fuller study would split the "
         "validation block between both formats and report each."],
        ["Preferences may drift over a twenty-minute session.",
         "Counterbalancing distributes drift across designs; block order is "
         "included in the model."],
        ["The feature space may be too coarse to express real taste, capping "
         "accuracy for both designs.",
         "Report the ceiling: the accuracy reached by a w fitted on all of a "
         "participant's answers together, which bounds what either design could "
         "have achieved."],
    ], [7 * cm, 9 * cm]))

    story.append(Paragraph("Ethics and data protection", styles["h3"]))
    story.append(Paragraph(
        "No personal data is collected: a session is identified by a random token, "
        "and the stored rows are film ids, orderings and timings. Consent is "
        "obtained before the first task and withdrawal is possible at any point, "
        "with the session's rows deleted on request. Ethics approval would be "
        "sought from the university committee before recruitment; the study poses "
        "no foreseeable risk beyond mild boredom.", styles["body"]))

    story.append(Paragraph("4. Task 4: the implemented interface", styles["h2"]))
    story.append(Paragraph(
        f"The interface implements the protocol above and is reachable from the "
        f"project landing page. It runs the {study.total_tasks()} tasks in the "
        f"counterbalanced order, validates every ranking server-side as a genuine "
        f"permutation, records the time each answer took, and stores every response "
        f"in the database rather than in the browser session, so a run that is "
        f"interrupted is not lost. At the end it fits w from each design, scores "
        f"both against the held-out block, and shows the participant what was "
        f"inferred about their taste along with the films it recommends - which "
        f"doubles as the debrief. All collected responses can be exported as CSV "
        f"from the landing page.", styles["body"]))

    document.build(story)
    return buffer.getvalue()
