"""The downloadable PDF report for Project 3.

The task sheet asks for a report describing the experiments, justifying the design
choices, and reporting the results in detail. It is generated from the cached
experiment, so every number in it is the number the interface is showing.
"""

import io
import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from home.views import STUDENTS

from .active import STRATEGY_LABELS
from .data import CLASS_NAMES

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


def _table(rows, widths, highlight=None):
    table = Table(_wrap_long_cells(rows, widths), colWidths=widths,
                  hAlign="LEFT", repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight is not None:
        style.append(("BACKGROUND", (0, highlight + 1), (-1, highlight + 1), BAND))
        style.append(("FONT", (0, highlight + 1), (-1, highlight + 1),
                      "Helvetica-Bold", 8.5))
    table.setStyle(TableStyle(style))
    return table


def _picture(path, width=16 * cm):
    if not path or not os.path.exists(path):
        return None
    from PIL import Image as PILImage
    with PILImage.open(path) as handle:
        ratio = handle.height / handle.width
    return Image(path, width=width, height=width * ratio)


def build(bundle, figures=None):
    """Render the report for a cached experiment. Returns PDF bytes."""
    styles = _styles()
    figures = figures or {}
    story = []
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="HCAI Project 3 - Active Learning for Learning-to-Defer",
        author=", ".join(student["name"] for student in STUDENTS),
    )

    dataset = bundle["dataset"]
    baseline = bundle["baseline"]
    system = bundle["system"]
    expert = bundle["expert"]
    policies = bundle["policies"]
    quality = bundle["competence_quality"]

    story.append(Paragraph("Project 3: Active Learning for Learning-to-Defer",
                           styles["title"]))
    story.append(Paragraph(
        f"Human-Centric Artificial Intelligence &middot; AG News &middot; "
        f"generated {date.today().isoformat()}", styles["subtitle"]))

    story.append(_table(
        [["Group member", "Matriculation"]] +
        [[s["name"], s["matriculation"]] for s in STUDENTS],
        [9 * cm, 5 * cm]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"A classifier reaching <b>{baseline['accuracy']:.4f}</b> test accuracy is "
        f"paired with a simulated expert reaching <b>{expert['accuracy']:.4f}</b>. "
        f"Letting the learned competence rule decide which of the two answers each "
        f"article raises the team to "
        f"<b>{policies['competence_zero_margin']['team_accuracy']:.4f}</b>, above the "
        f"classifier working alone. That rule is untuned: it defers whenever the "
        f"expert's estimated competence exceeds the classifier's, which is the "
        f"comparison the theory prescribes, with no threshold fitted after the fact. "
        f"The experiment was run in "
        f"{bundle['build_seconds']:.0f} seconds and cached; every figure and number "
        f"below comes from that run ({bundle['built_at']}).", styles["body"]))

    # --- data ----------------------------------------------------------
    story.append(Paragraph("1. Data and experimental design", styles["h2"]))
    story.append(Paragraph(
        f"AG News: {dataset['n_train']:,} training and {dataset['n_test']:,} test "
        f"articles, balanced over four topics ({', '.join(CLASS_NAMES)}), median "
        f"length {dataset['median_characters']} characters. The test set is used for "
        f"evaluation only. The one place it is swept is to trace the trade-off "
        f"curves below, and the operating points those sweeps single out are "
        f"labelled as chosen in hindsight; no headline figure is tuned on them.",
        styles["body"]))

    split = bundle["split"]
    story.append(Paragraph(
        f"The training set is divided once: {split['fit_rows']:,} rows fit the "
        f"classifier and {split['meta_rows']:,} are held back. Everything that "
        f"has to estimate <i>how often a predictor is right</i> is fitted on the "
        f"held-back rows. This is the single most important design decision in the "
        f"project: on its own training rows the classifier is right almost always, "
        f"so a competence model fitted there concludes it is never wrong and the "
        f"system never defers. Task 1 additionally reports a classifier trained on "
        f"all {dataset['n_train']:,} labels, as asked; the deferral system uses the "
        f"80% classifier, and both numbers are given so the gap is visible.",
        styles["body"]))

    # --- task 1 --------------------------------------------------------
    story.append(Paragraph("2. Task 1: the classifier", styles["h2"]))
    story.append(Paragraph(
        "TF-IDF over word unigrams and bigrams, then multinomial logistic "
        "regression. Topic in a news snippet is carried almost entirely by "
        "vocabulary, so a linear model over n-grams is close to the right "
        "inductive bias rather than a compromise, and it fits in under a minute on "
        "the full training set. Logistic regression is preferred over the slightly "
        "more accurate LinearSVC (measured 0.9271 against 0.9245 on identical "
        "features, reproducible with <b>python manage.py benchmark_classifier</b>) "
        "because every later task needs a <i>probability</i>: the "
        "deferral rule compares the classifier's confidence with the expert's "
        "estimated competence, and an SVM margin is not on that scale.",
        styles["body"]))

    story.append(_table([
        ["Model", "Trained on", "Test accuracy", "Macro F1"],
        ["Baseline (Task 1)", f"{dataset['n_train']:,} rows",
         f"{baseline['accuracy']:.4f}", f"{baseline['macro_f1']:.4f}"],
        ["Deferral system's classifier", f"{split['fit_rows']:,} rows",
         f"{system['accuracy']:.4f}", f"{system['macro_f1']:.4f}"],
    ], [6 * cm, 3.4 * cm, 3.4 * cm, 3 * cm]))

    story.append(_table(
        [["Topic", "Test articles", "Classifier recall"]] +
        [[name, f"{system['per_class'][name]['n']:,}",
          f"{system['per_class'][name]['recall']:.4f}"] for name in CLASS_NAMES],
        [6 * cm, 4 * cm, 4 * cm]))
    story.append(Paragraph("Per-topic recall of the deferral system's classifier.",
                           styles["caption"]))

    picture = _picture(figures.get("confusion"))
    if picture:
        story.append(picture)
        story.append(Paragraph("Where the classifier confuses topics.",
                               styles["caption"]))

    # --- task 2 --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Task 2: the simulated expert", styles["h2"]))
    story.append(Paragraph(
        f"<b>{expert['name']}.</b> {expert['description']}", styles["body"]))
    story.append(Paragraph(
        f"Its region covers {expert['region_share']:.1%} of the test set. Inside "
        f"it the expert is right {expert['accuracy_in_region']:.1%} of the time and "
        f"outside it {expert['accuracy_out_of_region']:.1%}, giving "
        f"{expert['accuracy']:.4f} overall - well below the classifier. That is the "
        f"point: an expert who is uniformly better would make the deferral question "
        f"trivial. Being better only <i>somewhere</i> is what forces the system to "
        f"learn where.", styles["body"]))
    story.append(Paragraph(
        "The expert is deterministic given a seed: the same article always gets the "
        "same answer. This matters because the active-learning loop can select a "
        "point more than once and must not receive a fresh coin flip each time.",
        styles["body"]))

    story.append(_table(
        [["Topic", "Expert accuracy", "Classifier recall",
          "Share inside the beat"]] +
        [[name,
          f"{expert['per_class'][name]['accuracy']:.4f}",
          f"{system['per_class'][name]['recall']:.4f}",
          f"{expert['per_class'][name]['in_region_share']:.0%}"]
         for name in CLASS_NAMES],
        [4.2 * cm, 4 * cm, 4 * cm, 3.8 * cm]))
    story.append(Paragraph(
        "Strengths and weaknesses by topic. Note that the classifier already beats "
        "the expert on some topics the expert is strong at, so deferring on the "
        "whole beat would lose accuracy.", styles["caption"]))

    picture = _picture(figures.get("expert_profile"))
    if picture:
        story.append(picture)
        story.append(Paragraph("Classifier against expert, topic by topic.",
                               styles["caption"]))

    # --- task 3 --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4. Task 3: learning to defer", styles["h2"]))
    story.append(Paragraph(
        "Deferring is worth it exactly when the expert is likelier to be right than "
        "the classifier, so the rule compares two estimated competences: "
        "<b>defer(x) iff P(expert correct | x) &gt; P(classifier correct | x)</b>.",
        styles["body"]))

    story.append(Paragraph("Estimating the two competences", styles["h3"]))
    story.append(Paragraph(
        f"P(expert correct | x) is a logistic model on the article text, fitted on "
        f"the held-back rows, reaching AUC "
        f"{quality['expert_competence_auc']:.3f} on the test set. For the "
        f"classifier side two estimates were fitted on the same rows and compared: "
        f"Platt scaling of the classifier's own confidence (AUC "
        f"{quality['calibrated_confidence_auc']:.3f}) and a text model of its "
        f"correctness (AUC {quality['classifier_text_model_auc']:.3f}). The "
        f"calibrated confidence is clearly the better estimate and is what the rule "
        f"uses. The raw, uncalibrated confidence has AUC "
        f"{quality['raw_confidence_auc']:.3f}: it ranks well but is not on the "
        f"probability scale the comparison needs, which is why it is calibrated "
        f"rather than used directly.", styles["body"]))
    story.append(Paragraph(
        f"Using the weaker text model instead costs almost the whole benefit - the "
        f"team falls to "
        f"{policies['competence_text_zero_margin']['team_accuracy']:.4f}, barely "
        f"above the classifier alone. Which quantity is used to represent the "
        f"classifier's competence matters more than the form of the rule.",
        styles["body"]))

    story.append(Paragraph("Results, and the quality of the decisions",
                           styles["h3"]))
    story.append(Paragraph(
        "Team accuracy on its own hides whether the deferrals were the right ones: "
        "a rule that defers everything simply inherits the expert's accuracy "
        "without deciding anything. Each rule is therefore also scored against the "
        "oracle decision - defer iff the expert is right here and the classifier "
        "wrong - and by what the deferrals bought: articles rescued from a wrong "
        "classifier answer, against articles spoiled by handing over one that was "
        "already right.", styles["body"]))

    order = [
        ("Classifier alone", policies["classifier_only"]),
        ("Expert alone", policies["expert_only"]),
        ("Confidence threshold, tuned in hindsight", policies["confidence_best"]),
        ("Competence rule, margin 0", policies["competence_zero_margin"]),
        ("Competence rule, margin tuned in hindsight", policies["competence_best"]),
        ("Competence rule, text model", policies["competence_text_zero_margin"]),
        ("Oracle ceiling", policies["oracle"]),
    ]
    rows = [["Policy", "Team acc.", "Deferred", "Defer F1", "Rescued", "Spoiled", "Net"]]
    for label, entry in order:
        rows.append([
            label,
            f"{entry['team_accuracy']:.4f}",
            f"{entry['deferral_rate']:.1%}",
            f"{entry['deferral_f1']:.3f}",
            f"{entry['rescued']:,}",
            f"{entry['spoiled']:,}",
            f"{entry['net_gain']:+,}",
        ])
    story.append(_table(rows, [5.2 * cm, 2.1 * cm, 1.9 * cm, 1.8 * cm,
                               1.7 * cm, 1.7 * cm, 1.6 * cm]))
    story.append(Paragraph(
        f"Of {policies['oracle']['oracle_opportunities']:,} articles where "
        f"deferring would have helped, the competence rule finds "
        f"{policies['competence_zero_margin']['rescued']:,} while spoiling "
        f"{policies['competence_zero_margin']['spoiled']:,}. The oracle row is the "
        f"ceiling any rule could reach with this classifier and this expert.",
        styles["caption"]))

    picture = _picture(figures.get("accuracy"))
    if picture:
        story.append(picture)
        story.append(Paragraph("Everyone's test accuracy on one axis.",
                               styles["caption"]))

    picture = _picture(figures.get("deferral"))
    if picture:
        story.append(picture)
        story.append(Paragraph(
            "What each rule buys per unit of expert effort. A rule is better when "
            "its curve sits higher at the same deferral rate.", styles["caption"]))

    # --- task 4 --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("5. Task 4: active learning for expert competence",
                           styles["h2"]))
    settings_used = bundle["active_settings"]
    story.append(Paragraph(
        f"Now no expert label exists at training time. The classifier and its "
        f"calibrated confidence still can be fitted, since neither needs the "
        f"expert; only P(expert correct | x) has to be bought. The loop starts with "
        f"nothing, queries {settings_used['batch']} points per round for "
        f"{settings_used['rounds']} rounds from a pool of "
        f"{settings_used['pool_size']:,} unlabelled articles, refits the expert "
        f"model after each round, and is evaluated on the test set.",
        styles["body"]))

    story.append(Paragraph("The strategy, and why", styles["h3"]))
    story.append(Paragraph(
        "The system does not need a good model of the expert everywhere; it needs "
        "one binary decision to come out right. The informative labels are "
        "therefore the ones that pin down the boundary where the two competences "
        "cross, since a single label there can flip the decision. That is the "
        "chosen strategy - but on its own it fails, and the failure is measurable "
        "rather than hypothetical.", styles["body"]))

    story.append(Paragraph("The sampling-bias failure, measured", styles["h3"]))
    summary = bundle["active_summary"]
    rows = [["Strategy", "Expert model AUC", "Est. base rate", "Team acc.",
             "Defer F1", "Labels to target"]]
    for strategy, entry in summary.items():
        rows.append([
            entry["label"],
            f"{entry['expert_model_auc']:.3f}",
            f"{entry['expert_base_rate']:.3f}",
            f"{entry['team_accuracy']:.4f}",
            f"{entry['deferral_f1']:.3f}",
            str(entry["queries_to_target"]) if entry["queries_to_target"]
            else "not reached",
        ])
    story.append(_table(rows, [5 * cm, 2.6 * cm, 2.4 * cm, 2 * cm, 1.8 * cm,
                               2.2 * cm]))
    story.append(Paragraph(
        f"After {summary['random']['queries']:,} labels. The target is the team "
        f"accuracy the fully-supervised rule of Task 3 reached "
        f"({bundle['active_target']:.4f}).", styles["caption"]))

    story.append(Paragraph(
        f"Querying only near the boundary makes the sample unrepresentative of the "
        f"data the model is later asked about, and the fitted competence model does "
        f"not transfer: its AUC reaches only "
        f"{summary['decision_boundary']['expert_model_auc']:.3f} against "
        f"{summary['random']['expert_model_auc']:.3f} for uniform sampling. Pure "
        f"expert-model uncertainty is worse still - it drifts into a skewed corner "
        f"of the pool and its estimate of the expert's own base rate collapses to "
        f"{summary['competence_uncertainty']['expert_base_rate']:.3f}, against "
        f"{summary['random']['expert_base_rate']:.3f} from a uniform sample of the "
        f"same size - the nearest thing to an unbiased estimate available here, "
        f"not the population value. This is the classic sampling-bias failure of "
        f"greedy active learning.",
        styles["body"]))
    story.append(Paragraph(
        f"The strategy actually proposed, <b>boundary sampling with exploration</b>, "
        f"splits every batch: half the labels uniform, to keep the sample "
        f"representative and the base rate honest, and half on the boundary. That "
        f"repairs the mechanism - AUC recovers to "
        f"{summary['hybrid_boundary']['expert_model_auc']:.3f} and the base-rate "
        f"estimate stays sound.", styles["body"]))

    story.append(Paragraph("The honest conclusion", styles["h3"]))

    # `queries_to_reach` returns None for a strategy that never gets there, so
    # neither number can be dropped into the sentence unchecked.
    def _budget(strategy):
        reached = summary[strategy]["queries_to_target"]
        return f"{reached} labels" if reached else "not within the budget tried"

    story.append(Paragraph(
        f"On this problem active learning does not beat uniform sampling on team "
        f"accuracy. To reach the fully-supervised target, uniform sampling needs "
        f"{_budget('random')} and the hybrid {_budget('hybrid_boundary')}. Two "
        f"reasons, both "
        f"visible in the numbers. First, this expert's competence is defined by "
        f"topic and topic is easy to read off the text, so a few hundred labels "
        f"drawn anywhere already characterise it. Second, the deferral decision "
        f"only changes the answer on about "
        f"{policies['competence_zero_margin']['deferral_rate']:.0%} of articles, so "
        f"team accuracy is insensitive to refinements in the competence model. "
        f"Where the targeted strategies do show an edge is in the quality of the "
        f"deferral decisions at small budgets rather than in accuracy.",
        styles["body"]))
    story.append(Paragraph(
        "The practical recommendation that follows is to use uniform sampling "
        "unless the expert's competence is concentrated in a narrow region that "
        "random querying would rarely hit; and if a targeted strategy is used, to "
        "keep an exploration share in every batch, because the unmixed version is "
        "measurably worse than doing nothing clever at all.", styles["body"]))

    for key, caption in (
        ("active", "Team accuracy against the number of expert labels bought."),
        ("active_auc", "How well each strategy's expert-competence model "
                       "generalises. This is where the strategies genuinely differ."),
    ):
        picture = _picture(figures.get(key))
        if picture:
            story.append(picture)
            story.append(Paragraph(caption, styles["caption"]))

    story.append(Paragraph("6. Task 5: a human in the loop", styles["h2"]))
    story.append(Paragraph(
        "The optional extension is implemented in the interface: the "
        "“Be the expert” page shows the articles the system chose to "
        "defer, takes the reader's own label, and scores them against the "
        "simulated expert on the same articles. It is the same query loop with a "
        "person in place of the simulation.", styles["body"]))

    document.build(story)
    return buffer.getvalue()
