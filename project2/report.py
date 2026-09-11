"""The downloadable PDF report for Project 2.

The report is generated from the *current* state of the interface, so it always
describes the model the user is actually looking at: the selected model class, the
value of lambda, the resulting complexity, and the figures for that model. It
covers the design choices the task sheet asks about and the experiments behind
them.
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

from . import learning
from .penguins import NUMERIC_FEATURES, readable

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#cbd5e1")
BAND = colors.HexColor("#f1f5f9")


def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=20, leading=24, textColor=INK, alignment=TA_LEFT,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontName="Helvetica", fontSize=10.5, leading=14,
                                   textColor=MUTED, spaceAfter=14),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=13, leading=16, textColor=INK,
                             spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Helvetica-Bold",
                             fontSize=11, leading=14, textColor=INK,
                             spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica",
                               fontSize=9.5, leading=13.5, textColor=INK,
                               spaceAfter=6),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
                                  fontName="Helvetica-Oblique", fontSize=8.5,
                                  leading=11, textColor=MUTED, spaceAfter=10),
    }
    return styles


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
    # repeatRows keeps the header on every page: a table that breaks across
    # a page boundary is unreadable without it.
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
    """An image scaled to the text width, if it exists on disk."""
    if not path or not os.path.exists(path):
        return None
    from PIL import Image as PILImage           # reportlab ships with Pillow
    with PILImage.open(path) as handle:
        ratio = handle.height / handle.width
    return Image(path, width=width, height=width * ratio)


def build(context):
    """Render the report and return the PDF as bytes.

    `context` is assembled by the view and carries the current selection, the
    fitted family, and paths to the figures already generated for the page.
    """
    styles = _styles()
    story = []
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="HCAI Project 2 - Explainability",
        author=", ".join(student["name"] for student in STUDENTS),
    )

    model = context["model"]
    family = context["family"]
    lam = context["lam"]

    # --- cover ---------------------------------------------------------
    story.append(Paragraph("Project 2: Explainability", styles["title"]))
    story.append(Paragraph(
        "Human-Centric Artificial Intelligence &middot; Palmer Penguins &middot; "
        f"generated {date.today().isoformat()}", styles["subtitle"]))

    story.append(_table(
        [["Group member", "Matriculation"]] +
        [[student["name"], student["matriculation"]] for student in STUDENTS],
        [9 * cm, 5 * cm]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"This report describes the model currently selected in the interface: a "
        f"<b>{learning.MODEL_LABELS[model.model_class]}</b> at "
        f"<b>&lambda; = {lam:g}</b>, which has "
        f"<b>{model.omega} {model.omega_label}</b> and a test accuracy of "
        f"<b>{model.accuracy_test:.3f}</b>.", styles["body"]))

    # --- data ----------------------------------------------------------
    story.append(Paragraph("1. Data and preprocessing", styles["h2"]))
    story.append(Paragraph(
        f"The Palmer Penguins dataset has {context['total_rows']} rows and eight "
        f"columns. The target is <b>species</b> (Adelie, Chinstrap, Gentoo). Four "
        f"features are measurements (bill length and depth, flipper length, body "
        f"mass) and three are labels (island, sex, year). <b>year</b> is stored as "
        f"a number but only takes three values, so it is treated as a label; "
        f"leaving it numeric would let a model read 2007 to 2009 as a quantity.",
        styles["body"]))
    story.append(Paragraph(
        f"{context['dropped_rows']} rows are missing a measurement or a sex and are "
        f"dropped rather than imputed, leaving {context['rows']}. With so few "
        f"affected rows, dropping keeps every downstream number - counterfactual "
        f"distances, ALE bin averages - computed on values that were really "
        f"observed. Measurements are standardised for logistic regression (it is "
        f"scale-sensitive) and left alone for the tree (it is not); labels are "
        f"one-hot encoded. Both steps sit inside the pipeline, so they are refitted "
        f"on the training split alone and nothing about the test set leaks in.",
        styles["body"]))
    story.append(Paragraph(
        f"The split is {int(round((1 - context['test_size']) * 100))}/"
        f"{int(round(context['test_size'] * 100))} train/test, stratified by "
        f"species, with seed {context['seed']}.", styles["body"]))

    # --- tasks 1 to 3 ---------------------------------------------------
    story.append(Paragraph("2. Interpretability and model complexity "
                           "(Tasks 1 to 3)", styles["h2"]))
    story.append(Paragraph(
        "The objective from the lecture is minimised over a family of models:",
        styles["body"]))
    story.append(Paragraph(
        "<b>argmin<sub>f</sub> (1/n) &Sigma; loss(f(x<sub>i</sub>), y<sub>i</sub>) "
        "+ &lambda; &Omega;(f)</b>, handled in the interface as the equivalent "
        "maximisation <b>acc<sub>test</sub> &minus; &lambda; &Omega;(f)</b>.",
        styles["body"]))
    story.append(Paragraph(
        "&lambda; is not the fitting-time regularisation parameter. Each family is "
        "fitted once per fitting-time setting, giving a menu of models; &lambda; "
        "then selects one off that menu. Moving the slider therefore never refits "
        "anything, which is also why it responds instantly.", styles["body"]))

    tree = context["default_tree"]
    story.append(Paragraph("Task 1: one unconstrained tree", styles["h3"]))
    story.append(Paragraph(
        f"Before any trade-off, Task 1 asks only for a decision tree with its "
        f"test accuracy and its number of leaves. That is a plain "
        f"<b>DecisionTreeClassifier()</b> on the same split, with nothing "
        f"constrained, grown until its leaves are pure: it reaches "
        f"<b>{tree.accuracy_test:.3f}</b> on the test set with "
        f"<b>{tree.omega} leaves</b> at depth {tree.extra['depth']}, against "
        f"{tree.accuracy_train:.3f} on its own training rows -- exact by "
        f"construction, since growth only stops when every leaf is pure. It is "
        f"fitted separately from the Task 2 family, every member of which is "
        f"capped by max_leaf_nodes and so is not this tree.", styles["body"]))
    story.append(_picture(context["default_tree_path"]))

    story.append(Paragraph("Choice of &Omega;", styles["h3"]))
    story.append(Paragraph(
        "For the <b>decision tree</b>, &Omega; is the number of leaves, as the task "
        "sheet specifies; <b>max_leaf_nodes</b> is the fitting-time control. Note "
        "that the realised leaf count can fall below the cap, so the achieved "
        "&Omega; is read back off the fitted tree rather than assumed.",
        styles["body"]))
    story.append(Paragraph(
        "For <b>logistic regression</b>, &Omega; is the <b>number of non-zero "
        "coefficients</b>, fitted with an L1 penalty of strength C. This is the "
        "honest analogue of counting leaves: both count the pieces of the model a "
        "reader must hold in their head, and L1 is what actually drives "
        "coefficients to zero so that the count can fall. A norm such as "
        "||w||<sub>2</sub> also measures size, but it shrinks coefficients smoothly "
        "without ever removing one, so no term ever disappears and the number would "
        "not correspond to anything the reader could skip.", styles["body"]))

    story.append(Paragraph(
        f"Family fitted for the current model class "
        f"({learning.MODEL_LABELS[model.model_class]})", styles["h3"]))

    setting_name = "max_leaf_nodes" if model.model_class == learning.TREE else "C"
    rows = [[setting_name, f"Ω ({model.omega_label})", "Train acc.",
             "Test acc.", f"Objective (λ={lam:g})"]]
    highlight = None
    for index, entry in enumerate(context["trade_off"]):
        rows.append([
            f"{entry['setting']:g}" if isinstance(entry["setting"], float)
            else str(entry["setting"]),
            str(entry["omega"]),
            f"{entry['accuracy_train']:.3f}",
            f"{entry['accuracy_test']:.3f}",
            f"{entry['objective']:.3f}",
        ])
        if entry["selected"]:
            highlight = index
    story.append(_table(rows, [3 * cm, 3.4 * cm, 2.6 * cm, 2.6 * cm, 4 * cm],
                        highlight=highlight))
    story.append(Paragraph(
        "The highlighted row is the maximiser at the current &lambda;. Ties are "
        "broken towards the simpler model.", styles["caption"]))

    if context["switch_points"]:
        story.append(Paragraph("Where the slider changes its mind", styles["h3"]))
        rows = [["λ at least", setting_name, "Ω", "Test acc."]]
        for point in context["switch_points"]:
            rows.append([
                f"{point['lam']:.4f}",
                f"{point['setting']:g}" if isinstance(point["setting"], float)
                else str(point["setting"]),
                str(point["omega"]),
                f"{point['accuracy_test']:.3f}",
            ])
        story.append(_table(rows, [3.5 * cm, 3.5 * cm, 3 * cm, 3 * cm]))
        story.append(Paragraph(
            "Accuracy is bought with complexity, and these are the prices at which "
            "the interface stops paying.", styles["caption"]))

    picture = _picture(context.get("trade_off_path"))
    if picture:
        story.append(picture)
        story.append(Paragraph(
            "Test accuracy and the penalised objective against complexity. The gap "
            "between the curves is exactly &lambda;&Omega;.", styles["caption"]))

    story.append(PageBreak())

    story.append(Paragraph("The selected model", styles["h2"]))
    if model.model_class == learning.TREE:
        story.append(Paragraph(
            f"A decision tree with {model.omega} leaves and depth "
            f"{model.extra.get('depth', '?')}, test accuracy "
            f"{model.accuracy_test:.3f} (training {model.accuracy_train:.3f}).",
            styles["body"]))
    else:
        story.append(Paragraph(
            f"Logistic regression at C = {model.setting:g} with {model.omega} "
            f"non-zero coefficients across "
            f"{model.extra.get('features_used', '?')} encoded features, test "
            f"accuracy {model.accuracy_test:.3f} (training "
            f"{model.accuracy_train:.3f}).", styles["body"]))

    picture = _picture(context.get("model_path"))
    if picture:
        story.append(picture)
        story.append(Paragraph(
            "The fitted model as the interface shows it." if
            model.model_class == learning.TREE else
            "Non-zero coefficients on standardised features, grouped by species.",
            styles["caption"]))

    # --- task 4 ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Counterfactual explanations (Task 4)", styles["h2"]))
    story.append(Paragraph(
        "For a chosen penguin x and a desired species, N points are sampled "
        "locally around x, those the selected model predicts as the desired "
        "species are kept, and they are ranked by MAD-weighted "
        "L<sup>1</sup> distance to x. The best k are shown.", styles["body"]))

    story.append(Paragraph("Noising each kind of feature", styles["h3"]))
    story.append(Paragraph(
        "The task sheet asks what to do about data that is not decimal, and the two "
        "kinds genuinely need different treatment. <b>Measurements</b> get Gaussian "
        "noise scaled by the feature's own MAD, so a feature that naturally varies "
        "little is nudged little; samples are clipped to the observed range, "
        "because a counterfactual with a negative body mass would be arithmetic "
        "rather than an explanation. <b>Labels</b> cannot be nudged at all - there "
        "is no midpoint between two islands - so each is instead resampled with a "
        "small probability, uniformly among the other observed values. Keeping that "
        "probability low means a counterfactual differs from x in only a few "
        "labels, which is what makes it readable.", styles["body"]))

    story.append(Paragraph("Distance", styles["h3"]))
    story.append(Paragraph(
        "MAD-weighted L<sup>1</sup> over the measurements, plus a flat cost of 1 for "
        "each label that differs. Dividing by the MAD puts every measurement on a "
        "comparable footing; the flat cost prices a change of island at about one "
        "typical measurement step, so the ranking does not quietly prefer moving a "
        "penguin between islands over moving its bill by a millimetre.",
        styles["body"]))

    story.append(Paragraph("If nothing is found", styles["h3"]))
    story.append(Paragraph(
        "The search widens in stages - more samples and larger variance each time - "
        "as the sheet suggests. Some requests have no answer at all, and the "
        "interface says so rather than inventing one.", styles["body"]))

    if context.get("counterfactuals"):
        story.append(Paragraph("Worked example", styles["h3"]))
        story.append(Paragraph(context["counterfactual_summary"], styles["body"]))
        rows = [["Distance", "Confidence", "What would have to change"]]
        for row in context["counterfactuals"]:
            changes = "; ".join(
                f"{change['label']}: {change['before']} &rarr; {change['after']}"
                for change in row["changes"]) or "nothing"
            rows.append([f"{row['distance']:.2f}", f"{row['confidence']:.2f}",
                         Paragraph(changes, _styles()["body"])])
        story.append(_table(rows, [2.2 * cm, 2.4 * cm, 11.4 * cm]))

    # --- task 5 ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4. Feature effect plots (Task 5)", styles["h2"]))
    story.append(Paragraph(
        "Both curves are computed here from scratch; no explainability library is "
        "used.", styles["body"]))

    story.append(Paragraph("PDP", styles["h3"]))
    story.append(Paragraph(
        "For a feature j and a grid value v, every row is copied with "
        "x<sub>j</sub> replaced by v and the predicted probabilities are averaged. "
        "That average runs over the marginal distribution of the other features, "
        "which is what makes a PDP easy to read and also what makes it mislead when "
        "features are correlated: it evaluates the model on combinations that never "
        "occur, such as a short bill on a heavy Gentoo.", styles["body"]))

    story.append(Paragraph("ALE", styles["h3"]))
    story.append(Paragraph(
        "ALE avoids inventing rows. The feature range is cut into bins at its own "
        "quantiles; inside each bin only the rows that actually fall there are used; "
        "their local change in prediction across the bin is averaged, and those "
        "averages are accumulated and then centred to average zero. The curve "
        "therefore reads as a deviation from the mean prediction.", styles["body"]))

    story.append(Paragraph(
        "For which model can the derivative be computed exactly?", styles["h3"]))
    story.append(Paragraph(
        "<b>Logistic regression: exactly.</b> The prediction is a softmax of a "
        "linear function, so with &eta;<sub>c</sub> = w<sub>c</sub>&middot;z + "
        "b<sub>c</sub> over standardised features z, "
        "&part;&eta;<sub>c</sub>/&part;x<sub>j</sub> = "
        "w<sub>c,j</sub>/scale<sub>j</sub> and "
        "&part;p<sub>c</sub>/&part;x<sub>j</sub> = p<sub>c</sub>("
        "&part;&eta;<sub>c</sub>/&part;x<sub>j</sub> &minus; &Sigma;<sub>k</sub> "
        "p<sub>k</sub> &part;&eta;<sub>k</sub>/&part;x<sub>j</sub>). The chain rule "
        "through the scaler is what the 1/scale<sub>j</sub> term is. The "
        "implementation is checked against a central finite difference of "
        "predict_proba and agrees to about 1e-10.", styles["body"]))
    story.append(Paragraph(
        "<b>Decision tree: not at all, so it must be discretised.</b> A tree is "
        "piecewise constant: its derivative is zero everywhere inside a leaf and "
        "undefined on the split boundaries, so it carries no information about the "
        "effect. Measured on the fitted tree, 100% of the numerical derivatives come "
        "out exactly zero. The only workable route is the discretised one, taking a "
        "finite difference across each bin, which is what the app does for the tree. "
        "As a cross-check, running the discretised route on the logistic model as "
        "well reproduces its exact curve to within a few thousandths.",
        styles["body"]))

    for entry in context.get("effects", []):
        picture = _picture(entry["path"])
        if picture:
            story.append(picture)
            story.append(Paragraph(entry["caption"], styles["caption"]))

    story.append(Paragraph("5. What the plots say", styles["h2"]))
    story.append(Paragraph(context["reading"], styles["body"]))

    document.build(story)
    return buffer.getvalue()
