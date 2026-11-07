"""Generate the SSRN-ready research paper PDF from live pipeline outputs.

Builds two documents with ReportLab:

* ``outputs/paper/var_confidence_audit.pdf`` - the full paper (title page,
  abstract, introduction, data, methodology, results, conclusion,
  references) with page numbers and a running header.
* ``outputs/paper/figures_appendix.pdf`` - every figure in
  ``outputs/figures`` at full width.

All tables and headline numbers are pulled from ``data/processed/`` at
build time, so the paper always reflects the latest pipeline run.

Run as a script::

    python paper/generate_paper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from paper.sections import (  # noqa: E402
    abstract,
    conclusion,
    data,
    introduction,
    methodology,
    results,
)
from paper.sections.common import BODY_FONT, ITALIC_FONT, get_styles  # noqa: E402

logger = config.configure_logging(__name__)

TITLE = (
    "Do Banks' Disclosed VaR Figures Hold Up? "
    "An Empirical Audit of 10-Q Backtesting Disclosures"
)
RUNNING_HEADER = "Do Banks' Disclosed VaR Figures Hold Up?"

REFERENCES = [
    "Basel Committee on Banking Supervision. (1996). <i>Supervisory framework "
    "for the use of “backtesting” in conjunction with the internal models "
    "approach to market risk capital requirements</i>. Bank for International "
    "Settlements.",
    "Basel Committee on Banking Supervision. (2019). <i>Minimum capital "
    "requirements for market risk</i> (MAR). Bank for International "
    "Settlements.",
    "Berkowitz, J., &amp; O'Brien, J. (2002). How accurate are value-at-risk "
    "models at commercial banks? <i>The Journal of Finance, 57</i>(3), "
    "1093&ndash;1111.",
    "Campbell, S. D. (2006). A review of backtesting and backtesting "
    "procedures. <i>The Journal of Risk, 9</i>(2), 1&ndash;17.",
    "Christoffersen, P. F. (1998). Evaluating interval forecasts. "
    "<i>International Economic Review, 39</i>(4), 841&ndash;862.",
    "Engle, R. F., &amp; Manganelli, S. (2004). CAViaR: Conditional "
    "autoregressive value at risk by regression quantiles. <i>Journal of "
    "Business &amp; Economic Statistics, 22</i>(4), 367&ndash;381.",
    "Jorion, P. (2007). <i>Value at risk: The new benchmark for managing "
    "financial risk</i> (3rd ed.). McGraw-Hill.",
    "J.P. Morgan/Reuters. (1996). <i>RiskMetrics&trade; — Technical document</i> "
    "(4th ed.). Morgan Guaranty Trust Company.",
    "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk "
    "measurement models. <i>The Journal of Derivatives, 3</i>(2), 73&ndash;84.",
    "Lopez, J. A. (1999). Methods for evaluating value-at-risk estimates. "
    "<i>Federal Reserve Bank of San Francisco Economic Review, 2</i>, 3&ndash;17.",
    "O'Brien, J., &amp; Szerszeń, P. J. (2017). An evaluation of bank "
    "measures for market risk before, during and after the financial crisis. "
    "<i>Journal of Banking &amp; Finance, 80</i>, 215&ndash;234.",
    "P&eacute;rignon, C., &amp; Smith, D. R. (2010). The level and quality of "
    "Value-at-Risk disclosure by commercial banks. <i>Journal of Banking "
    "&amp; Finance, 34</i>(2), 362&ndash;377.",
]


def _on_page(canvas, doc) -> None:
    """Draw the running header and page number on each page."""
    canvas.saveState()
    width, height = letter
    if doc.page > 1:
        canvas.setFont(ITALIC_FONT, 9)
        canvas.drawString(inch, height - 0.55 * inch, RUNNING_HEADER)
        canvas.line(inch, height - 0.62 * inch, width - inch, height - 0.62 * inch)
    canvas.setFont(BODY_FONT, 10)
    canvas.drawCentredString(width / 2.0, 0.5 * inch, str(doc.page))
    canvas.restoreState()


def title_page(styles) -> list:
    """Build the title-page flowables."""
    return [
        Spacer(1, 1.6 * inch),
        Paragraph(TITLE, styles["Title"]),
        Spacer(1, 0.4 * inch),
        Paragraph("[Author Name]", styles["Author"]),
        Paragraph("Fox School of Business, Temple University", styles["Author"]),
        Paragraph("[author.email@temple.edu]", styles["Author"]),
        Spacer(1, 0.5 * inch),
        Paragraph("This draft: generated from the var-confidence-audit pipeline", styles["Author"]),
        Spacer(1, 0.6 * inch),
        Paragraph("<b>JEL codes:</b> G21, G28, G32", styles["Author"]),
        Paragraph(
            "<b>Keywords:</b> Value-at-Risk, backtesting, bank regulation, "
            "Basel III, Christoffersen test, Kupiec test",
            styles["Author"],
        ),
        PageBreak(),
    ]


def references_section(styles) -> list:
    """Build the APA-style references flowables."""
    flowables = [Paragraph("References", styles["Heading"])]
    flowables += [Paragraph(ref, styles["Reference"]) for ref in REFERENCES]
    return flowables


def build_paper() -> Path:
    """Assemble and write the main paper PDF.

    Returns:
        Path of the generated PDF.
    """
    styles = get_styles()
    out = config.PAPER_OUTPUT_DIR / "var_confidence_audit.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=TITLE,
        author="var-confidence-audit",
    )

    story: list = []
    story += title_page(styles)
    story += abstract.build(styles)
    story += introduction.build(styles)
    story += data.build(styles)
    story += methodology.build(styles)
    story += results.build(styles)
    story += conclusion.build(styles)
    story.append(PageBreak())
    story += references_section(styles)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    logger.info("Wrote paper to %s", out)
    return out


def build_figures_appendix() -> Path:
    """Assemble and write the figures appendix PDF.

    Returns:
        Path of the generated PDF.
    """
    from PIL import Image as PILImage

    styles = get_styles()
    out = config.PAPER_OUTPUT_DIR / "figures_appendix.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=f"{TITLE} - Figures Appendix",
    )

    story: list = [
        Paragraph("Figures Appendix", styles["Title"]),
        Paragraph(TITLE, styles["Author"]),
        Spacer(1, 0.3 * inch),
    ]
    figures = sorted(config.FIGURES_DIR.glob("*.png"))
    if not figures:
        story.append(
            Paragraph(
                "No figures found in outputs/figures — run the analysis "
                "pipeline first.",
                styles["BodyNoIndent"],
            )
        )
    for idx, path in enumerate(figures, start=1):
        with PILImage.open(path) as im:
            aspect = im.height / im.width
        width = 6.4
        story.append(Image(str(path), width=width * inch, height=width * aspect * inch))
        story.append(
            Paragraph(
                f"Appendix Figure A{idx}. {path.stem.replace('_', ' ').capitalize()}.",
                styles["Caption"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    logger.info("Wrote figures appendix to %s", out)
    return out


def main() -> None:
    """Generate both PDFs."""
    config.ensure_dirs()
    build_paper()
    build_figures_appendix()
    logger.info("Paper generation complete.")


if __name__ == "__main__":
    main()
