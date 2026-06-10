"""Shared ReportLab styles and helpers for the paper sections.

Every section module builds its flowables with the styles and table
helpers defined here so the paper has uniform typography: Times-Roman
12pt body text, bold 14pt section headers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, StyleSheet1
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402

BODY_FONT = "Times-Roman"
BOLD_FONT = "Times-Bold"
ITALIC_FONT = "Times-Italic"


def get_styles() -> StyleSheet1:
    """Build the paper's stylesheet.

    Returns:
        StyleSheet1 with Title, Author, Heading, SubHeading, Body,
        BodyNoIndent, Caption, Reference and Equation styles.
    """
    styles = StyleSheet1()
    styles.add(
        ParagraphStyle(
            "Title", fontName=BOLD_FONT, fontSize=16, leading=20,
            alignment=TA_CENTER, spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "Author", fontName=BODY_FONT, fontSize=12, leading=15,
            alignment=TA_CENTER, spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Heading", fontName=BOLD_FONT, fontSize=14, leading=17,
            spaceBefore=16, spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubHeading", fontName=BOLD_FONT, fontSize=12, leading=15,
            spaceBefore=10, spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Body", fontName=BODY_FONT, fontSize=12, leading=16,
            alignment=TA_JUSTIFY, firstLineIndent=18, spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyNoIndent", fontName=BODY_FONT, fontSize=12, leading=16,
            alignment=TA_JUSTIFY, spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Equation", fontName=ITALIC_FONT, fontSize=11, leading=15,
            alignment=TA_CENTER, spaceBefore=6, spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Caption", fontName=ITALIC_FONT, fontSize=10, leading=13,
            alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "Reference", fontName=BODY_FONT, fontSize=11, leading=14,
            leftIndent=18, firstLineIndent=-18, spaceAfter=4,
        )
    )
    return styles


def load_processed(name: str) -> pd.DataFrame:
    """Load one processed CSV by file name.

    Args:
        name: File name inside ``data/processed`` (e.g.
            ``violation_summary.csv``).

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: With a run-order hint when the file is missing.
    """
    path = config.PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run the analysis pipeline before generating "
            "the paper (see README run order)."
        )
    return pd.read_csv(path)


def df_to_table(
    df: pd.DataFrame,
    caption: str,
    styles: StyleSheet1,
    col_widths: list[float] | None = None,
    font_size: float = 8.5,
) -> list:
    """Convert a DataFrame into a styled ReportLab table with a caption.

    Args:
        df: Table content (already formatted as strings/numbers).
        caption: Caption text rendered beneath the table.
        styles: Paper stylesheet.
        col_widths: Optional column widths in points.
        font_size: Body font size of the table.

    Returns:
        List of flowables (table + caption + spacer).
    """
    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
                ("FONTNAME", (0, 1), (-1, -1), BODY_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return [table, Paragraph(caption, styles["Caption"]), Spacer(1, 0.08 * inch)]


def fmt_pct(x: float, decimals: int = 2) -> str:
    """Format a fraction as a percent string."""
    return f"{x * 100:.{decimals}f}%" if pd.notna(x) else "—"


def fmt_num(x: float, decimals: int = 3) -> str:
    """Format a float with fixed decimals, em-dash for NaN."""
    return f"{x:.{decimals}f}" if pd.notna(x) else "—"
