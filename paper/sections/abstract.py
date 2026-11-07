"""Abstract section of the paper (built from live pipeline results)."""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from paper.sections.common import load_processed  # noqa: E402


def build(styles) -> list:
    """Build the abstract flowables.

    Key magnitudes (number of banks, rejection counts) are pulled live
    from the processed results so the abstract always matches the tables.

    Args:
        styles: Paper stylesheet from :func:`common.get_styles`.

    Returns:
        List of ReportLab flowables.
    """
    summary = load_processed("violation_summary.csv")
    christ = load_processed("christoffersen_results.csv")

    full = summary[summary["period"] == "full_sample"]
    n_banks = full["bank"].nunique()
    mean_excess = full["excess_rate"].mean()
    cc = christ[(christ["sample"] == "full_sample") & (christ["test"] == "CC")]
    n_reject = int(cc["reject_5pct"].sum())

    text = (
        "Basel III permits large banks to set market-risk capital with internal "
        "Value-at-Risk (VaR) models, subject to backtesting, and U.S. banks must "
        "disclose trading VaR in quarterly 10-Q filings. These public disclosures "
        f"allow an independent audit of model adequacy. I hand-collect quarterly "
        f"VaR disclosures for {n_banks} large U.S. dealers (JPMorgan, Goldman "
        "Sachs, Morgan Stanley, Bank of America, Citigroup) from SEC EDGAR over "
        "2019&ndash;2024, normalize them into return space, and confront them with "
        "realized daily returns using the Kupiec (1995) proportion-of-failures "
        "test and the Christoffersen (1998) conditional-coverage framework. "
        f"Average excess violation rates are {mean_excess * 100:+.2f} percentage "
        f"points, and {n_reject} of {n_banks} banks reject conditional coverage "
        "at the 5% level, driven by pronounced violation clustering in "
        "high-VIX regimes. Simple reimplemented models (historical simulation, "
        "parametric, EWMA) match or beat disclosed figures, suggesting "
        "disclosure-based backtests are a cheap, informative supervisory "
        "complement."
    )

    return [
        Paragraph("Abstract", styles["Heading"]),
        Paragraph(text, styles["BodyNoIndent"]),
        Spacer(1, 0.1 * inch),
        Paragraph(
            "<b>JEL codes:</b> G21, G28, G32. <b>Keywords:</b> Value-at-Risk, "
            "backtesting, bank regulation, Basel III, Christoffersen test, "
            "Kupiec test.",
            styles["BodyNoIndent"],
        ),
        Spacer(1, 0.15 * inch),
    ]
