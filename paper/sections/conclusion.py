"""Conclusion section of the paper."""

from __future__ import annotations

from reportlab.platypus import Paragraph

PARAGRAPHS = [
    "This paper asked whether the VaR figures large U.S. banks disclose in "
    "their 10-Q filings survive an outside audit against realized market "
    "outcomes. Three findings stand out, each with a qualification. First, "
    "unconditional coverage cannot be judged here: the return-space level of "
    "disclosed VaR is set by a calibration to each bank's own return "
    "quantile, so average violation rates near the stated level reflect that "
    "calibration rather than bank conservatism. Second, the conditional "
    "coverage test rejects for most banks because violations cluster in "
    "high-VIX regimes. A placebo threshold that keeps the market-cap scaling "
    "but replaces the disclosures with their sample average clusters less "
    "for most banks, so this clustering reflects equity volatility and the "
    "normalization at least as much as the disclosures. Third, the "
    "disclosures do respond to stress: their stress to calm violation ratio "
    "is far below that of a constant threshold for every bank. Where stated "
    "levels are comparable (Bank of America and Citigroup at 99 percent), "
    "the disclosed figures sit closer to the stated rate than the "
    "reimplemented historical, parametric and EWMA models.",
    "<b>Regulatory implications.</b> The Basel traffic-light backtest "
    "evaluates only unconditional coverage over 250 days; the failure mode "
    "identified here — clustering — is invisible to it by construction. "
    "Requiring banks to disclose daily (or at least monthly) VaR alongside "
    "the quarterly average / high / low triple, and to report exceptions "
    "with their dates rather than counts, would let outside monitors run "
    "conditional-coverage audits at essentially zero supervisory cost. The "
    "FRTB's move toward expected shortfall does not remove the case for "
    "such disclosure: ES backtests also rest on the violation process.",
    "<b>Limitations.</b> Equity log returns are an imperfect stand-in for "
    "desk-level trading P&amp;L, the trading-intensity normalization is an "
    "assumption rather than a measurement, and quarters whose disclosures "
    "could not be machine-read were interpolated. The audit therefore "
    "speaks most credibly to the dynamics of disclosed VaR — clustering "
    "and stress sensitivity, which are invariant to the level "
    "normalization — and more cautiously to its level.",
    "<b>Extensions.</b> Natural next steps include widening the panel to "
    "European and Asian dealers, replacing the return proxy with the "
    "quarterly trading-revenue series from regulatory Y-9C filings, "
    "applying duration-based and dynamic-quantile backtests (Engle and "
    "Manganelli, 2004), and auditing the expected-shortfall disclosures "
    "that will accompany FRTB adoption.",
]


def build(styles) -> list:
    """Build the conclusion flowables.

    Args:
        styles: Paper stylesheet.

    Returns:
        List of ReportLab flowables.
    """
    flowables = [Paragraph("5. Conclusion", styles["Heading"])]
    flowables += [Paragraph(p, styles["Body"]) for p in PARAGRAPHS]
    return flowables
