"""Introduction section of the paper."""

from __future__ import annotations

from reportlab.platypus import Paragraph

PARAGRAPHS = [
    # Basel context.
    "Since the 1996 Market Risk Amendment, and continuing under Basel III and "
    "the transition toward the Fundamental Review of the Trading Book (FRTB), "
    "large banks have been permitted to compute market-risk capital with "
    "internal models. Under the internal models approach, a bank's one-day "
    "Value-at-Risk (VaR) — typically stated at the 99% or 95% confidence "
    "level — maps directly into regulatory capital through a supervisory "
    "multiplier. The integrity of that mapping rests entirely on whether the "
    "model's stated coverage is honest: a 99% VaR should be exceeded on "
    "roughly 1% of trading days, no more and no less.",
    # Regulatory backtesting.
    "Supervisors police this claim with backtesting. The Basel framework "
    "compares the last 250 trading days of P&amp;L against the model's daily "
    "VaR and assigns the bank to a green, yellow, or red zone based on the "
    "number of exceptions: four or fewer exceptions leave the capital "
    "multiplier at its floor, five to nine raise it progressively, and ten or "
    "more trigger the red zone and a presumption of model failure. The "
    "backtest is mechanically simple, but it is run privately, on data "
    "outsiders cannot see, and its outcomes are only coarsely disclosed.",
    # Motivation: public audit.
    "U.S. securities regulation, however, opens a window. Item 3 of Regulation "
    "S-K requires quantitative market-risk disclosure, and the large dealers "
    "satisfy it by publishing average, high, and low one-day trading VaR in "
    "every 10-Q and 10-K, often together with the number of backtesting "
    "exceptions. These disclosures make an independent, outside audit of VaR "
    "adequacy possible: anyone can confront the disclosed risk envelope with "
    "realized market outcomes. This paper performs that audit for five large "
    "U.S. trading banks — JPMorgan Chase, Goldman Sachs, Morgan Stanley, Bank "
    "of America, and Citigroup — over 2019&ndash;2024, a window that contains the "
    "COVID-19 crash of March 2020, the 2022 rates repricing, and the March "
    "2023 regional-banking stress.",
    # Literature.
    "The exercise connects to a small but pointed literature. Berkowitz and "
    "O'Brien (2002), using confidential supervisory data, found that the "
    "trading VaR models of large U.S. banks were conservative on average yet "
    "failed to capture the dynamics of P&amp;L volatility — violations arrived "
    "in clusters. P&eacute;rignon and Smith (2010) documented, from public "
    "disclosures, that banks report too few exceptions relative to their "
    "stated confidence levels, consistent with deliberately conservative "
    "(over-stated) VaR. The statistical machinery for adjudicating these "
    "claims is due to Kupiec (1995), whose proportion-of-failures test checks "
    "unconditional coverage, and Christoffersen (1998), whose likelihood-ratio "
    "framework separates correct coverage from independence of violations and "
    "combines them into a conditional-coverage test.",
    # Contribution / roadmap.
    "This paper's contribution is a fully reproducible, open-source pipeline "
    "that (i) scrapes VaR disclosures directly from SEC EDGAR, (ii) "
    "normalizes them into return space against a market-cap-based trading "
    "proxy, (iii) applies the Kupiec and Christoffersen batteries across "
    "calm and stressed (VIX &gt; 25) regimes, and (iv) benchmarks the "
    "disclosed figures against three reimplemented VaR models — historical "
    "simulation, parametric variance-covariance, and RiskMetrics EWMA — "
    "estimated on the same return series. Section 2 describes the data, "
    "Section 3 the methodology, Section 4 the results, and Section 5 "
    "concludes with regulatory implications.",
]


def build(styles) -> list:
    """Build the introduction flowables.

    Args:
        styles: Paper stylesheet.

    Returns:
        List of ReportLab flowables.
    """
    flowables = [Paragraph("1. Introduction", styles["Heading"])]
    flowables += [Paragraph(p, styles["Body"]) for p in PARAGRAPHS]
    return flowables
