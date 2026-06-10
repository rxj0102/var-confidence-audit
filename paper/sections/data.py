"""Data section of the paper, with live summary statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.platypus import Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from paper.sections.common import df_to_table, fmt_pct  # noqa: E402

INTRO = (
    "The sample covers the five largest U.S. trading banks — JPMorgan Chase "
    "(JPM), Goldman Sachs (GS), Morgan Stanley (MS), Bank of America (BAC), "
    "and Citigroup (C) — at quarterly frequency from 2019Q1 through 2024Q4 "
    "(24 quarters per bank). Three data sources are combined."
)

EDGAR_PARA = (
    "<b>VaR disclosures (SEC EDGAR).</b> For each bank, every 10-Q and 10-K "
    "with a reporting period inside the sample window is retrieved through "
    "the EDGAR submissions API and cross-checked against the EDGAR full-text "
    "search API. Each primary filing document is parsed with BeautifulSoup; "
    "regular expressions targeting standard disclosure phrases "
    "(&ldquo;Value-at-Risk&rdquo;, &ldquo;one-day VaR&rdquo;, &ldquo;99% "
    "confidence&rdquo;) locate the trading-VaR table and extract the stated "
    "confidence level, the average / high / low one-day VaR for the quarter "
    "(normalized to millions of dollars), the disclosed methodology, and the "
    "number of backtesting exceptions where reported. Quarters whose dollar "
    "amounts could not be read directly from the filing text are linearly "
    "interpolated from adjacent extracted quarters (or, where extraction "
    "failed entirely for a bank, filled from the levels published in that "
    "bank's annual market-risk discussion) and are flagged as interpolated "
    "throughout; all scraping respects SEC fair-access rules (declared "
    "user agent, two-second request spacing, exponential-backoff retry)."
)

MARKET_PARA = (
    "<b>Market data.</b> Daily adjusted close prices for the five tickers and "
    "the VIX index are obtained from Yahoo Finance; daily log returns serve "
    "as the P&amp;L proxy. Two context series — the ICE BofA US High Yield "
    "OAS (BAMLH0A0HYM2) and the upper bound of the federal funds target range "
    "(DFEDTARU) — are drawn from FRED. Trading days with a VIX close above 25 "
    "are classified as stress days; the stress set is dominated by March&ndash;"
    "June 2020, September&ndash;October 2022, and March 2023 (the SVB episode)."
)

CAVEATS = (
    "<b>Caveats.</b> Two normalization assumptions deserve emphasis. First, "
    "actual desk-level trading P&amp;L is not public, so daily equity log "
    "returns proxy for the trading book's relative P&amp;L; equity returns "
    "embed leverage and non-trading earnings news, making the audit a joint "
    "test of the disclosure and the proxy. Second, disclosed dollar VaR is "
    "converted into return space by dividing by a trading-equity base equal "
    "to the bank's market capitalization times a fixed per-bank trading "
    "intensity, and violations are evaluated on absolute returns, which is "
    "deliberately conservative. Results should accordingly be read as "
    "evidence on the relative calibration and the dynamics (clustering, "
    "stress sensitivity) of disclosed VaR rather than as a literal "
    "regulatory backtest. Interpolated disclosure quarters are flagged and "
    "results are robust to dropping them."
)


def build(styles) -> list:
    """Build the data section flowables, including Table A (summary stats).

    Args:
        styles: Paper stylesheet.

    Returns:
        List of ReportLab flowables.
    """
    disclosures = pd.read_csv(
        config.RAW_DIR / "var_disclosures.csv", parse_dates=["period_end"]
    )

    rows = []
    for ticker, grp in disclosures.groupby("bank"):
        rows.append(
            {
                "Bank": ticker,
                "Conf.": fmt_pct(grp["confidence_level"].iloc[0], 0),
                "Methodology": grp["methodology"].mode().iloc[0],
                "Mean VaR ($mm)": f"{grp['var_1day_avg_mm'].mean():.0f}",
                "Min ($mm)": f"{grp['var_1day_avg_mm'].min():.0f}",
                "Max ($mm)": f"{grp['var_1day_avg_mm'].max():.0f}",
                "Quarters": len(grp),
                "Interp. (%)": fmt_pct(grp["interpolated_flag"].mean(), 0),
            }
        )
    table_df = pd.DataFrame(rows).set_index("Bank").loc[list(config.BANKS)].reset_index()

    flowables = [
        Paragraph("2. Data", styles["Heading"]),
        Paragraph(INTRO, styles["Body"]),
        Paragraph(EDGAR_PARA, styles["Body"]),
        Paragraph(MARKET_PARA, styles["Body"]),
    ]
    flowables += df_to_table(
        table_df,
        "Table A. Disclosed one-day trading VaR, 2019Q1&ndash;2024Q4. "
        "&ldquo;Interp.&rdquo; is the share of quarters not read directly "
        "from filing text.",
        styles,
    )
    flowables.append(Paragraph(CAVEATS, styles["Body"]))
    return flowables
