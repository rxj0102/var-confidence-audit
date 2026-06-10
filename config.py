"""Central configuration for the var-confidence-audit project.

All paths are resolved relative to the project root via :mod:`pathlib` so the
pipeline can be executed from any working directory.  Credentials (e.g. the
FRED API key) are loaded from a ``.env`` file via :mod:`python-dotenv` and are
never hard-coded.

Attributes:
    BANKS: Mapping of ticker -> human-readable bank name.
    CIK: Mapping of ticker -> SEC EDGAR Central Index Key.
    START_DATE / END_DATE: Sample period boundaries (inclusive).
    PRIMARY_CONFIDENCE / SECONDARY_CONFIDENCE: VaR confidence levels audited.
    VIX_REGIMES: VIX-level cut points defining low / medium / high regimes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PAPER_OUTPUT_DIR = OUTPUTS_DIR / "paper"

# Load credentials from .env at the project root (if present).
load_dotenv(PROJECT_ROOT / ".env")

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT", "var-confidence-audit research agent research@example.edu"
)

# ---------------------------------------------------------------------------
# Sample definition
# ---------------------------------------------------------------------------
BANKS: dict[str, str] = {
    "JPM": "JPMorgan Chase & Co.",
    "GS": "The Goldman Sachs Group, Inc.",
    "MS": "Morgan Stanley",
    "BAC": "Bank of America Corporation",
    "C": "Citigroup Inc.",
}

# SEC EDGAR Central Index Keys.
CIK: dict[str, int] = {
    "JPM": 19617,
    "GS": 886982,
    "MS": 895421,
    "BAC": 70858,
    "C": 831001,
}

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# VaR confidence levels: primary level used in the headline tests and a
# secondary level for robustness.
PRIMARY_CONFIDENCE = 0.99
SECONDARY_CONFIDENCE = 0.95

# Stated confidence level each bank uses in its public 10-Q/10-K trading VaR
# disclosures (used as a fallback when text extraction is inconclusive).
DISCLOSED_CONFIDENCE: dict[str, float] = {
    "JPM": 0.95,
    "GS": 0.95,
    "MS": 0.95,
    "BAC": 0.99,
    "C": 0.99,
}

# Disclosed VaR methodology per bank (fallback for the scraper).
DISCLOSED_METHODOLOGY: dict[str, str] = {
    "JPM": "Historical Simulation",
    "GS": "Historical Simulation",
    "MS": "Historical Simulation",
    "BAC": "Historical Simulation",
    "C": "Monte Carlo",
}

# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------
# VIX regimes: low (<=15), medium (15, 25], high (>25).
VIX_REGIMES: dict[str, tuple[float, float]] = {
    "low": (0.0, 15.0),
    "medium": (15.0, 25.0),
    "high": (25.0, float("inf")),
}

# A trading day is classified as "stress" when VIX closes above this level.
STRESS_VIX_THRESHOLD = 25.0

# ---------------------------------------------------------------------------
# Normalization assumptions
# ---------------------------------------------------------------------------
# Approximate diluted shares outstanding (billions) used to build a market-cap
# proxy from daily prices.  Held constant over the sample as a deliberate
# simplification, flagged in the paper's caveats section.
SHARES_OUTSTANDING_B: dict[str, float] = {
    "JPM": 2.95,
    "GS": 0.345,
    "MS": 1.65,
    "BAC": 8.10,
    "C": 1.95,
}

# Trading-book intensity: assumed ratio of the trading portfolio equity base
# to the firm's total market capitalization.  Disclosed dollar VaR is divided
# by (market cap x intensity) to obtain a return-space VaR threshold that is
# comparable to daily equity log returns (our P&L proxy).  This is a
# normalization assumption, documented in the Data section of the paper.
TRADING_INTENSITY: dict[str, float] = {
    "JPM": 0.0055,
    "GS": 0.0125,
    "MS": 0.0070,
    "BAC": 0.0045,
    "C": 0.0110,
}

# ---------------------------------------------------------------------------
# Market data series
# ---------------------------------------------------------------------------
VIX_TICKER = "^VIX"
FRED_SERIES: dict[str, str] = {
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "DFEDTARU": "Federal Funds Target Range - Upper Limit",
}

# ---------------------------------------------------------------------------
# Backtest model parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 250          # Basel-style 250-day rolling window.
EWMA_LAMBDA = 0.94            # RiskMetrics decay factor.
Z_99 = 2.326                  # Standard normal 99% quantile.
Z_95 = 1.645                  # Standard normal 95% quantile.

# SEC EDGAR politeness settings.
SEC_REQUEST_DELAY_SECONDS = 2.0
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF_BASE_SECONDS = 2.0


def ensure_dirs() -> None:
    """Create all output/processed directories if they do not exist."""
    for path in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, PAPER_OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(name: str) -> logging.Logger:
    """Configure root logging once and return a module logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
