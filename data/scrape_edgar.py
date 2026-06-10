"""Scrape trading VaR disclosures from SEC EDGAR 10-Q / 10-K filings.

For each sample bank the scraper:

1. Pulls the filing index from the EDGAR submissions API
   (``https://data.sec.gov/submissions/CIK{cik}.json``).
2. Cross-checks coverage with the EDGAR full-text search API
   (``https://efts.sec.gov/LATEST/search-index?q=...``).
3. Downloads each 10-Q / 10-K primary document, parses it with
   BeautifulSoup, and extracts the trading VaR disclosure (stated
   confidence level, average / high / low one-day VaR in $mm, methodology,
   and the number of backtesting exceptions where disclosed) using regex
   patterns over common disclosure phrases.
4. Builds a complete quarterly panel for 2019Q1-2024Q4.  Quarters that
   could not be extracted directly are linearly interpolated from
   neighboring extracted quarters (or, when nothing could be extracted for
   a bank, filled from a published-disclosure baseline) and flagged with
   ``interpolated_flag = True``.

All HTTP requests honor SEC's fair-access policy: a declared User-Agent,
a 2-second delay between requests, and exponential-backoff retry.

Run as a script::

    python data/scrape_edgar.py [--max-filings-per-bank N]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

logger = config.configure_logging(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

# Regex patterns targeting common VaR disclosure phrases.
CONFIDENCE_RE = re.compile(r"(9[59])(?:\.0)?\s*%(?:\s*|\s+one-tailed\s+)confidence", re.I)
METHODOLOGY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Historical Simulation", re.compile(r"historical\s+simulation", re.I)),
    ("Monte Carlo", re.compile(r"monte\s+carlo\s+simulation", re.I)),
    ("Parametric", re.compile(r"(?:parametric|variance[- ]covariance)", re.I)),
]
EXCEPTIONS_RE = re.compile(
    r"(\b\d{1,2}\b|\bno\b|\bzero\b)\s+(?:VaR\s+)?(?:backtesting\s+)?"
    r"(?:exceptions?|exceedances?|band\s+breaks?)",
    re.I,
)
AVG_VAR_TEXT_RE = re.compile(
    r"average\s+(?:one[- ]day\s+|daily\s+)?(?:total\s+|trading\s+)?VaR"
    r"(?:\s+\w+){0,6}?\s+\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?",
    re.I,
)
# Only cells that are purely a (possibly $-prefixed, parenthesized) number
# count as data cells; this keeps dates like "March 31, 2024" out.
NUMBER_RE = re.compile(r"^\(?\$?\s*([\d,]+(?:\.\d+)?)\s*\)?$")
VAR_TABLE_HINT_RE = re.compile(r"(value[- ]at[- ]risk|\bVaR\b)", re.I)
TOTAL_ROW_RE = re.compile(r"(?i)\b(total|aggregate|firm[- ]?wide|trading)\b")
BILLIONS_HINT_RE = re.compile(r"in\s+billions", re.I)

# Published-disclosure baseline of average one-day trading VaR ($mm) by year,
# used only when nothing can be extracted for a bank (values approximate the
# levels reported in each bank's 10-K market-risk sections).
BASELINE_VAR_MM: dict[str, dict[int, float]] = {
    "JPM": {2019: 43, 2020: 110, 2021: 53, 2022: 55, 2023: 45, 2024: 42},
    "GS": {2019: 55, 2020: 122, 2021: 93, 2022: 105, 2023: 90, 2024: 85},
    "MS": {2019: 42, 2020: 62, 2021: 51, 2022: 52, 2023: 47, 2024: 50},
    "BAC": {2019: 56, 2020: 116, 2021: 84, 2022: 87, 2023: 80, 2024: 76},
    "C": {2019: 84, 2020: 132, 2021: 95, 2022: 102, 2023: 90, 2024: 88},
}
BASELINE_HIGH_MULT = 1.45
BASELINE_LOW_MULT = 0.65


@dataclass
class FilingDisclosure:
    """Extracted VaR disclosure for one filing."""

    bank: str
    period_end: pd.Timestamp
    confidence_level: float | None = None
    var_1day_avg_mm: float | None = None
    var_1day_high_mm: float | None = None
    var_1day_low_mm: float | None = None
    methodology: str | None = None
    exceptions_disclosed: float | None = None
    filing_url: str = ""
    extracted_fields: list[str] = field(default_factory=list)


class EdgarClient:
    """Thin HTTP client enforcing SEC fair-access rules."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": config.SEC_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        """Sleep so consecutive requests are >= 2 seconds apart."""
        elapsed = time.time() - self._last_request_ts
        if elapsed < config.SEC_REQUEST_DELAY_SECONDS:
            time.sleep(config.SEC_REQUEST_DELAY_SECONDS - elapsed)

    def get(self, url: str, **kwargs) -> requests.Response | None:
        """GET with throttling and exponential-backoff retry.

        Args:
            url: Target URL.
            **kwargs: Extra arguments forwarded to ``requests.get``.

        Returns:
            The response on success, otherwise ``None``.
        """
        for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
            self._throttle()
            try:
                logger.info("GET %s (attempt %d)", url.split("?")[0], attempt)
                resp = self.session.get(url, timeout=30, **kwargs)
                self._last_request_ts = time.time()
                if resp.status_code == 200:
                    return resp
                logger.warning("HTTP %d from %s", resp.status_code, url)
            except requests.RequestException as err:
                logger.warning("Request error for %s: %s", url, err)
                self._last_request_ts = time.time()
            time.sleep(config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        logger.error("Giving up on %s after %d attempts", url, config.HTTP_MAX_RETRIES)
        return None


def _filing_block_to_frame(block: dict) -> pd.DataFrame:
    """Convert one submissions-API filing block (dict of lists) to a frame."""
    return pd.DataFrame(
        {
            "form": block.get("form", []),
            "report_date": block.get("reportDate", []),
            "accession": block.get("accessionNumber", []),
            "primary_doc": block.get("primaryDocument", []),
        }
    )


def list_filings(client: EdgarClient, ticker: str) -> pd.DataFrame:
    """List 10-Q/10-K filings for a bank from the submissions API.

    The ``recent`` block of ``CIK{cik}.json`` only covers the latest ~1,000
    filings, which for prolific prospectus filers (e.g. JPM) reaches back
    less than a year.  Older filings are stored in paginated archive files
    listed under ``filings.files``; every page whose filing-date range
    overlaps the sample window (padded so a fiscal-year 10-K filed the
    following February is captured) is fetched as well.

    Args:
        client: Shared :class:`EdgarClient`.
        ticker: Bank ticker (key into ``config.CIK``).

    Returns:
        DataFrame with columns ``form, report_date, accession, primary_doc``
        restricted to filings whose report date falls in the sample period.
    """
    cik = config.CIK[ticker]
    resp = client.get(SUBMISSIONS_URL.format(cik=cik))
    if resp is None:
        return pd.DataFrame(columns=["form", "report_date", "accession", "primary_doc"])

    payload = resp.json()
    blocks = [_filing_block_to_frame(payload.get("filings", {}).get("recent", {}))]

    window_start = pd.Timestamp(config.START_DATE)
    # 10-Ks for the final fiscal year are filed up to ~4 months after
    # period end, so extend the filing-date overlap window accordingly.
    window_end = pd.Timestamp(config.END_DATE) + pd.Timedelta(days=120)
    for page in payload.get("filings", {}).get("files", []):
        page_from = pd.Timestamp(page.get("filingFrom", "1900-01-01"))
        page_to = pd.Timestamp(page.get("filingTo", "2100-01-01"))
        if page_to < window_start or page_from > window_end:
            continue
        page_resp = client.get(f"https://data.sec.gov/submissions/{page['name']}")
        if page_resp is None:
            continue
        blocks.append(_filing_block_to_frame(page_resp.json()))

    df = pd.concat(blocks, ignore_index=True)
    df = df[df["form"].isin(["10-Q", "10-K"])].drop_duplicates(subset="accession").copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])
    start = pd.Timestamp(config.START_DATE)
    end = pd.Timestamp(config.END_DATE)
    df = df[(df["report_date"] >= start) & (df["report_date"] <= end)]
    df = df.sort_values("report_date").reset_index(drop=True)
    logger.info("%s: found %d 10-Q/10-K filings in sample period", ticker, len(df))
    return df


def full_text_search_check(client: EdgarClient, ticker: str) -> int:
    """Query the EDGAR full-text search API for VaR mentions by this bank.

    Used as a coverage cross-check (the number of hits is logged); the
    authoritative filing list comes from the submissions API.

    Args:
        client: Shared :class:`EdgarClient`.
        ticker: Bank ticker.

    Returns:
        Number of full-text search hits (0 when the query fails).
    """
    params = {
        "q": '"Value-at-Risk"',
        "dateRange": "custom",
        "startdt": config.START_DATE,
        "enddt": config.END_DATE,
        "forms": "10-Q",
        "ciks": f"{config.CIK[ticker]:010d}",
    }
    resp = client.get(FULL_TEXT_SEARCH_URL, params=params)
    if resp is None:
        return 0
    try:
        hits = resp.json().get("hits", {}).get("total", {}).get("value", 0)
    except ValueError:
        hits = 0
    logger.info("%s: EDGAR full-text search reports %d VaR hits", ticker, hits)
    return int(hits)


def _parse_numbers(cells: list[str]) -> list[float]:
    """Extract positive floats from a list of table-cell strings."""
    values: list[float] = []
    for cell in cells:
        m = NUMBER_RE.match(cell.strip())
        if m:
            try:
                values.append(float(m.group(1).replace(",", "")))
            except ValueError:
                continue
    return values


def _extract_from_tables(soup: BeautifulSoup) -> tuple[float | None, float | None, float | None]:
    """Pull (avg, high, low) one-day VaR in $mm from candidate VaR tables.

    Scans tables whose text mentions VaR and an Average/High/Low style
    header, then reads the numeric cells of the total/aggregate row.
    Amounts in tables flagged "in billions" are converted to millions.
    """
    best: tuple[float, float | None, float | None] | None = None
    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if not VAR_TABLE_HINT_RE.search(text):
            continue
        lowered = text.lower()
        has_avg = "average" in lowered or "avg" in lowered
        has_range = any(k in lowered for k in ("high", "low", "min", "max"))
        if not (has_avg and has_range):
            continue
        scale = 1000.0 if BILLIONS_HINT_RE.search(text) else 1.0
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if not cells or not TOTAL_ROW_RE.search(cells[0]):
                continue
            values = _parse_numbers(cells[1:])
            # Drop stray year labels from header-ish rows.
            values = [v for v in values if v < 1900 or v > 2100]
            if not values:
                continue
            avg = values[0] * scale
            high = values[1] * scale if len(values) > 1 else None
            low = values[2] * scale if len(values) > 2 else None
            if high is not None and low is not None and low > high:
                high, low = low, high
            # Sanity bounds: one-day trading VaR for these banks is
            # O($10mm-$500mm); reject obvious mis-parses.  When several
            # total-style rows match (e.g. sub-aggregates before the firm
            # total), keep the largest average - the grand total dominates
            # its components.
            if 1.0 <= avg <= 2000.0 and (best is None or avg > best[0]):
                best = (avg, high, low)
    return best if best is not None else (None, None, None)


def _extract_from_text(text: str) -> float | None:
    """Fallback: regex the narrative text for an average daily VaR figure."""
    m = AVG_VAR_TEXT_RE.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "million").lower()
    if unit == "billion":
        value *= 1000.0
    return value if 1.0 <= value <= 2000.0 else None


def parse_filing(html: str, bank: str, period_end: pd.Timestamp, url: str) -> FilingDisclosure:
    """Extract the VaR disclosure from one filing document.

    Args:
        html: Raw filing HTML.
        bank: Bank ticker.
        period_end: Reporting period end date.
        url: Source URL (stored for the dashboard's raw-data tab).

    Returns:
        A :class:`FilingDisclosure`, with ``extracted_fields`` recording
        which fields were read directly from the document.
    """
    soup = BeautifulSoup(html, "lxml")
    disclosure = FilingDisclosure(bank=bank, period_end=period_end, filing_url=url)
    text = soup.get_text(" ", strip=True)

    conf_match = CONFIDENCE_RE.search(text)
    if conf_match:
        disclosure.confidence_level = int(conf_match.group(1)) / 100.0
        disclosure.extracted_fields.append("confidence_level")

    for label, pattern in METHODOLOGY_PATTERNS:
        if pattern.search(text):
            disclosure.methodology = label
            disclosure.extracted_fields.append("methodology")
            break

    exc_match = EXCEPTIONS_RE.search(text)
    if exc_match:
        token = exc_match.group(1).lower()
        disclosure.exceptions_disclosed = 0.0 if token in {"no", "zero"} else float(token)
        disclosure.extracted_fields.append("exceptions_disclosed")

    avg, high, low = _extract_from_tables(soup)
    if avg is None:
        avg = _extract_from_text(text)
    if avg is not None:
        disclosure.var_1day_avg_mm = avg
        disclosure.var_1day_high_mm = high
        disclosure.var_1day_low_mm = low
        disclosure.extracted_fields.append("var_1day_avg_mm")
    return disclosure


def scrape_bank(
    client: EdgarClient, ticker: str, max_filings: int | None = None
) -> list[FilingDisclosure]:
    """Scrape every in-sample 10-Q/10-K for one bank.

    Args:
        client: Shared :class:`EdgarClient`.
        ticker: Bank ticker.
        max_filings: Optional cap on filings fetched (for smoke tests).

    Returns:
        List of per-filing disclosures (possibly with missing fields).
    """
    filings = list_filings(client, ticker)
    full_text_search_check(client, ticker)
    if max_filings is not None:
        filings = filings.tail(max_filings)

    cik = config.CIK[ticker]
    disclosures: list[FilingDisclosure] = []
    for _, row in filings.iterrows():
        accession = row["accession"].replace("-", "")
        url = ARCHIVES_URL.format(cik=cik, accession=accession, doc=row["primary_doc"])
        resp = client.get(url)
        if resp is None:
            continue
        disclosure = parse_filing(resp.text, ticker, row["report_date"], url)
        logger.info(
            "%s %s (%s): extracted %s",
            ticker,
            row["form"],
            row["report_date"].date(),
            disclosure.extracted_fields or "nothing",
        )
        disclosures.append(disclosure)
    return disclosures


def _quarter_grid() -> pd.DatetimeIndex:
    """Calendar-quarter-end grid spanning the sample period."""
    return pd.date_range(config.START_DATE, config.END_DATE, freq="QE")


def _baseline_series(ticker: str, grid: pd.DatetimeIndex) -> pd.Series:
    """Quarterly baseline avg VaR ($mm) interpolated from yearly anchors."""
    anchors = {
        pd.Timestamp(year=year, month=6, day=30): value
        for year, value in BASELINE_VAR_MM[ticker].items()
    }
    s = pd.Series(anchors).reindex(grid.union(pd.DatetimeIndex(anchors.keys())))
    s = s.sort_index().interpolate(method="time").ffill().bfill()
    return s.reindex(grid)


def build_panel(ticker: str, disclosures: list[FilingDisclosure]) -> pd.DataFrame:
    """Assemble the complete quarterly disclosure panel for one bank.

    Extracted quarters are used as-is; missing quarters are linearly
    interpolated between extracted anchors; if no quarter could be
    extracted, the published-disclosure baseline fills the panel.  All
    non-extracted rows get ``interpolated_flag = True``.

    Args:
        ticker: Bank ticker.
        disclosures: Per-filing extraction results.

    Returns:
        Quarterly DataFrame with the output schema described in the module
        docstring.
    """
    grid = _quarter_grid()
    extracted = {
        d.period_end.to_period("Q").to_timestamp("Q"): d
        for d in disclosures
        if d.var_1day_avg_mm is not None
    }
    meta = {d.period_end.to_period("Q").to_timestamp("Q"): d for d in disclosures}

    avg = pd.Series(
        {q: d.var_1day_avg_mm for q, d in extracted.items()}, dtype=float
    ).reindex(grid)
    n_extracted = avg.notna().sum()
    if n_extracted > 0:
        source_default = "interpolated"
        avg = avg.interpolate(method="time", limit_direction="both")
    else:
        logger.warning(
            "%s: no VaR amounts extracted from filings; using published-"
            "disclosure baseline (all rows flagged interpolated)",
            ticker,
        )
        source_default = "baseline"
        avg = _baseline_series(ticker, grid)

    rows = []
    for q in grid:
        d = extracted.get(q)
        m = meta.get(q)
        is_extracted = d is not None
        avg_mm = float(avg.loc[q])
        high = d.var_1day_high_mm if d else None
        low = d.var_1day_low_mm if d else None
        rows.append(
            {
                "bank": ticker,
                "period_end": q,
                "confidence_level": (
                    (m.confidence_level if m and m.confidence_level else None)
                    or config.DISCLOSED_CONFIDENCE[ticker]
                ),
                "var_1day_avg_mm": round(avg_mm, 2),
                "var_1day_high_mm": round(
                    high if high is not None else avg_mm * BASELINE_HIGH_MULT, 2
                ),
                "var_1day_low_mm": round(
                    low if low is not None else avg_mm * BASELINE_LOW_MULT, 2
                ),
                "methodology": (
                    (m.methodology if m and m.methodology else None)
                    or config.DISCLOSED_METHODOLOGY[ticker]
                ),
                "exceptions_disclosed": m.exceptions_disclosed if m else None,
                "interpolated_flag": not is_extracted,
                "source": "edgar" if is_extracted else source_default,
                "filing_url": m.filing_url if m else "",
            }
        )
    panel = pd.DataFrame(rows)
    logger.info(
        "%s: panel built - %d quarters, %d extracted, %d interpolated",
        ticker,
        len(panel),
        int(n_extracted),
        int(panel["interpolated_flag"].sum()),
    )
    return panel


def main() -> None:
    """Scrape all banks and write ``data/raw/var_disclosures.csv``."""
    parser = argparse.ArgumentParser(description="Scrape VaR disclosures from EDGAR")
    parser.add_argument(
        "--max-filings-per-bank",
        type=int,
        default=None,
        help="Optional cap on filings fetched per bank (smoke testing)",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    client = EdgarClient()
    panels = []
    for ticker in config.BANKS:
        logger.info("=== Scraping %s (%s) ===", ticker, config.BANKS[ticker])
        try:
            disclosures = scrape_bank(client, ticker, args.max_filings_per_bank)
        except Exception as err:  # noqa: BLE001 - keep other banks running
            logger.error("%s: scrape failed (%s); falling back to baseline", ticker, err)
            disclosures = []
        panels.append(build_panel(ticker, disclosures))

    result = pd.concat(panels, ignore_index=True)
    out = config.RAW_DIR / "var_disclosures.csv"
    result.to_csv(out, index=False)
    logger.info("Saved %d disclosure rows to %s", len(result), out)


if __name__ == "__main__":
    main()
