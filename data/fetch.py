"""Fetch market data needed by the VaR audit pipeline.

Downloads (1) daily adjusted close prices for the five sample banks via
yfinance and computes daily log returns as a P&L proxy, (2) the VIX index,
and (3) two macro context series from FRED (high-yield OAS and the upper
bound of the fed funds target range).  Everything is written to
``data/raw/`` as CSVs with a datetime index.

Run as a script::

    python data/fetch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

logger = config.configure_logging(__name__)


def _download_with_retry(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from yfinance with retry.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD), inclusive.

    Returns:
        DataFrame of adjusted close prices indexed by date, one column per
        ticker.

    Raises:
        RuntimeError: If all retry attempts fail or no data is returned.
    """
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
        try:
            logger.info(
                "Downloading %s from yfinance (attempt %d/%d)",
                tickers,
                attempt,
                config.HTTP_MAX_RETRIES,
            )
            data = yf.download(
                tickers,
                start=start,
                # yfinance treats `end` as exclusive; pad one day so the
                # configured end date is included.
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
            if data is None or data.empty:
                raise RuntimeError("yfinance returned an empty frame")
            close = data["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=tickers[0])
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.dropna(how="all").sort_index()
            return close
        except Exception as err:  # noqa: BLE001 - retry on any transport error
            last_err = err
            wait = config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning("Download failed (%s); retrying in %.1fs", err, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {tickers}: {last_err}")


def fetch_bank_prices() -> pd.DataFrame:
    """Fetch daily adjusted close prices for all sample banks.

    Returns:
        DataFrame of adjusted close prices (columns = tickers).
    """
    tickers = list(config.BANKS)
    prices = _download_with_retry(tickers, config.START_DATE, config.END_DATE)
    prices = prices[tickers]
    out = config.RAW_DIR / "bank_prices.csv"
    prices.to_csv(out, index_label="date")
    logger.info("Saved %d rows of bank prices to %s", len(prices), out)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price panel.

    Args:
        prices: DataFrame of adjusted close prices.

    Returns:
        DataFrame of daily log returns (first row dropped).
    """
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    out = config.RAW_DIR / "bank_returns.csv"
    returns.to_csv(out, index_label="date")
    logger.info("Saved %d rows of log returns to %s", len(returns), out)
    return returns


def fetch_vix() -> pd.Series:
    """Fetch the VIX daily close.

    Returns:
        Series of VIX closing levels indexed by date.
    """
    vix = _download_with_retry([config.VIX_TICKER], config.START_DATE, config.END_DATE)
    series = vix.iloc[:, 0].rename("vix")
    out = config.RAW_DIR / "vix.csv"
    series.to_csv(out, index_label="date")
    logger.info("Saved %d rows of VIX to %s", len(series), out)
    return series


def fetch_fred_series() -> pd.DataFrame | None:
    """Fetch macro context series from FRED via fredapi.

    Requires ``FRED_API_KEY`` in the environment / ``.env``.  The series are
    contextual (used in the notebook and paper narrative); the core pipeline
    does not depend on them, so a missing key downgrades to a warning.

    Returns:
        DataFrame of FRED series, or ``None`` when no API key is configured.
    """
    if not config.FRED_API_KEY:
        logger.warning(
            "FRED_API_KEY not set - skipping FRED download "
            "(set it in .env to fetch %s)",
            list(config.FRED_SERIES),
        )
        return None

    from fredapi import Fred

    fred = Fred(api_key=config.FRED_API_KEY)
    frames: dict[str, pd.Series] = {}
    for series_id, label in config.FRED_SERIES.items():
        for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
            try:
                logger.info("Fetching FRED series %s (%s)", series_id, label)
                s = fred.get_series(
                    series_id,
                    observation_start=config.START_DATE,
                    observation_end=config.END_DATE,
                )
                frames[series_id] = s
                break
            except Exception as err:  # noqa: BLE001
                wait = config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "FRED fetch %s failed (%s); retry in %.1fs", series_id, err, wait
                )
                time.sleep(wait)
        else:
            logger.error("Giving up on FRED series %s", series_id)

    if not frames:
        return None
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    out = config.RAW_DIR / "fred_macro.csv"
    df.to_csv(out)
    logger.info("Saved %d rows of FRED data to %s", len(df), out)
    return df


def main() -> None:
    """Run the full market-data fetch."""
    config.ensure_dirs()
    logger.info(
        "Fetching market data for %s, %s to %s",
        list(config.BANKS),
        config.START_DATE,
        config.END_DATE,
    )
    prices = fetch_bank_prices()
    compute_log_returns(prices)
    fetch_vix()
    fetch_fred_series()
    logger.info("Market data fetch complete.")


if __name__ == "__main__":
    main()
