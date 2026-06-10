"""Violation analysis: disclosed VaR thresholds vs realized daily returns.

Loads daily bank returns and quarterly VaR disclosures, normalizes each
disclosed dollar VaR into return space (dollar VaR divided by a market-cap
based trading-equity proxy), flags violation days where the absolute daily
return exceeds the disclosed threshold, and aggregates violation rates by
quarter, full sample, and stress (VIX > 25) vs calm subperiods.

Note on the violation definition: the audit uses absolute returns against
the one-sided VaR threshold.  This is deliberately conservative (it can
only overstate violations relative to a pure loss-side definition) and is
flagged in the paper's caveats.

Outputs:
    * ``data/processed/daily_violations.csv`` - daily panel reused by the
      Christoffersen / Kupiec test modules.
    * ``data/processed/violation_summary.csv`` - aggregated summary table.
    * ``outputs/figures/violation_rates_bar.png``
    * ``outputs/figures/violation_calendar_{bank}.png``

Run as a script::

    python analysis/violations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

logger = config.configure_logging(__name__)

FIG_DPI = 300


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load returns, prices, VIX and VaR disclosures from ``data/raw``.

    Returns:
        Tuple of (returns, prices, vix, disclosures).

    Raises:
        FileNotFoundError: If a required raw input is missing (run
            ``data/fetch.py`` and ``data/scrape_edgar.py`` first).
    """
    returns = pd.read_csv(config.RAW_DIR / "bank_returns.csv", index_col="date", parse_dates=True)
    prices = pd.read_csv(config.RAW_DIR / "bank_prices.csv", index_col="date", parse_dates=True)
    vix = pd.read_csv(config.RAW_DIR / "vix.csv", index_col="date", parse_dates=True)["vix"]
    disclosures = pd.read_csv(
        config.RAW_DIR / "var_disclosures.csv", parse_dates=["period_end"]
    )
    logger.info(
        "Loaded %d return days, %d disclosure rows", len(returns), len(disclosures)
    )
    return returns, prices, vix, disclosures


def classify_vix_regime(vix_level: float) -> str:
    """Map a VIX level to its configured regime label.

    Args:
        vix_level: VIX closing level.

    Returns:
        Regime label (``low`` / ``medium`` / ``high``).
    """
    for label, (lo, hi) in config.VIX_REGIMES.items():
        if lo < vix_level <= hi or (label == "low" and vix_level <= hi):
            return label
    return "high"


def build_daily_panel(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    vix: pd.Series,
    disclosures: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the daily bank panel with VaR thresholds and violations.

    Each trading day inherits the disclosed quarterly average VaR of the
    quarter it falls in.  The dollar VaR is converted to a return-space
    threshold by dividing by (market cap x trading intensity).

    Args:
        returns: Daily log returns (columns = tickers).
        prices: Daily adjusted close prices (columns = tickers).
        vix: VIX close series.
        disclosures: Quarterly disclosure panel from the scraper.

    Returns:
        Long-format daily panel with one row per (date, bank).
    """
    vix_aligned = vix.reindex(returns.index).ffill()
    frames = []
    for ticker in config.BANKS:
        disc = disclosures[disclosures["bank"] == ticker].set_index("period_end")
        ret = returns[ticker].dropna()
        px = prices[ticker].reindex(ret.index).ffill()
        mcap = px * config.SHARES_OUTSTANDING_B[ticker] * 1e9

        quarters = ret.index.to_period("Q").to_timestamp("Q")
        var_mm = pd.Series(quarters, index=ret.index).map(disc["var_1day_avg_mm"])
        conf = pd.Series(quarters, index=ret.index).map(disc["confidence_level"])
        interp = pd.Series(quarters, index=ret.index).map(disc["interpolated_flag"])

        threshold = (var_mm * 1e6) / (mcap * config.TRADING_INTENSITY[ticker])
        df = pd.DataFrame(
            {
                "date": ret.index,
                "bank": ticker,
                "log_return": ret.values,
                "var_mm": var_mm.values,
                "stated_conf": conf.values,
                "threshold_pct": threshold.values,
                "interpolated": interp.values,
                "vix": vix_aligned.reindex(ret.index).values,
            }
        ).dropna(subset=["threshold_pct", "log_return"])
        df["violation"] = (df["log_return"].abs() > df["threshold_pct"]).astype(int)
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["stress"] = panel["vix"] > config.STRESS_VIX_THRESHOLD
    panel["regime"] = panel["vix"].map(classify_vix_regime)
    panel["year"] = panel["date"].dt.year
    panel["quarter"] = panel["date"].dt.to_period("Q").astype(str)
    panel["expected_rate"] = 1.0 - panel["stated_conf"]
    return panel


def _summarize(group: pd.DataFrame) -> pd.Series:
    """Aggregate one (bank, period) group into a summary row."""
    n = len(group)
    n_viol = int(group["violation"].sum())
    expected = float(group["expected_rate"].mean())
    actual = n_viol / n if n else np.nan
    return pd.Series(
        {
            "stated_conf": float(group["stated_conf"].mean()),
            "actual_violation_rate": actual,
            "expected_rate": expected,
            "excess_rate": actual - expected,
            "n_violations": n_viol,
            "n_trading_days": n,
        }
    )


def summarize_violations(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the violation summary table at quarterly + aggregate levels.

    Args:
        panel: Daily panel from :func:`build_daily_panel`.

    Returns:
        Summary DataFrame with one row per (bank, period); ``period`` is a
        quarter label, ``full_sample``, ``stress`` or ``calm``.
    """
    blocks = []

    quarterly = (
        panel.groupby(["bank", "quarter"])
        .apply(_summarize, include_groups=False)
        .reset_index()
        .rename(columns={"quarter": "period"})
    )
    blocks.append(quarterly)

    full = (
        panel.groupby("bank").apply(_summarize, include_groups=False).reset_index()
    )
    full["period"] = "full_sample"
    blocks.append(full)

    for label, mask in [("stress", panel["stress"]), ("calm", ~panel["stress"])]:
        sub = (
            panel[mask].groupby("bank").apply(_summarize, include_groups=False).reset_index()
        )
        sub["period"] = label
        blocks.append(sub)

    summary = pd.concat(blocks, ignore_index=True)
    cols = [
        "bank",
        "period",
        "stated_conf",
        "actual_violation_rate",
        "expected_rate",
        "excess_rate",
        "n_violations",
        "n_trading_days",
    ]
    return summary[cols]


def plot_violation_rates(summary: pd.DataFrame) -> Path:
    """Bar chart of actual vs expected violation rates with error bars.

    Error bars are binomial standard errors of the estimated violation
    proportion.

    Args:
        summary: Output of :func:`summarize_violations`.

    Returns:
        Path of the saved figure.
    """
    full = summary[summary["period"] == "full_sample"].set_index("bank").loc[list(config.BANKS)]
    se = np.sqrt(
        full["actual_violation_rate"]
        * (1 - full["actual_violation_rate"])
        / full["n_trading_days"]
    )

    x = np.arange(len(full))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x - width / 2,
        full["actual_violation_rate"] * 100,
        width,
        yerr=se * 100 * 1.96,
        capsize=4,
        label="Actual",
        color="#c0392b",
    )
    ax.bar(
        x + width / 2,
        full["expected_rate"] * 100,
        width,
        label="Expected (stated confidence)",
        color="#2c3e50",
    )
    ax.set_xticks(x, full.index)
    ax.set_ylabel("Violation rate (% of trading days)")
    ax.set_title("Actual vs expected VaR violation rates, 2019-2024 (95% CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "violation_rates_bar.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_violation_calendars(panel: pd.DataFrame) -> list[Path]:
    """Calendar heatmap (year x month violation counts) per bank.

    Args:
        panel: Daily panel from :func:`build_daily_panel`.

    Returns:
        Paths of the saved figures.
    """
    paths = []
    for ticker in config.BANKS:
        sub = panel[panel["bank"] == ticker].copy()
        sub["month"] = sub["date"].dt.month
        grid = (
            sub.pivot_table(
                index="year", columns="month", values="violation", aggfunc="sum"
            )
            .reindex(columns=range(1, 13))
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            grid,
            cmap="Reds",
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "Violations"},
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"{ticker}: VaR violation clustering by calendar month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        fig.tight_layout()
        out = config.FIGURES_DIR / f"violation_calendar_{ticker}.png"
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
        paths.append(out)
        logger.info("Saved %s", out)
    return paths


def main() -> None:
    """Run the full violation analysis."""
    config.ensure_dirs()
    returns, prices, vix, disclosures = load_inputs()
    panel = build_daily_panel(returns, prices, vix, disclosures)

    panel_out = config.PROCESSED_DIR / "daily_violations.csv"
    panel.to_csv(panel_out, index=False)
    logger.info("Saved daily panel (%d rows) to %s", len(panel), panel_out)

    summary = summarize_violations(panel)
    summary_out = config.PROCESSED_DIR / "violation_summary.csv"
    summary.to_csv(summary_out, index=False)
    logger.info("Saved violation summary (%d rows) to %s", len(summary), summary_out)

    full = summary[summary["period"] == "full_sample"]
    for _, row in full.iterrows():
        logger.info(
            "%s: actual %.2f%% vs expected %.2f%% (%d violations / %d days)",
            row["bank"],
            row["actual_violation_rate"] * 100,
            row["expected_rate"] * 100,
            row["n_violations"],
            row["n_trading_days"],
        )

    plot_violation_rates(summary)
    plot_violation_calendars(panel)
    logger.info("Violation analysis complete.")


if __name__ == "__main__":
    main()
