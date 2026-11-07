"""Kupiec (1995) Proportion-of-Failures (POF) backtest.

Implements the POF likelihood-ratio test on the daily violation indicator
panel for each bank: full sample, per calendar year, and per VIX regime.
Also computes Kupiec's two-tailed non-rejection interval for the number of
violations given the sample size, and flags banks falling outside it.

    LR_pof = -2 ln[ p^T1 (1-p)^(T-T1) / ( (T1/T)^T1 (1 - T1/T)^(T-T1) ) ]
    LR_pof ~ chi-squared(1) under H0.

Outputs:
    * ``data/processed/kupiec_results.csv``
    * ``outputs/figures/kupiec_intervals.png``

Run as a script::

    python analysis/kupiec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

logger = config.configure_logging(__name__)

SIG_LEVELS = (0.01, 0.05, 0.10)
FIG_DPI = 300


def _xlogy(x: float, y: float) -> float:
    """Return ``x * ln(y)`` with the convention ``0 * ln(0) = 0``."""
    if x == 0:
        return 0.0
    return x * np.log(y)


def kupiec_pof(t1: int, t: int, p: float) -> tuple[float, float]:
    """Compute the Kupiec POF likelihood-ratio statistic and p-value.

    Args:
        t1: Number of violations.
        t: Total number of observations.
        p: Stated violation probability.

    Returns:
        Tuple (LR statistic, p-value) under chi-squared(1).
    """
    if t == 0:
        return np.nan, np.nan
    p_hat = t1 / t
    log_null = _xlogy(t1, p) + _xlogy(t - t1, 1 - p)
    log_alt = _xlogy(t1, p_hat) + _xlogy(t - t1, 1 - p_hat)
    lr = max(-2.0 * (log_null - log_alt), 0.0)
    return lr, float(stats.chi2.sf(lr, df=1))


def kupiec_interval(t: int, p: float, significance: float = 0.05) -> tuple[int, int]:
    """Two-tailed non-rejection interval for the violation count.

    Finds the smallest and largest violation counts for which the POF test
    fails to reject H0 at the given significance level.

    Args:
        t: Number of observations.
        p: Stated violation probability.
        significance: Test size (default 5%).

    Returns:
        Tuple (lower bound, upper bound) of acceptable violation counts.
    """
    critical = stats.chi2.ppf(1 - significance, df=1)
    acceptable = [t1 for t1 in range(t + 1) if kupiec_pof(t1, t, p)[0] <= critical]
    if not acceptable:
        expected = int(round(t * p))
        return expected, expected
    return min(acceptable), max(acceptable)


def run_pof_battery(panel: pd.DataFrame) -> pd.DataFrame:
    """Run the POF test per bank for full sample, each year, each regime.

    Args:
        panel: Daily violation panel from ``analysis/violations.py``.

    Returns:
        Tidy results DataFrame, one row per (bank, slice).
    """
    rows = []
    for ticker in config.BANKS:
        bank = panel[panel["bank"] == ticker]
        slices: list[tuple[str, str, pd.DataFrame]] = [("full_sample", "full", bank)]
        slices += [
            (str(year), "year", grp) for year, grp in bank.groupby("year")
        ]
        slices += [
            (regime, "vix_regime", grp) for regime, grp in bank.groupby("regime")
        ]
        for label, slice_type, grp in slices:
            t = len(grp)
            t1 = int(grp["violation"].sum())
            p = float(grp["expected_rate"].mean())
            lr, pval = kupiec_pof(t1, t, p)
            lo, hi = kupiec_interval(t, p, 0.05)
            row = {
                "bank": ticker,
                "slice": label,
                "slice_type": slice_type,
                "n_obs": t,
                "n_violations": t1,
                "stated_rate": p,
                "actual_rate": t1 / t if t else np.nan,
                "lr_pof": lr,
                "p_value": pval,
                "ci_lower_violations": lo,
                "ci_upper_violations": hi,
                "outside_interval": bool(t1 < lo or t1 > hi),
            }
            for sig in SIG_LEVELS:
                row[f"reject_{int(sig * 100)}pct"] = (
                    bool(pval < sig) if np.isfinite(pval) else None
                )
            rows.append(row)
            logger.info(
                "%s/%s: %d/%d violations, LR=%.3f p=%.4f, interval [%d, %d]%s",
                ticker,
                label,
                t1,
                t,
                lr,
                pval,
                lo,
                hi,
                " OUTSIDE" if row["outside_interval"] else "",
            )
    return pd.DataFrame(rows)


def plot_intervals(results: pd.DataFrame) -> Path:
    """Plot annual violations vs the Kupiec non-rejection band per bank.

    Args:
        results: Output of :func:`run_pof_battery`.

    Returns:
        Path of the saved figure.
    """
    annual = results[results["slice_type"] == "year"].copy()
    annual["year"] = annual["slice"].astype(int)
    banks = list(config.BANKS)
    fig, axes = plt.subplots(1, len(banks), figsize=(4 * len(banks), 4.2), sharey=False)
    for ax, ticker in zip(np.atleast_1d(axes), banks):
        sub = annual[annual["bank"] == ticker].sort_values("year")
        ax.fill_between(
            sub["year"],
            sub["ci_lower_violations"],
            sub["ci_upper_violations"],
            alpha=0.25,
            color="#2c3e50",
            label="Kupiec 95% band",
        )
        colors = np.where(sub["outside_interval"], "#c0392b", "#27ae60")
        ax.scatter(sub["year"], sub["n_violations"], c=colors, zorder=3, s=45)
        ax.plot(sub["year"], sub["n_violations"], color="#7f8c8d", lw=1, zorder=2)
        ax.set_title(ticker)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Violations per year")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Annual VaR violations vs Kupiec non-rejection interval (red = outside)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "kupiec_intervals.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def main() -> None:
    """Run the Kupiec POF battery and persist outputs."""
    config.ensure_dirs()
    panel = pd.read_csv(
        config.PROCESSED_DIR / "daily_violations.csv", parse_dates=["date"]
    )
    results = run_pof_battery(panel)
    out = config.PROCESSED_DIR / "kupiec_results.csv"
    results.to_csv(out, index=False)
    logger.info("Saved %d Kupiec rows to %s", len(results), out)
    plot_intervals(results)
    logger.info("Kupiec analysis complete.")


if __name__ == "__main__":
    main()
