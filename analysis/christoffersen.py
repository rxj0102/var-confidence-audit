"""Christoffersen (1998) three-component VaR backtesting framework.

Implements the unconditional coverage (UC), independence (IND) and
conditional coverage (CC) likelihood-ratio tests on the daily violation
indicator series produced by ``analysis/violations.py``, for each bank
over the full sample and the stress / calm subperiods.

Test statistics (with p = stated violation rate, n1 violations, n0
non-violations, p_hat = n1 / (n0 + n1)):

    LR_uc  = -2 ln[ p^n1 (1-p)^n0 / ( p_hat^n1 (1-p_hat)^n0 ) ]   ~ chi2(1)
    LR_ind = -2 ln[ L(p_hat) / L(pi_01, pi_11) ]                  ~ chi2(1)
    LR_cc  = LR_uc + LR_ind                                       ~ chi2(2)

Outputs:
    * ``data/processed/christoffersen_results.csv``
    * ``outputs/figures/christoffersen_pvalue_heatmap.png``

Run as a script::

    python analysis/christoffersen.py
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

logger = config.configure_logging(__name__)

SIGNIFICANCE = 0.05
FIG_DPI = 300


def _xlogy(x: float, y: float) -> float:
    """Return ``x * ln(y)`` with the convention ``0 * ln(0) = 0``."""
    if x == 0:
        return 0.0
    return x * np.log(y)


def lr_uc(violations: np.ndarray, p: float) -> tuple[float, float]:
    """Unconditional coverage (Kupiec-style) likelihood ratio test.

    Args:
        violations: Binary violation indicator series (0/1).
        p: Stated violation probability (e.g. 0.01 for 99% VaR).

    Returns:
        Tuple (LR statistic, p-value) under chi-squared(1).
    """
    n = len(violations)
    n1 = int(violations.sum())
    n0 = n - n1
    p_hat = n1 / n if n else np.nan
    if n == 0:
        return np.nan, np.nan
    log_null = _xlogy(n1, p) + _xlogy(n0, 1 - p)
    log_alt = _xlogy(n1, p_hat) + _xlogy(n0, 1 - p_hat)
    lr = -2.0 * (log_null - log_alt)
    return lr, float(stats.chi2.sf(lr, df=1))


def lr_ind(violations: np.ndarray) -> tuple[float, float]:
    """Independence test against first-order Markov violation clustering.

    Builds the 2x2 transition count matrix of the indicator series and
    compares the constant-probability likelihood against the Markov
    alternative with transition probabilities pi_01 and pi_11.

    Args:
        violations: Binary violation indicator series (0/1).

    Returns:
        Tuple (LR statistic, p-value) under chi-squared(1).
    """
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        return np.nan, np.nan
    prev, curr = v[:-1], v[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_null = _xlogy(n01 + n11, pi) + _xlogy(n00 + n10, 1 - pi)
    log_alt = (
        _xlogy(n01, pi01)
        + _xlogy(n00, 1 - pi01)
        + _xlogy(n11, pi11)
        + _xlogy(n10, 1 - pi11)
    )
    lr = -2.0 * (log_null - log_alt)
    lr = max(lr, 0.0)
    return lr, float(stats.chi2.sf(lr, df=1))


def christoffersen_tests(violations: np.ndarray, p: float) -> dict[str, float]:
    """Run UC, IND and CC tests on one violation series.

    Args:
        violations: Binary violation indicator series (0/1).
        p: Stated violation probability.

    Returns:
        Dict of statistics and p-values for all three tests.
    """
    uc_stat, uc_p = lr_uc(violations, p)
    ind_stat, ind_p = lr_ind(violations)
    cc_stat = uc_stat + ind_stat
    cc_p = float(stats.chi2.sf(cc_stat, df=2)) if np.isfinite(cc_stat) else np.nan
    return {
        "lr_uc": uc_stat,
        "p_uc": uc_p,
        "lr_ind": ind_stat,
        "p_ind": ind_p,
        "lr_cc": cc_stat,
        "p_cc": cc_p,
    }


def run_all(panel: pd.DataFrame) -> pd.DataFrame:
    """Run the three tests per bank for full sample and subperiods.

    Args:
        panel: Daily violation panel from ``analysis/violations.py``.

    Returns:
        Tidy results DataFrame, one row per (bank, sample, test).
    """
    rows = []
    samples = {
        "full_sample": pd.Series(True, index=panel.index),
        "stress": panel["stress"],
        "calm": ~panel["stress"],
    }
    for ticker in config.BANKS:
        for sample_label, mask in samples.items():
            sub = panel[(panel["bank"] == ticker) & mask].sort_values("date")
            if len(sub) < 30:
                logger.warning(
                    "%s/%s: only %d observations, skipping", ticker, sample_label, len(sub)
                )
                continue
            p = float(sub["expected_rate"].mean())
            res = christoffersen_tests(sub["violation"].to_numpy(), p)
            for test, df_chi in [("UC", 1), ("IND", 1), ("CC", 2)]:
                stat = res[f"lr_{test.lower()}"]
                pval = res[f"p_{test.lower()}"]
                rows.append(
                    {
                        "bank": ticker,
                        "sample": sample_label,
                        "test": test,
                        "lr_stat": stat,
                        "p_value": pval,
                        "df": df_chi,
                        "reject_5pct": bool(pval < SIGNIFICANCE) if np.isfinite(pval) else None,
                        "n_obs": len(sub),
                        "n_violations": int(sub["violation"].sum()),
                        "stated_rate": p,
                    }
                )
            logger.info(
                "%s/%s: UC p=%.4f IND p=%.4f CC p=%.4f",
                ticker,
                sample_label,
                res["p_uc"],
                res["p_ind"],
                res["p_cc"],
            )
    return pd.DataFrame(rows)


def plot_pvalue_heatmap(results: pd.DataFrame) -> Path:
    """Heatmap of full-sample p-values across banks and tests.

    Cells shaded red indicate rejection of H0 at the 5% level.

    Args:
        results: Output of :func:`run_all`.

    Returns:
        Path of the saved figure.
    """
    full = results[results["sample"] == "full_sample"]
    grid = full.pivot(index="bank", columns="test", values="p_value").reindex(
        index=list(config.BANKS), columns=["UC", "IND", "CC"]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=0.10,
        center=SIGNIFICANCE,
        linewidths=0.5,
        cbar_kws={"label": "p-value"},
        ax=ax,
    )
    ax.set_title("Christoffersen test p-values (full sample)\nred cells reject H0 at 5%")
    fig.tight_layout()
    out = config.FIGURES_DIR / "christoffersen_pvalue_heatmap.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def main() -> None:
    """Run all Christoffersen tests and persist outputs."""
    config.ensure_dirs()
    panel = pd.read_csv(
        config.PROCESSED_DIR / "daily_violations.csv", parse_dates=["date"]
    )
    results = run_all(panel)
    out = config.PROCESSED_DIR / "christoffersen_results.csv"
    results.to_csv(out, index=False)
    logger.info("Saved %d test rows to %s", len(results), out)
    plot_pvalue_heatmap(results)
    logger.info("Christoffersen analysis complete.")


if __name__ == "__main__":
    main()
