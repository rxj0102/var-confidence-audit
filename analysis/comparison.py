"""Cross-bank comparison of VaR disclosure quality.

Combines the violation summary, Christoffersen and Kupiec outputs to:

* rank banks by excess violation rate, conditional-coverage p-value and
  cross-year consistency (std dev of annual violation rates);
* test whether the disclosed VaR methodology group (Historical Simulation
  vs Monte Carlo vs Parametric) affects quarterly violation rates (one-way
  ANOVA);
* compute a stress amplification factor (stress-period violation rate
  divided by calm-period violation rate) per bank and per year.

Outputs:
    * ``data/processed/cross_bank_comparison.csv``
    * ``data/processed/methodology_anova.csv``
    * ``data/processed/stress_amplification.csv``
    * ``outputs/figures/stress_amplification_heatmap.png``

Run as a script::

    python analysis/comparison.py
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

FIG_DPI = 300


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed analysis outputs and raw disclosures.

    Returns:
        Tuple of (daily panel, violation summary, christoffersen results,
        disclosures).
    """
    panel = pd.read_csv(config.PROCESSED_DIR / "daily_violations.csv", parse_dates=["date"])
    summary = pd.read_csv(config.PROCESSED_DIR / "violation_summary.csv")
    christ = pd.read_csv(config.PROCESSED_DIR / "christoffersen_results.csv")
    disclosures = pd.read_csv(config.RAW_DIR / "var_disclosures.csv", parse_dates=["period_end"])
    return panel, summary, christ, disclosures


def build_ranking(
    panel: pd.DataFrame, summary: pd.DataFrame, christ: pd.DataFrame
) -> pd.DataFrame:
    """Rank banks on excess violations, CC p-value and consistency.

    Args:
        panel: Daily violation panel.
        summary: Violation summary table.
        christ: Christoffersen results table.

    Returns:
        One row per bank with the three metrics and their ranks
        (rank 1 = worst excess violations, lowest CC p-value, least
        consistent).
    """
    full = summary[summary["period"] == "full_sample"].set_index("bank")

    cc = (
        christ[(christ["sample"] == "full_sample") & (christ["test"] == "CC")]
        .set_index("bank")["p_value"]
    )

    annual_rates = (
        panel.groupby(["bank", "year"])["violation"].mean().rename("rate").reset_index()
    )
    consistency = annual_rates.groupby("bank")["rate"].std().rename("annual_rate_std")

    out = pd.DataFrame(
        {
            "excess_violation_rate": full["excess_rate"],
            "actual_violation_rate": full["actual_violation_rate"],
            "expected_rate": full["expected_rate"],
            "cc_p_value": cc,
            "annual_rate_std": consistency,
        }
    ).loc[list(config.BANKS)]
    out["rank_excess"] = out["excess_violation_rate"].rank(ascending=False).astype(int)
    out["rank_cc_pvalue"] = out["cc_p_value"].rank(ascending=True).astype(int)
    out["rank_consistency"] = out["annual_rate_std"].rank(ascending=False).astype(int)
    out["composite_rank"] = (
        out[["rank_excess", "rank_cc_pvalue", "rank_consistency"]].mean(axis=1)
    )
    out = out.sort_values("composite_rank").reset_index(names="bank")
    return out


def methodology_anova(
    summary: pd.DataFrame, disclosures: pd.DataFrame
) -> pd.DataFrame:
    """One-way ANOVA of quarterly violation rates across methodology groups.

    Args:
        summary: Violation summary table (quarterly rows used).
        disclosures: Quarterly disclosure panel with methodology labels.

    Returns:
        One-row DataFrame with the F statistic, p-value and group means.
    """
    method_map = disclosures.groupby("bank")["methodology"].agg(
        lambda s: s.mode().iloc[0]
    )
    quarterly = summary[
        ~summary["period"].isin(["full_sample", "stress", "calm"])
    ].copy()
    quarterly["methodology"] = quarterly["bank"].map(method_map)

    groups = [
        grp["actual_violation_rate"].dropna().to_numpy()
        for _, grp in quarterly.groupby("methodology")
    ]
    labels = sorted(quarterly["methodology"].unique())
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = np.nan, np.nan
        logger.warning("Only one methodology group present; ANOVA not identified")

    means = quarterly.groupby("methodology")["actual_violation_rate"].mean()
    row = {"f_stat": f_stat, "p_value": p_value, "n_groups": len(groups)}
    for label in labels:
        row[f"mean_rate_{label.replace(' ', '_').lower()}"] = means.get(label, np.nan)
    logger.info("Methodology ANOVA: F=%.3f p=%.4f (%d groups)", f_stat, p_value, len(groups))
    return pd.DataFrame([row])


def stress_amplification(panel: pd.DataFrame) -> pd.DataFrame:
    """Stress-to-calm violation-rate ratio per bank, overall and by year.

    Args:
        panel: Daily violation panel.

    Returns:
        DataFrame with one row per (bank, year or 'all'); the ratio is NaN
        when a year has no stress days or a zero calm-period rate.
    """
    rows = []
    slices = [("all", panel)] + [(str(y), g) for y, g in panel.groupby("year")]
    for label, grp in slices:
        for ticker, sub in grp.groupby("bank"):
            stress_rate = sub.loc[sub["stress"], "violation"].mean()
            calm_rate = sub.loc[~sub["stress"], "violation"].mean()
            ratio = (
                stress_rate / calm_rate
                if calm_rate and calm_rate > 0 and not np.isnan(stress_rate)
                else np.nan
            )
            rows.append(
                {
                    "bank": ticker,
                    "year": label,
                    "stress_rate": stress_rate,
                    "calm_rate": calm_rate,
                    "n_stress_days": int(sub["stress"].sum()),
                    "amplification": ratio,
                }
            )
    return pd.DataFrame(rows)


def plot_amplification_heatmap(amplification: pd.DataFrame) -> Path:
    """Heatmap of stress amplification factors across banks and years.

    Args:
        amplification: Output of :func:`stress_amplification`.

    Returns:
        Path of the saved figure.
    """
    annual = amplification[amplification["year"] != "all"]
    grid = annual.pivot(index="bank", columns="year", values="amplification").reindex(
        list(config.BANKS)
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".1f",
        cmap="OrRd",
        linewidths=0.5,
        cbar_kws={"label": "Stress / calm violation-rate ratio"},
        ax=ax,
    )
    ax.set_title("Stress amplification of VaR violations (blank = no stress days)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Bank")
    fig.tight_layout()
    out = config.FIGURES_DIR / "stress_amplification_heatmap.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def main() -> None:
    """Run the full cross-bank comparison."""
    config.ensure_dirs()
    panel, summary, christ, disclosures = load_inputs()

    ranking = build_ranking(panel, summary, christ)
    ranking_out = config.PROCESSED_DIR / "cross_bank_comparison.csv"
    ranking.to_csv(ranking_out, index=False)
    logger.info("Saved cross-bank ranking to %s", ranking_out)

    anova = methodology_anova(summary, disclosures)
    anova.to_csv(config.PROCESSED_DIR / "methodology_anova.csv", index=False)

    amp = stress_amplification(panel)
    amp.to_csv(config.PROCESSED_DIR / "stress_amplification.csv", index=False)
    plot_amplification_heatmap(amp)
    logger.info("Cross-bank comparison complete.")


if __name__ == "__main__":
    main()
