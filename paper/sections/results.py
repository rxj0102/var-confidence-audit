"""Results section: Tables 1-5 and embedded figures, built live from CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from paper.sections.common import df_to_table, fmt_num, fmt_pct, load_processed  # noqa: E402


def figure(name: str, caption: str, styles, width: float = 6.2) -> list:
    """Embed a figure from ``outputs/figures`` with a caption.

    Args:
        name: Figure file name.
        caption: Caption text.
        styles: Paper stylesheet.
        width: Display width in inches (height preserves aspect ratio).

    Returns:
        List of flowables (empty if the figure file is missing).
    """
    path = config.FIGURES_DIR / name
    if not path.exists():
        return [Paragraph(f"[figure missing: {name}]", styles["Caption"])]
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        aspect = im.height / im.width
    img = Image(str(path), width=width * inch, height=width * aspect * inch)
    return [Spacer(1, 0.08 * inch), img, Paragraph(caption, styles["Caption"])]


def table1(styles) -> list:
    """Table 1: violation rates by bank, actual vs expected."""
    summary = load_processed("violation_summary.csv")
    full = summary[summary["period"] == "full_sample"].set_index("bank").loc[list(config.BANKS)]
    df = pd.DataFrame(
        {
            "Bank": full.index,
            "Stated conf.": [fmt_pct(v, 0) for v in full["stated_conf"]],
            "Expected rate": [fmt_pct(v) for v in full["expected_rate"]],
            "Actual rate": [fmt_pct(v) for v in full["actual_violation_rate"]],
            "Excess (pp)": [f"{v * 100:+.2f}" for v in full["excess_rate"]],
            "Violations": full["n_violations"].astype(int).values,
            "Days": full["n_trading_days"].astype(int).values,
        }
    )
    return df_to_table(
        df,
        "Table 1. Full-sample violation rates against disclosed VaR, "
        "2019&ndash;2024. Violations use absolute daily log returns against "
        "the normalized disclosed threshold.",
        styles,
    )


def table2(styles) -> list:
    """Table 2: Christoffersen tests, full sample and subperiods."""
    christ = load_processed("christoffersen_results.csv")
    rows = []
    for ticker in config.BANKS:
        row: dict[str, str] = {"Bank": ticker}
        for sample, label in [("full_sample", "Full"), ("stress", "Stress"), ("calm", "Calm")]:
            for test in ["UC", "IND", "CC"]:
                hit = christ[
                    (christ["bank"] == ticker)
                    & (christ["sample"] == sample)
                    & (christ["test"] == test)
                ]
                if hit.empty:
                    row[f"{label} {test}"] = "—"
                else:
                    p = hit["p_value"].iloc[0]
                    star = "*" if p < 0.05 else ""
                    row[f"{label} {test}"] = f"{fmt_num(p)}{star}"
        rows.append(row)
    return df_to_table(
        pd.DataFrame(rows),
        "Table 2. Christoffersen (1998) test p-values by bank and subsample "
        "(UC = unconditional coverage, IND = independence, CC = conditional "
        "coverage). * rejects H0 at the 5% level.",
        styles,
        font_size=7.5,
    )


def table3(styles) -> list:
    """Table 3: annual Kupiec POF results."""
    kupiec = load_processed("kupiec_results.csv")
    annual = kupiec[kupiec["slice_type"] == "year"]
    rows = []
    for ticker in config.BANKS:
        row: dict[str, str] = {"Bank": ticker}
        for year, grp in annual[annual["bank"] == ticker].groupby("slice"):
            r = grp.iloc[0]
            star = "*" if r["p_value"] < 0.05 else ""
            row[str(year)] = f"{int(r['n_violations'])} ({fmt_num(r['p_value'])}{star})"
        rows.append(row)
    return df_to_table(
        pd.DataFrame(rows),
        "Table 3. Annual violations with Kupiec POF p-values in parentheses; "
        "* rejects H0 at the 5% level.",
        styles,
        font_size=7.5,
    )


def table4(styles) -> list:
    """Table 4: disclosed vs reimplemented model violation rates."""
    comp = load_processed("model_comparison.csv")
    grid = comp.pivot(index="bank", columns="model", values="violation_rate").reindex(
        list(config.BANKS)
    )
    order = ["disclosed", "historical_simulation", "parametric_normal", "parametric_t", "ewma"]
    labels = ["Disclosed", "Hist. Sim.", "Param. (N)", "Param. (t)", "EWMA"]
    df = pd.DataFrame({"Bank": grid.index})
    for col, label in zip(order, labels):
        df[label] = [fmt_pct(v) for v in grid[col]]
    return df_to_table(
        df,
        "Table 4. Violation rates of disclosed VaR vs reimplemented 99% "
        "models on identical return series.",
        styles,
    )


def table5(styles) -> list:
    """Table 5: cross-bank ranking and stress amplification."""
    ranking = load_processed("cross_bank_comparison.csv")
    amp = load_processed("stress_amplification.csv")
    amp_all = amp[amp["year"] == "all"].set_index("bank")["amplification"]
    df = pd.DataFrame(
        {
            "Bank": ranking["bank"],
            "Excess rate (pp)": [f"{v * 100:+.2f}" for v in ranking["excess_violation_rate"]],
            "CC p-value": [fmt_num(v) for v in ranking["cc_p_value"]],
            "Annual rate s.d.": [fmt_num(v, 4) for v in ranking["annual_rate_std"]],
            "Stress amplif.": [fmt_num(amp_all.get(b), 2) for b in ranking["bank"]],
            "Composite rank": [f"{v:.1f}" for v in ranking["composite_rank"]],
        }
    )
    return df_to_table(
        df,
        "Table 5. Cross-bank ranking (rank 1 = weakest disclosure quality) "
        "and full-sample stress amplification factors.",
        styles,
    )


def narrative(styles) -> list:
    """Short results narrative built from the live numbers."""
    summary = load_processed("violation_summary.csv")
    christ = load_processed("christoffersen_results.csv")
    full = summary[summary["period"] == "full_sample"]
    stress = summary[summary["period"] == "stress"].set_index("bank")
    calm = summary[summary["period"] == "calm"].set_index("bank")

    cc = christ[(christ["sample"] == "full_sample") & (christ["test"] == "CC")]
    ind = christ[(christ["sample"] == "full_sample") & (christ["test"] == "IND")]
    n_cc = int(cc["reject_5pct"].sum())
    n_ind = int(ind["reject_5pct"].sum())
    worst = full.loc[full["excess_rate"].idxmax()]
    stress_mult = (
        stress["actual_violation_rate"] / calm["actual_violation_rate"].replace(0, pd.NA)
    ).mean()

    text = (
        f"Table 1 reports full-sample coverage. The largest excess violation "
        f"rate belongs to {worst['bank']} "
        f"({worst['actual_violation_rate'] * 100:.2f}% actual against "
        f"{worst['expected_rate'] * 100:.2f}% expected). Tables 2 and 3 show "
        f"the formal tests: {n_cc} of {full['bank'].nunique()} banks reject "
        f"conditional coverage and {n_ind} reject independence at the 5% "
        f"level over the full sample, indicating that the dominant failure "
        f"mode is violation clustering rather than the average level of "
        f"disclosed VaR. The subsample columns of Table 2 and the stress "
        f"amplification factors in Table 5 sharpen the point: violation "
        f"rates in high-VIX regimes average roughly "
        f"{stress_mult:.1f}x their calm-period levels, exactly the pattern "
        f"the independence test is designed to detect. Table 4 shows that "
        f"reimplemented models estimated on public return data alone are "
        f"competitive with the disclosed figures, with EWMA — the only "
        f"specification that updates volatility daily — typically closest "
        f"to its nominal 1% target."
    )
    return [Paragraph(text, styles["Body"])]


def build(styles) -> list:
    """Build the results section flowables.

    Args:
        styles: Paper stylesheet.

    Returns:
        List of ReportLab flowables.
    """
    flowables = [Paragraph("4. Results", styles["Heading"])]
    flowables += narrative(styles)
    flowables += table1(styles)
    flowables += figure(
        "violation_rates_bar.png",
        "Figure 1. Actual vs expected violation rates with 95% confidence "
        "intervals.",
        styles,
    )
    flowables += table2(styles)
    flowables += figure(
        "christoffersen_pvalue_heatmap.png",
        "Figure 2. Christoffersen test p-values (full sample); red cells "
        "reject H0 at 5%.",
        styles,
        width=5.0,
    )
    flowables += table3(styles)
    flowables += figure(
        "kupiec_intervals.png",
        "Figure 3. Annual violations against the Kupiec 95% non-rejection "
        "band.",
        styles,
    )
    flowables += table4(styles)
    flowables += figure(
        "model_violation_comparison.png",
        "Figure 4. Violation rates of disclosed VaR vs reimplemented models.",
        styles,
    )
    flowables += table5(styles)
    flowables += figure(
        "stress_amplification_heatmap.png",
        "Figure 5. Stress amplification of violations across banks and years.",
        styles,
    )
    return flowables
