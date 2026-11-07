"""Interactive Streamlit dashboard for the VaR confidence audit.

Five tabs: bank overview, statistical tests, model comparison, cross-bank
results, and the raw scraped disclosures.  All content is read from
``data/processed/`` and ``data/raw/``; run the pipeline first
(see README run order).

Launch::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

st.set_page_config(page_title="VaR Confidence Audit", layout="wide")


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    """Load every pipeline artifact needed by the dashboard.

    Returns:
        Dict of named DataFrames.

    Raises:
        FileNotFoundError: If pipeline outputs are missing.
    """
    return {
        "daily": pd.read_csv(
            config.PROCESSED_DIR / "daily_violations.csv", parse_dates=["date"]
        ),
        "summary": pd.read_csv(config.PROCESSED_DIR / "violation_summary.csv"),
        "christoffersen": pd.read_csv(config.PROCESSED_DIR / "christoffersen_results.csv"),
        "kupiec": pd.read_csv(config.PROCESSED_DIR / "kupiec_results.csv"),
        "ranking": pd.read_csv(config.PROCESSED_DIR / "cross_bank_comparison.csv"),
        "anova": pd.read_csv(config.PROCESSED_DIR / "methodology_anova.csv"),
        "amplification": pd.read_csv(config.PROCESSED_DIR / "stress_amplification.csv"),
        "models": pd.read_csv(
            config.PROCESSED_DIR / "model_var_estimates.csv", parse_dates=["date"]
        ),
        "model_comparison": pd.read_csv(config.PROCESSED_DIR / "model_comparison.csv"),
        "disclosures": pd.read_csv(
            config.RAW_DIR / "var_disclosures.csv", parse_dates=["period_end"]
        ),
    }


def badge(passed: bool) -> str:
    """Render a pass/fail badge string for a test outcome."""
    return "✅ PASS" if passed else "❌ FAIL"


def style_reject(df: pd.DataFrame, p_cols: list[str]):
    """Color p-value cells: green = fail to reject H0, red = reject at 5%."""

    def _color(v):
        if pd.isna(v):
            return ""
        return (
            "background-color: #fadbd8" if v < 0.05 else "background-color: #d5f5e3"
        )

    return df.style.map(_color, subset=p_cols).format(precision=4)


def tab_bank_overview(data: dict[str, pd.DataFrame], bank: str) -> None:
    """Render Tab 1: single-bank overview with tiles and time series."""
    summary = data["summary"]
    full = summary[(summary["bank"] == bank) & (summary["period"] == "full_sample")].iloc[0]

    kupiec_full = data["kupiec"]
    kp = kupiec_full[
        (kupiec_full["bank"] == bank) & (kupiec_full["slice"] == "full_sample")
    ].iloc[0]
    cc = data["christoffersen"]
    cc_row = cc[
        (cc["bank"] == bank) & (cc["sample"] == "full_sample") & (cc["test"] == "CC")
    ].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Actual violation rate", f"{full['actual_violation_rate']:.2%}")
    c2.metric("Expected rate", f"{full['expected_rate']:.2%}")
    c3.metric("Excess rate", f"{full['excess_rate']:+.2%}")
    c4.metric("Kupiec POF", badge(not kp["reject_5pct"]), f"p = {kp['p_value']:.3f}")
    c5.metric(
        "Christoffersen CC", badge(not cc_row["reject_5pct"]), f"p = {cc_row['p_value']:.3f}"
    )

    daily = data["daily"]
    sub = daily[daily["bank"] == bank]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["date"], y=sub["log_return"] * 100, mode="lines",
            name="Daily log return", line={"color": "#7f8c8d", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["date"], y=sub["threshold_pct"] * 100, mode="lines",
            name="Disclosed VaR threshold", line={"color": "#2c3e50", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["date"], y=-sub["threshold_pct"] * 100, mode="lines",
            showlegend=False, line={"color": "#2c3e50", "width": 2},
        )
    )
    viol = sub[sub["violation"] == 1]
    fig.add_trace(
        go.Scatter(
            x=viol["date"], y=viol["log_return"] * 100, mode="markers",
            name="Violation", marker={"color": "red", "size": 6},
        )
    )
    fig.update_layout(
        title=f"{bank}: daily returns vs disclosed VaR threshold",
        yaxis_title="% of price", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    quarterly = summary[
        (summary["bank"] == bank)
        & (~summary["period"].isin(["full_sample", "stress", "calm"]))
    ]
    bar = px.bar(
        quarterly, x="period", y="actual_violation_rate",
        title=f"{bank}: quarterly violation rate vs theoretical rate",
        labels={"actual_violation_rate": "Violation rate"},
    )
    bar.add_hline(
        y=float(quarterly["expected_rate"].mean()), line_dash="dash",
        annotation_text="theoretical rate",
    )
    st.plotly_chart(bar, use_container_width=True)


def tab_statistical_tests(data: dict[str, pd.DataFrame]) -> None:
    """Render Tab 2: Christoffersen and Kupiec test tables."""
    sample = st.radio(
        "Sample window", ["full_sample", "stress", "calm"], horizontal=True,
        format_func=lambda s: {"full_sample": "Full sample", "stress": "Stress (VIX > 25)",
                               "calm": "Calm (VIX ≤ 25)"}[s],
    )
    christ = data["christoffersen"]
    sub = christ[christ["sample"] == sample]
    grid = sub.pivot(index="bank", columns="test", values="p_value").reindex(
        index=list(config.BANKS), columns=["UC", "IND", "CC"]
    )
    st.subheader("Christoffersen (1998) tests — p-values")
    st.caption("Green = fail to reject H0 at 5% (model adequate); red = reject.")
    st.dataframe(style_reject(grid.reset_index(), ["UC", "IND", "CC"]), hide_index=True)

    st.subheader("Kupiec (1995) POF test — annual")
    kupiec = data["kupiec"]
    annual = kupiec[kupiec["slice_type"] == "year"]
    kgrid = annual.pivot(index="bank", columns="slice", values="p_value").reindex(
        list(config.BANKS)
    )
    st.dataframe(
        style_reject(kgrid.reset_index(), [c for c in kgrid.columns]), hide_index=True
    )

    st.subheader("Kupiec POF by VIX regime")
    regime = kupiec[kupiec["slice_type"] == "vix_regime"][
        ["bank", "slice", "n_obs", "n_violations", "actual_rate", "p_value", "outside_interval"]
    ]
    st.dataframe(regime, hide_index=True)


def tab_model_comparison(data: dict[str, pd.DataFrame], bank: str) -> None:
    """Render Tab 3: disclosed vs reimplemented model VaR."""
    models = data["models"]
    sub = models[models["bank"] == bank]
    daily = data["daily"]
    disc = daily[daily["bank"] == bank]

    fig = go.Figure()
    for col, label in [
        ("historical_simulation", "Historical Simulation"),
        ("parametric_normal", "Parametric (normal)"),
        ("parametric_t", "Parametric (t)"),
        ("ewma", "EWMA"),
    ]:
        fig.add_trace(
            go.Scatter(x=sub["date"], y=sub[col] * 100, mode="lines", name=label,
                       line={"width": 1})
        )
    fig.add_trace(
        go.Scatter(x=disc["date"], y=disc["threshold_pct"] * 100, mode="lines",
                   name="Disclosed (normalized)", line={"color": "black", "width": 2})
    )
    fig.update_layout(
        title=f"{bank}: disclosed vs reimplemented 99% one-day VaR",
        yaxis_title="VaR (% of price)", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    comp = data["model_comparison"]
    bank_comp = comp[comp["bank"] == bank]
    bar = px.bar(
        bank_comp, x="model", y="violation_rate",
        title=f"{bank}: violation rate by model",
        labels={"violation_rate": "Violation rate"},
    )
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("Model accuracy ranking (|violation rate − stated rate|)")
    ranking = comp.copy()
    ranking["calibration_error"] = (
        ranking["violation_rate"] - ranking["stated_rate"]
    ).abs()
    ranking = (
        ranking.groupby("model", as_index=False)["calibration_error"].mean()
        .sort_values("calibration_error")
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    st.dataframe(ranking, hide_index=True)


def tab_cross_bank(data: dict[str, pd.DataFrame]) -> None:
    """Render Tab 4: cross-bank rankings and methodology comparison."""
    amp = data["amplification"]
    annual = amp[amp["year"] != "all"]
    grid = annual.pivot(index="bank", columns="year", values="amplification").reindex(
        list(config.BANKS)
    )
    heat = px.imshow(
        grid, text_auto=".1f", color_continuous_scale="OrRd",
        title="Stress amplification (stress / calm violation-rate ratio)",
        labels={"color": "Ratio"},
    )
    st.plotly_chart(heat, use_container_width=True)

    st.subheader("Bank ranking (composite of excess rate, CC p-value, consistency)")
    st.dataframe(data["ranking"], hide_index=True)

    st.subheader("Methodology group comparison (one-way ANOVA)")
    anova = data["anova"].iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("F statistic", f"{anova['f_stat']:.3f}")
    c2.metric("p-value", f"{anova['p_value']:.4f}")
    st.caption(
        "H0: mean quarterly violation rate is equal across disclosed VaR "
        "methodology groups (Historical Simulation / Monte Carlo / Parametric)."
    )
    st.dataframe(data["anova"], hide_index=True)


def tab_raw_data(data: dict[str, pd.DataFrame]) -> None:
    """Render Tab 5: filterable scraped disclosures with CSV download."""
    disclosures = data["disclosures"]
    c1, c2, c3 = st.columns(3)
    banks = c1.multiselect("Banks", list(config.BANKS), default=list(config.BANKS))
    sources = c2.multiselect(
        "Source", sorted(disclosures["source"].unique()),
        default=sorted(disclosures["source"].unique()),
    )
    only_extracted = c3.checkbox("Only directly extracted rows", value=False)

    filtered = disclosures[
        disclosures["bank"].isin(banks) & disclosures["source"].isin(sources)
    ]
    if only_extracted:
        filtered = filtered[~filtered["interpolated_flag"]]

    st.dataframe(
        filtered,
        hide_index=True,
        column_config={
            "filing_url": st.column_config.LinkColumn("Filing source", display_text="EDGAR ↗")
        },
    )
    st.download_button(
        "Download as CSV",
        filtered.to_csv(index=False).encode(),
        file_name="var_disclosures_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    """Assemble the dashboard."""
    st.title("VaR Confidence Audit")
    st.caption(
        "Do banks' disclosed VaR figures hold up? An empirical audit of 10-Q "
        "backtesting disclosures, 2019–2024."
    )
    try:
        data = load_data()
    except FileNotFoundError as err:
        st.error(
            f"Missing pipeline output: `{err.filename}`.\n\n"
            "Run the pipeline first — see the README run order "
            "(fetch → scrape → analysis → backtest)."
        )
        st.stop()

    bank = st.sidebar.selectbox("Bank", list(config.BANKS), format_func=lambda t: f"{t} — {config.BANKS[t]}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Data sources**: SEC EDGAR, Yahoo Finance, FRED.\n\n"
        "**Stress definition**: VIX > 25."
    )

    tabs = st.tabs(
        ["Bank Overview", "Statistical Tests", "Model Comparison", "Cross-Bank", "Raw Data"]
    )
    with tabs[0]:
        tab_bank_overview(data, bank)
    with tabs[1]:
        tab_statistical_tests(data)
    with tabs[2]:
        tab_model_comparison(data, bank)
    with tabs[3]:
        tab_cross_bank(data)
    with tabs[4]:
        tab_raw_data(data)


if __name__ == "__main__":
    main()
