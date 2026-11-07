"""Independent reimplementation of bank VaR models from daily returns.

For each sample bank, three 99% one-day VaR models are estimated on actual
daily log returns:

1. Historical Simulation (HS): 99th percentile of the empirical loss
   distribution over a rolling 250-day window (no distributional
   assumption).
2. Parametric (variance-covariance): rolling 250-day mean / std with
   VaR = -(mu - z_0.99 * sigma); a Student-t variant re-estimates the
   degrees of freedom by MLE (refit every 21 days for tractability).
3. EWMA (RiskMetrics): sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) *
   r2_{t-1} with lambda = 0.94 and VaR = z_0.99 * sigma_t.

Each model's VaR is expressed as % of price and converted to dollars with
the market-cap-based trading-equity proxy used for the disclosed figures,
making model and disclosed VaR directly comparable.  Violation rates and
Kupiec POF tests are reported for every model and for the disclosed VaR.

Outputs:
    * ``data/processed/model_var_estimates.csv``
    * ``data/processed/model_comparison.csv``
    * ``outputs/figures/model_var_timeseries_{bank}.png``
    * ``outputs/figures/model_violation_comparison.png``

Run as a script::

    python backtest/var_models.py
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
from analysis.kupiec import kupiec_pof  # noqa: E402

logger = config.configure_logging(__name__)

FIG_DPI = 300
ALPHA = 0.01            # 99% VaR -> 1% tail probability.
T_REFIT_EVERY = 21      # Re-estimate Student-t df monthly.
MODELS = ["historical_simulation", "parametric_normal", "parametric_t", "ewma"]


def historical_simulation_var(returns: pd.Series) -> pd.Series:
    """Rolling 250-day historical-simulation 99% VaR (% of price, positive).

    Args:
        returns: Daily log returns for one bank.

    Returns:
        Series of one-day VaR estimates aligned to the forecast date
        (window strictly precedes the day being forecast).
    """
    losses = -returns
    var = losses.rolling(config.ROLLING_WINDOW).quantile(1 - ALPHA)
    return var.shift(1)


def parametric_normal_var(returns: pd.Series) -> pd.Series:
    """Rolling 250-day variance-covariance 99% VaR under normality.

    VaR = -(mu - z_0.99 * sigma), reported as a positive loss quantile.

    Args:
        returns: Daily log returns for one bank.

    Returns:
        Series of one-day VaR estimates (shifted one day).
    """
    mu = returns.rolling(config.ROLLING_WINDOW).mean()
    sigma = returns.rolling(config.ROLLING_WINDOW).std()
    var = -(mu - config.Z_99 * sigma)
    return var.shift(1)


def parametric_t_var(returns: pd.Series) -> pd.Series:
    """Rolling parametric 99% VaR with Student-t innovations.

    The degrees of freedom are estimated by MLE on the trailing 250-day
    window, refit every ``T_REFIT_EVERY`` days and held constant in
    between; location/scale follow the rolling mean and std.

    Args:
        returns: Daily log returns for one bank.

    Returns:
        Series of one-day VaR estimates (shifted one day).
    """
    mu = returns.rolling(config.ROLLING_WINDOW).mean()
    sigma = returns.rolling(config.ROLLING_WINDOW).std()

    df_series = pd.Series(np.nan, index=returns.index)
    values = returns.to_numpy()
    for i in range(config.ROLLING_WINDOW, len(returns), T_REFIT_EVERY):
        window = values[i - config.ROLLING_WINDOW : i]
        try:
            df_hat, _, _ = stats.t.fit(window)
            df_series.iloc[i] = float(np.clip(df_hat, 2.1, 100.0))
        except Exception:  # noqa: BLE001 - keep NaN on a failed fit
            continue
    df_series = df_series.ffill()

    # t quantile scaled so the innovation variance matches sigma^2.
    t_q = pd.Series(
        stats.t.ppf(1 - ALPHA, df_series.to_numpy()), index=returns.index
    )
    scale_adj = np.sqrt((df_series - 2) / df_series)
    var = -(mu - t_q * scale_adj * sigma)
    return var.shift(1)


def ewma_var(returns: pd.Series) -> pd.Series:
    """RiskMetrics EWMA 99% VaR with lambda = 0.94.

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r2_{t-1};
    VaR_t = z_0.99 * sigma_t.

    Args:
        returns: Daily log returns for one bank.

    Returns:
        Series of one-day VaR estimates (already one-step-ahead by
        construction).
    """
    lam = config.EWMA_LAMBDA
    r2 = returns.to_numpy() ** 2
    sigma2 = np.empty(len(r2))
    sigma2[0] = r2[: min(25, len(r2))].mean()
    for t in range(1, len(r2)):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r2[t - 1]
    return pd.Series(config.Z_99 * np.sqrt(sigma2), index=returns.index)


def compute_model_panel(
    returns: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Estimate all models for all banks and assemble a long panel.

    Args:
        returns: Daily log returns (columns = tickers).
        prices: Daily adjusted close prices (columns = tickers).

    Returns:
        Long DataFrame with columns date, bank, log_return, one column per
        model VaR (% of price), and dollar VaR conversions (in $mm).
    """
    frames = []
    for ticker in config.BANKS:
        ret = returns[ticker].dropna()
        logger.info("Estimating VaR models for %s (%d days)", ticker, len(ret))
        models = {
            "historical_simulation": historical_simulation_var(ret),
            "parametric_normal": parametric_normal_var(ret),
            "parametric_t": parametric_t_var(ret),
            "ewma": ewma_var(ret),
        }
        df = pd.DataFrame({"log_return": ret})
        for name, series in models.items():
            df[name] = series
        px = prices[ticker].reindex(df.index).ffill()
        trading_base = (
            px * config.SHARES_OUTSTANDING_B[ticker] * 1e9 * config.TRADING_INTENSITY[ticker]
        )
        for name in models:
            df[f"{name}_dollar_mm"] = df[name] * trading_base / 1e6
        df["bank"] = ticker
        df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def evaluate_models(panel: pd.DataFrame, disclosed: pd.DataFrame) -> pd.DataFrame:
    """Compare violation rates of each model and the disclosed VaR.

    A model violation is a day where the absolute return exceeds the
    model's VaR estimate (matching the disclosed-VaR audit definition).
    Kupiec's POF test is run on every series at the 1% stated rate for
    models and at each bank's stated rate for the disclosure.

    Args:
        panel: Model VaR panel from :func:`compute_model_panel`.
        disclosed: Daily violation panel from ``analysis/violations.py``.

    Returns:
        Comparison DataFrame, one row per (bank, model).
    """
    rows = []
    for ticker in config.BANKS:
        sub = panel[panel["bank"] == ticker].dropna(subset=MODELS)
        for model in MODELS:
            viol = (sub["log_return"].abs() > sub[model]).astype(int)
            t, t1 = len(viol), int(viol.sum())
            lr, pval = kupiec_pof(t1, t, ALPHA)
            rows.append(
                {
                    "bank": ticker,
                    "model": model,
                    "stated_rate": ALPHA,
                    "n_obs": t,
                    "n_violations": t1,
                    "violation_rate": t1 / t if t else np.nan,
                    "kupiec_lr": lr,
                    "kupiec_p": pval,
                    "kupiec_reject_5pct": bool(pval < 0.05) if np.isfinite(pval) else None,
                }
            )

        disc = disclosed[disclosed["bank"] == ticker]
        t, t1 = len(disc), int(disc["violation"].sum())
        p = float(disc["expected_rate"].mean())
        lr, pval = kupiec_pof(t1, t, p)
        rows.append(
            {
                "bank": ticker,
                "model": "disclosed",
                "stated_rate": p,
                "n_obs": t,
                "n_violations": t1,
                "violation_rate": t1 / t if t else np.nan,
                "kupiec_lr": lr,
                "kupiec_p": pval,
                "kupiec_reject_5pct": bool(pval < 0.05) if np.isfinite(pval) else None,
            }
        )
    result = pd.DataFrame(rows)
    for _, row in result.iterrows():
        logger.info(
            "%s/%s: violation rate %.2f%% (Kupiec p=%.4f)",
            row["bank"],
            row["model"],
            row["violation_rate"] * 100,
            row["kupiec_p"],
        )
    return result


def plot_var_timeseries(panel: pd.DataFrame, disclosed: pd.DataFrame) -> list[Path]:
    """Plot disclosed vs model VaR time series for each bank.

    Args:
        panel: Model VaR panel.
        disclosed: Daily disclosed-VaR panel (threshold_pct column).

    Returns:
        Paths of the saved figures.
    """
    paths = []
    labels = {
        "historical_simulation": "Historical Simulation",
        "parametric_normal": "Parametric (normal)",
        "parametric_t": "Parametric (t)",
        "ewma": "EWMA (lambda=0.94)",
    }
    for ticker in config.BANKS:
        sub = panel[panel["bank"] == ticker].set_index("date")
        disc = disclosed[disclosed["bank"] == ticker].set_index("date")
        fig, ax = plt.subplots(figsize=(11, 5))
        for model, label in labels.items():
            ax.plot(sub.index, sub[model] * 100, lw=0.9, label=label, alpha=0.85)
        ax.plot(
            disc.index,
            disc["threshold_pct"] * 100,
            lw=1.6,
            color="black",
            label="Disclosed (normalized)",
        )
        ax.set_title(f"{ticker}: disclosed VaR vs reimplemented model VaR (1-day, % of price)")
        ax.set_ylabel("VaR (%)")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = config.FIGURES_DIR / f"model_var_timeseries_{ticker}.png"
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
        paths.append(out)
        logger.info("Saved %s", out)
    return paths


def plot_violation_comparison(comparison: pd.DataFrame) -> Path:
    """Grouped bar chart of violation rates across models and banks.

    Args:
        comparison: Output of :func:`evaluate_models`.

    Returns:
        Path of the saved figure.
    """
    order = ["disclosed"] + MODELS
    grid = comparison.pivot(index="bank", columns="model", values="violation_rate")
    grid = grid.reindex(index=list(config.BANKS), columns=order) * 100

    x = np.arange(len(grid))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model in enumerate(order):
        ax.bar(x + (i - 2) * width, grid[model], width, label=model.replace("_", " "))
    ax.axhline(1.0, color="black", ls="--", lw=1, label="1% target (99% VaR)")
    ax.set_xticks(x, grid.index)
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("Violation rates: disclosed VaR vs reimplemented models")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "model_violation_comparison.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def main() -> None:
    """Run the model reimplementation and comparison."""
    config.ensure_dirs()
    returns = pd.read_csv(
        config.RAW_DIR / "bank_returns.csv", index_col="date", parse_dates=True
    )
    prices = pd.read_csv(
        config.RAW_DIR / "bank_prices.csv", index_col="date", parse_dates=True
    )
    disclosed = pd.read_csv(
        config.PROCESSED_DIR / "daily_violations.csv", parse_dates=["date"]
    )

    panel = compute_model_panel(returns, prices)
    panel_out = config.PROCESSED_DIR / "model_var_estimates.csv"
    panel.to_csv(panel_out, index=False)
    logger.info("Saved model VaR estimates (%d rows) to %s", len(panel), panel_out)

    comparison = evaluate_models(panel, disclosed)
    comp_out = config.PROCESSED_DIR / "model_comparison.csv"
    comparison.to_csv(comp_out, index=False)
    logger.info("Saved model comparison to %s", comp_out)

    plot_var_timeseries(panel, disclosed)
    plot_violation_comparison(comparison)
    logger.info("Model backtest complete.")


if __name__ == "__main__":
    main()
