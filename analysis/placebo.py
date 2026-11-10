"""Placebo test: do the disclosed VaR dynamics add anything over a threshold that ignores them?

The return-space threshold is (disclosed dollar VaR) / (market cap x trading intensity), and the
intensity is calibrated so the average threshold matches each bank's empirical return quantile.
Two consequences follow:
  * the unconditional level is fixed by construction, so Kupiec and UC tests are not informative
    about whether banks calibrate VaR well;
  * equity volatility clusters on its own, so violations may cluster in stress for any slow threshold.

This script reruns the headline tests on three thresholds with the same average level per bank:
  disclosed     the repo's threshold (quarterly disclosed VaR, daily market cap)
  mcap_only     disclosed VaR replaced by the bank's sample-average VaR (keeps market-cap movement)
  constant      one constant threshold per bank, equal to the mean disclosed threshold
If the disclosed threshold does not beat the placebos, the clustering and stress findings describe
equity volatility, not the banks' VaR models.

Output: data/processed/placebo_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from analysis.christoffersen import christoffersen_tests  # noqa: E402


def main() -> pd.DataFrame:
    panel = pd.read_csv(config.PROCESSED_DIR / "daily_violations.csv", parse_dates=["date"])
    rows = []
    for bank, g in panel.groupby("bank"):
        g = g.sort_values("date").reset_index(drop=True)
        mcap_scale = g["threshold_pct"] / g["var_mm"]          # 1e6 / (mcap x intensity), varies daily with price
        thresholds = {
            "disclosed": g["threshold_pct"],
            "mcap_only": mcap_scale * g["var_mm"].mean(),
            "constant": pd.Series(g["threshold_pct"].mean(), index=g.index),
        }
        p = float(g["expected_rate"].mean())
        for name, thr in thresholds.items():
            # rescale so every variant has exactly the same average threshold as the disclosed one
            thr = thr * g["threshold_pct"].mean() / thr.mean()
            v = (g["log_return"].abs() > thr).astype(int).to_numpy()
            res = christoffersen_tests(v, p)
            stress = g["stress"].astype(bool).to_numpy()
            sr, cr = v[stress].mean(), v[~stress].mean()
            rows.append({"bank": bank, "threshold": name, "n_obs": len(v), "violation_rate": v.mean(), "stated_rate": p,
                         "stress_rate": sr, "calm_rate": cr, "amplification": sr / cr if cr > 0 else np.nan,
                         **{k: res[k] for k in ("lr_uc", "p_uc", "lr_ind", "p_ind", "lr_cc", "p_cc")}})
    out = pd.DataFrame(rows)
    out.to_csv(config.PROCESSED_DIR / "placebo_comparison.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
