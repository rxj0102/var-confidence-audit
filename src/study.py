"""Do banks' trading VaR models hold up against the banks' own trading results?

Design, fixed before any test was computed:

  data        every backtesting count the five banks state in their 10-K filings
              (data/disclosures.csv): days on which the trading loss exceeded the VaR
  days        trading days stated in the filing, otherwise 252
  expected    (1 - confidence) x days for each bank-year
  tests       per bank and series: total observed against expected, with an exact two-sided
              binomial p-value; per bank-year: the Kupiec likelihood ratio, counting rejections
              in each direction at 5%; across years: a dispersion test of whether exceptions
              are more bunched in time than independent days would make them
  stress      2007, 2008 and 2020 are the stress years. They were chosen as the standard
              crisis years, not from the counts
  Basel       for the 99% series, the Basel traffic light per year on a 250 day scale:
              green 0 to 4 exceptions, yellow 5 to 9, red 10 or more
"""
from __future__ import annotations

from math import exp, lgamma, log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "disclosures.csv"
DEFAULT_DAYS = 252
STRESS_YEARS = (2007, 2008, 2020)
LABEL = {("JPM", "primary"): "JPMorgan", ("GS", "primary"): "Goldman Sachs",
         ("GS", "regulatory"): "Goldman Sachs, regulatory", ("MS", "primary"): "Morgan Stanley",
         ("BAC", "primary"): "Bank of America", ("C", "primary"): "Citigroup"}


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA, comment="#")
    d["days_used"] = d["days"].fillna(DEFAULT_DAYS).astype(int)
    d["p"] = 1.0 - d["confidence"]
    d["expected"] = d["p"] * d["days_used"]
    return d


def binom_logpmf(k: int, n: int, p: float) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)


def binom_two_sided(k: int, n: int, p: float) -> float:
    """Exact two-sided p-value: total probability of outcomes no more likely than k."""
    lk = binom_logpmf(k, n, p)
    total = sum(exp(lp) for lp in (binom_logpmf(i, n, p) for i in range(n + 1)) if lp <= lk + 1e-9)
    return float(min(1.0, total))


def binom_sum_two_sided(k: int, ns: list[int], ps: list[float]) -> float:
    """Two-sided p-value for a sum of independent binomials with different p, by convolution."""
    dist = np.array([1.0])
    for n, p in zip(ns, ps):
        pmf = np.exp([binom_logpmf(i, n, p) for i in range(n + 1)])
        dist = np.convolve(dist, pmf)
    return float(min(1.0, dist[dist <= dist[k] * (1 + 1e-9)].sum()))


def kupiec(k: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec proportion of failures likelihood ratio and its p-value (chi-square, 1 df)."""
    phat = k / n
    ll0 = (n - k) * log(1 - p) + (k * log(p) if k else 0.0)
    ll1 = ((n - k) * log(1 - phat) if k < n else 0.0) + (k * log(phat) if k else 0.0)
    lr = -2.0 * (ll0 - ll1)
    return float(lr), float(chi2.sf(lr, 1))


def dispersion(obs: np.ndarray, n: np.ndarray, p: np.ndarray) -> tuple[float, float, int]:
    """Are yearly counts more variable than independent days would make them?

    Under the null each year's count is binomial with its own n and p. The statistic sums
    (observed - expected)^2 / variance, computed around the series' own average rate so that a
    model that is uniformly too cautious is not counted as bunching. Chi-square, years - 1 df.
    """
    rate = obs.sum() / n.sum()
    e = rate * n
    v = e * (1 - rate)
    stat = float((((obs - e) ** 2) / v).sum()) if rate > 0 else 0.0
    df = len(obs) - 1
    return stat, float(chi2.sf(stat, df)), df


def basel_zone(k: int, days: int) -> str:
    scaled = k * 250 / days
    return "green" if scaled < 4.5 else ("yellow" if scaled < 9.5 else "red")


def summarize(d: pd.DataFrame) -> dict:
    obs, exp_ = int(d["exceptions"].sum()), float(d["expected"].sum())
    rows = list(d.itertuples())
    lr = [kupiec(r.exceptions, r.days_used, r.p) for r in rows]
    stat, pd_, df = dispersion(d["exceptions"].to_numpy(float), d["days_used"].to_numpy(float), d["p"].to_numpy())
    return {
        "years": int(len(d)), "first": int(d["year"].min()), "last": int(d["year"].max()),
        "observed": obs, "expected": exp_, "ratio": obs / exp_,
        "p": binom_sum_two_sided(obs, d["days_used"].tolist(), d["p"].tolist()),
        "years_below_expected": int((d["exceptions"] < d["expected"]).sum()),
        "years_too_few": int(sum(p < 0.05 and r.exceptions < r.expected for (_, p), r in zip(lr, rows))),
        "years_too_many": int(sum(p < 0.05 and r.exceptions > r.expected for (_, p), r in zip(lr, rows))),
        "years_zero": int((d["exceptions"] == 0).sum()),
        "dispersion": stat, "dispersion_p": pd_, "dispersion_df": df,
        "max_year": int(d.loc[d["exceptions"].idxmax(), "year"]), "max_count": int(d["exceptions"].max()),
    }


def blocks(d: pd.DataFrame) -> dict:
    """One block per bank, series and confidence level."""
    out = {}
    for (bank, series, conf), g in d.groupby(["bank", "series", "confidence"], sort=False):
        out[f"{bank}|{series}|{conf:.2f}"] = {"bank": bank, "series": series, "confidence": conf,
                                              "label": LABEL[(bank, series)], **summarize(g)}
    return out


def stress_split(d: pd.DataFrame) -> dict:
    s = d["year"].isin(STRESS_YEARS)
    out = {}
    for name, g in (("stress", d[s]), ("other", d[~s])):
        out[name] = {"bank_years": int(len(g)), "observed": int(g["exceptions"].sum()),
                     "expected": float(g["expected"].sum()), "ratio": float(g["exceptions"].sum() / g["expected"].sum())}
    return out


def basel(d: pd.DataFrame) -> dict:
    g = d[d["confidence"] == 0.99].copy()
    g["zone"] = [basel_zone(r.exceptions, r.days_used) for r in g.itertuples()]
    counts = g["zone"].value_counts()
    return {"bank_years": int(len(g)), **{z: int(counts.get(z, 0)) for z in ("green", "yellow", "red")},
            "non_green": [f"{r.bank} {r.year} ({r.exceptions})" for r in g.itertuples() if r.zone != "green"]}
