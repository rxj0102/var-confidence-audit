"""Unit tests for the Kupiec / Christoffersen tests and VaR model timing."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis.christoffersen import (
    christoffersen_tests,
    lr_ind,
    lr_uc,
    run_all,
    transition_counts,
)
from analysis.kupiec import kupiec_interval, kupiec_pof
from analysis.violations import classify_vix_regime
from backtest import var_models as vm
import config


# ---------------------------------------------------------------- Kupiec
def test_kupiec_zero_when_rate_matches():
    lr, p = kupiec_pof(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)


def test_kupiec_matches_direct_formula():
    t1, t, p = 18, 1000, 0.01
    ph = t1 / t
    expected = -2 * (
        t1 * np.log(p) + (t - t1) * np.log(1 - p)
        - t1 * np.log(ph) - (t - t1) * np.log(1 - ph)
    )
    assert kupiec_pof(t1, t, p)[0] == pytest.approx(expected)


def test_kupiec_zero_violations_is_finite():
    lr, pval = kupiec_pof(0, 250, 0.01)
    assert np.isfinite(lr) and lr > 0 and 0 < pval < 1


def test_kupiec_interval_is_consistent_with_test():
    lo, hi = kupiec_interval(250, 0.01, 0.05)
    crit = stats.chi2.ppf(0.95, 1)
    assert kupiec_pof(lo, 250, 0.01)[0] <= crit < kupiec_pof(hi + 1, 250, 0.01)[0]
    assert lo <= 2.5 <= hi


# --------------------------------------------------------- Christoffersen
def test_uc_equals_kupiec():
    v = np.zeros(500, dtype=int)
    v[::40] = 1
    assert lr_uc(v, 0.01)[0] == pytest.approx(kupiec_pof(int(v.sum()), 500, 0.01)[0])


def test_cc_is_uc_plus_ind():
    rng = np.random.default_rng(1)
    v = (rng.random(2000) < 0.02).astype(int)
    r = christoffersen_tests(v, 0.01)
    assert r["lr_cc"] == pytest.approx(r["lr_uc"] + r["lr_ind"])
    assert r["p_cc"] == pytest.approx(stats.chi2.sf(r["lr_cc"], 2))


def test_independence_detects_clustering():
    v = np.zeros(1000, dtype=int)
    for s in range(50, 1000, 100):
        v[s : s + 5] = 1  # five violations in a row, ten times
    assert lr_ind(v)[1] < 0.001
    rng = np.random.default_rng(3)
    iid = (rng.random(5000) < 0.05).astype(int)
    assert lr_ind(iid)[1] > 0.01


def test_transitions_skip_gaps_in_subsample():
    # violations at day 0 and day 10 of a subsample that excludes days 1..9
    v = np.array([1, 1])
    assert transition_counts(v) == (0, 0, 0, 1)  # naive: looks like a cluster
    assert transition_counts(v, np.array([0, 10])) == (0, 0, 0, 0)
    assert transition_counts(v, np.array([4, 5])) == (0, 0, 0, 1)


def test_run_all_stress_subsample_ignores_non_adjacent_days():
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    stress = np.zeros(n, dtype=bool)
    stress[::2] = True  # alternate days: no two stress days are consecutive
    viol = (rng.random(n) < 0.05).astype(int)
    panel = pd.DataFrame(
        {"bank": "JPM", "date": dates, "violation": viol, "stress": stress,
         "expected_rate": 0.05}
    )
    old = config.BANKS
    config.BANKS = {"JPM": "x"}
    try:
        res = run_all(panel)
    finally:
        config.BANKS = old
    ind = res[(res["sample"] == "stress") & (res["test"] == "IND")]
    assert ind["lr_stat"].isna().all()  # no consecutive pairs -> no estimate


# ------------------------------------------------------------ regimes
def test_vix_regimes_and_nan():
    assert classify_vix_regime(12.0) == "low"
    assert classify_vix_regime(20.0) == "medium"
    assert classify_vix_regime(40.0) == "high"
    assert pd.isna(classify_vix_regime(float("nan")))


# ------------------------------------------------------- VaR models
@pytest.fixture
def rets():
    rng = np.random.default_rng(5)
    return pd.Series(rng.standard_normal(900) * 0.01, index=pd.bdate_range("2019-01-01", periods=900))


@pytest.mark.parametrize("fn", [vm.historical_simulation_var, vm.parametric_normal_var, vm.ewma_var])
def test_var_forecast_is_point_in_time(fn, rets):
    """Changing today's return must not change today's VaR."""
    t = 600
    base = fn(rets)
    shocked = rets.copy()
    shocked.iloc[t] = -0.30
    alt = fn(shocked)
    assert alt.iloc[t] == pytest.approx(base.iloc[t])
    assert alt.iloc[t + 1] != pytest.approx(base.iloc[t + 1])  # but tomorrow reacts


def test_loss_side_violation_rate_close_to_alpha(rets):
    var = vm.historical_simulation_var(rets).dropna()
    flags = vm.violation_flags(rets.loc[var.index], var)
    assert 0.003 < flags.mean() < 0.02  # ~1%, not ~2% as with |r|
    abs_flags = (rets.loc[var.index].abs() > var).astype(int)
    assert abs_flags.mean() > 1.5 * flags.mean()  # the old definition ~doubles it


def test_normal_var_close_to_theory(rets):
    var = vm.parametric_normal_var(rets).dropna()
    assert var.mean() == pytest.approx(config.Z_99 * 0.01, rel=0.15)
