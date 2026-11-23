"""Checks on the sources table and on the statistics."""
from math import comb

import numpy as np
import pandas as pd
import pytest

from src import study
from src.disclosures import SPEC, states_count

D = study.load()


# ---------- the sources ----------

def test_table_matches_the_spec():
    spec = {(r.bank, r.year, r.series): r for r in SPEC}
    assert len(D) == len(spec) == len(SPEC)
    for row in D.itertuples():
        r = spec[(row.bank, row.year, row.series)]
        assert row.exceptions == r.exceptions and row.confidence == r.confidence and row.phrase == r.phrase


def test_every_sentence_states_its_count_and_contains_its_phrase():
    for row in D.itertuples():
        assert row.phrase in row.sentence
        assert states_count(row.sentence, row.exceptions), (row.bank, row.year)


def test_nonzero_counts_are_never_stated_as_zero():
    """A sentence with a positive count must not read as 'did not exceed' for that year."""
    for row in D[D["exceptions"] > 0].itertuples():
        assert f"did not exceed our 95% one-day VaR during {row.year}" not in row.sentence


def test_one_row_per_bank_series_year_and_sane_values():
    assert not D.duplicated(["bank", "series", "year"]).any()
    assert set(D["confidence"]) == {0.95, 0.99}
    assert D["exceptions"].between(0, 30).all()
    assert D["accession"].str.match(r"^\d{10}-\d{2}-\d{6}$").all()
    assert (D.loc[D["days"].notna(), "days"].between(250, 263)).all()


@pytest.mark.parametrize("sentence,n,ok", [
    ("exceeded our 95% one-day VaR on three occasions during 2006.", 3, True),
    ("exceeded our 95% one-day VaR on three occasions during 2006.", 2, False),
    ("did not exceed our 95% one-day VaR during 2009.", 0, True),
    ("observed 15 VaR back-testing exceptions", 15, True),
    ("observed 15 VaR back-testing exceptions", 1, False),
    ("a net trading loss of $87 million on one business day", 87, False),
])
def test_states_count(sentence, n, ok):
    assert states_count(sentence, n) is ok


# ---------- statistics ----------

def test_binomial_two_sided_matches_direct_sum():
    n, p, k = 30, 0.1, 7
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    direct = sum(x for x in pmf if x <= pmf[k] * (1 + 1e-9))
    assert study.binom_two_sided(k, n, p) == pytest.approx(direct)


def test_sum_of_binomials_equals_single_binomial_when_p_is_shared():
    assert study.binom_sum_two_sided(9, [100, 150], [0.05, 0.05]) == pytest.approx(study.binom_two_sided(9, 250, 0.05))


def test_kupiec():
    lr, p = study.kupiec(13, 252, 0.05)
    assert lr < 0.1 and p > 0.8                                  # right on the expected count
    lr0, p0 = study.kupiec(0, 252, 0.05)
    assert lr0 == pytest.approx(-2 * 252 * np.log(0.95)) and p0 < 1e-5
    assert study.kupiec(30, 250, 0.01)[1] < 1e-10


def test_dispersion_is_large_only_for_bunched_counts():
    n = np.full(10, 252.0)
    p = np.full(10, 0.05)
    even = np.full(10, 12.0)
    bunched = np.array([0, 0, 0, 0, 60, 0, 0, 0, 0, 60.0])
    assert study.dispersion(even, n, p)[1] == pytest.approx(1.0)
    assert study.dispersion(bunched, n, p)[1] < 1e-10
    uniform_low = np.full(10, 2.0)                               # cautious but steady: not bunched
    assert study.dispersion(uniform_low, n, p)[1] == pytest.approx(1.0)


@pytest.mark.parametrize("k,zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_zones(k, zone):
    assert study.basel_zone(k, 250) == zone


def test_expected_uses_stated_days_or_default():
    jpm = D[(D["bank"] == "JPM") & (D["year"] == 2017)].iloc[0]
    assert jpm["expected"] == pytest.approx(0.05 * 258)
    gs = D[(D["bank"] == "GS") & (D["year"] == 2017) & (D["series"] == "primary")].iloc[0]
    assert gs["expected"] == pytest.approx(0.05 * study.DEFAULT_DAYS)


def test_stress_split_partitions_the_rows():
    s = study.stress_split(D)
    assert s["stress"]["bank_years"] + s["other"]["bank_years"] == len(D)
    assert s["stress"]["observed"] + s["other"]["observed"] == D["exceptions"].sum()


def _frame(counts, p=0.01, days=252):
    return pd.DataFrame({"bank": "X", "series": "primary", "exceptions": counts,
                         "days_used": days, "p": p, "expected": p * days})


def test_bunching_p_matches_the_normal_approximation_without_overdispersion():
    # Counts equal to each year's expectation have no dispersion: phi floors at 1.
    d = _frame([3, 2, 3, 2, 3, 2])
    p, phi = study.p_allowing_for_bunching(d)
    assert phi == 1.0
    z = (d.exceptions.sum() - d.expected.sum()) / np.sqrt((d.days_used * d.p * (1 - d.p)).sum())
    from scipy.stats import norm
    assert p == pytest.approx(2 * norm.sf(abs(z)))


def test_bunching_widens_the_p_value():
    # Same total, once spread evenly and once bunched into two years.
    even = _frame([1, 1, 1, 1, 1, 1, 1, 1])
    bunched = _frame([0, 0, 0, 0, 0, 0, 4, 4])
    p_even, _ = study.p_allowing_for_bunching(even)
    p_bunched, phi = study.p_allowing_for_bunching(bunched)
    assert phi > 3
    assert p_bunched > p_even
