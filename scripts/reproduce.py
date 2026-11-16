"""Run the study on data/disclosures.csv and write results/.

  results/metrics.json          everything the README reports
  results/figures/exceptions.png  exceptions per year against the number the VaR implies
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import study  # noqa: E402

ORDER = ["JPM|primary|0.99", "JPM|primary|0.95", "GS|primary|0.95", "MS|primary|0.95",
         "GS|regulatory|0.99", "BAC|primary|0.99", "C|primary|0.99"]


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def figure(d, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    panels = [("JPM", "primary"), ("GS", "primary"), ("MS", "primary"),
              ("GS", "regulatory"), ("BAC", "primary"), ("C", "primary")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, (bank, series) in zip(axes.ravel(), panels):
        g = d[(d["bank"] == bank) & (d["series"] == series)].sort_values("year")
        colors = ["#2a78d6" if c == 0.95 else "#eb6834" for c in g["confidence"]]
        ax.bar(g["year"], g["exceptions"], color=colors, width=0.7)
        z = g[g["exceptions"] == 0]
        ax.scatter(z["year"], np.zeros(len(z)), marker="o", s=14, color=[c for c, k in zip(colors, g["exceptions"]) if k == 0],
                   zorder=3, clip_on=False)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.step(g["year"], g["expected"], where="mid", color=ink, linewidth=1.2, linestyle="--")
        ax.set_title(study.LABEL[(bank, series)], loc="left", fontsize=10)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(2005.4, 2025.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Days the loss exceeded VaR", color=muted)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2a78d6", label="95% VaR"), Patch(color="#eb6834", label="99% VaR"),
                        Line2D([0], [0], color=ink, linestyle="--", label="Expected if the VaR were right"),
                        Line2D([0], [0], marker="o", color=muted, linestyle="", markersize=4, label="Reported zero")],
               loc="upper left", bbox_to_anchor=(0.005, 0.955), frameon=False, ncol=4, fontsize=9)
    fig.suptitle("Backtesting exceptions each bank reports in its 10-K (blank years: no report)", x=0.01, y=0.995, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exceptions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    d = study.load()
    b = study.blocks(d)
    groups = {"95%": study.summarize(d[d["confidence"] == 0.95]), "99%": study.summarize(d[d["confidence"] == 0.99])}
    payload = rounded({
        "data": {"rows": int(len(d)), "banks": int(d["bank"].nunique()), "first": int(d["year"].min()),
                 "last": int(d["year"].max()), "filings": int(d["accession"].nunique()),
                 "rows_with_stated_days": int(d["days"].notna().sum()), "default_days": study.DEFAULT_DAYS,
                 "stress_years": list(study.STRESS_YEARS)},
        "blocks": {k: b[k] for k in ORDER}, "groups": groups,
        "stress": {c: study.stress_split(d[d["confidence"] == float(c[:2]) / 100]) for c in ("95%", "99%")},
        "basel": study.basel(d),
    })
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    figure(d, out / "figures")
    for k, v in payload["blocks"].items():
        print(f"{k:22s} {v['first']}-{v['last']} yrs {v['years']:2d} obs {v['observed']:3d} exp {v['expected']:6.1f} "
              f"ratio {v['ratio']:.2f} p {v['p']:.2g} below {v['years_below_expected']} few {v['years_too_few']} "
              f"many {v['years_too_many']} zero {v['years_zero']} disp {v['dispersion']:.1f} p {v['dispersion_p']:.2g} max {v['max_year']}:{v['max_count']}")
    print(json.dumps(payload["groups"]), json.dumps(payload["stress"]), json.dumps(payload["basel"]), sep="\n")


if __name__ == "__main__":
    main()
