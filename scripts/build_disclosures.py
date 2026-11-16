"""Write data/disclosures.csv from the cached 10-K text and src/disclosures.py.

    python scripts/fetch_filings.py      # first, to fill data/raw/edgar/
    python scripts/build_disclosures.py

Fails if a phrase is missing from its filing, occurs more than once, or sits in a sentence that
does not state the recorded count or the year it covers.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.disclosures import SPEC, states_count  # noqa: E402

INDEX = ROOT / "data" / "raw" / "edgar" / "filings.json"
OUT = ROOT / "data" / "disclosures.csv"


def tenk(index: list[dict], bank: str, fiscal_year: int) -> dict:
    hits = [f for f in index if f["bank"] == bank and f["form"] == "10-K" and f["report_date"][:4] == str(fiscal_year)]
    if len(hits) != 1:
        raise SystemExit(f"{bank} {fiscal_year}: expected one 10-K, found {len(hits)}")
    return hits[0]


def main() -> None:
    index = json.loads(INDEX.read_text())
    rows, problems = [], []
    for r in SPEC:
        f = tenk(index, r.bank, r.filing_year)
        text = re.sub(r"\s+", " ", (ROOT / f["text_file"]).read_text())
        n = text.count(r.phrase)
        if n != 1:
            problems.append(f"{r.bank} {r.year} {r.series}: phrase found {n} times")
            continue
        i = text.index(r.phrase)
        start = max(text.rfind(". ", 0, i), text.rfind("; ", 0, i)) + 2
        end = text.find(". ", i + len(r.phrase))
        sentence = text[start:end + 1 if end > 0 else len(text)].strip()
        if not states_count(sentence, r.exceptions):
            problems.append(f"{r.bank} {r.year}: sentence does not state {r.exceptions}: {sentence[:160]}")
        if str(r.year) not in sentence and "twelve months" not in sentence.lower() and "12 months" not in sentence:
            context = text[max(0, start - 1500):start]
            if str(r.year) not in context:
                problems.append(f"{r.bank} {r.year}: year not found near the sentence")
        rows.append({**r._asdict(), "accession": f["accession"], "report_date": f["report_date"],
                     "filing_date": f["filing_date"], "sentence": sentence})
    if problems:
        raise SystemExit("\n".join(problems))
    df = pd.DataFrame(rows).sort_values(["bank", "series", "year"])
    df["days"] = df["days"].astype("Int64")
    OUT.write_text("# Backtesting exceptions disclosed by five US banks in their 10-K filings.\n"
                   "# Built by scripts/build_disclosures.py from SEC EDGAR; each count is stated in the sentence beside it.\n"
                   + df.to_csv(index=False))
    print(f"{len(df)} rows written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
