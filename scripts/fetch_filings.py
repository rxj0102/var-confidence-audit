"""Download every 10-K and 10-Q of the five banks from SEC EDGAR and keep their text.

    export EDGAR_USER_AGENT="Your Name your.address@example.com"
    python scripts/fetch_filings.py

SEC asks every client to identify itself in the User-Agent header and to stay under ten requests
a second. Responses are cached in data/raw/edgar/ (not committed), so a rerun downloads nothing.
For each filing the script keeps the plain text of every HTML document over 150 kB, which covers
the 10-K body and an annual report filed as Exhibit 13, where JPMorgan puts its risk section.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.banks import BANKS  # noqa: E402

CACHE = ROOT / "data" / "raw" / "edgar"
FORMS = {"10-K", "10-Q"}
FIRST = "2007-01-01"
MIN_BYTES = 150_000
DELAY = 0.15


def user_agent() -> str:
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if "@" not in ua:
        raise SystemExit('Set EDGAR_USER_AGENT="Your Name your.address@example.com" (SEC requires it).')
    return ua


def get(url: str, ua: str) -> bytes:
    key = CACHE / "http" / re.sub(r"[^A-Za-z0-9.]+", "_", url.split("//", 1)[1])
    if key.exists():
        return key.read_bytes()
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            body = urllib.request.urlopen(req, timeout=90).read()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    time.sleep(DELAY)
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(body)
    return body


def html_to_text(raw: bytes) -> str:
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s)


def filings(cik: int, ua: str) -> list[dict]:
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", ua))
    blocks = [sub["filings"]["recent"]] + [json.loads(get(f"https://data.sec.gov/submissions/{f['name']}", ua))
                                            for f in sub["filings"].get("files", [])]
    out = []
    for b in blocks:
        for form, acc, fdate, rdate in zip(b["form"], b["accessionNumber"], b["filingDate"], b["reportDate"]):
            if form in FORMS and fdate >= FIRST:
                out.append({"form": form, "accession": acc, "filing_date": fdate, "report_date": rdate})
    return sorted(out, key=lambda r: r["filing_date"])


def main() -> None:
    ua = user_agent()
    rows = []
    for bank, cik in BANKS.items():
        fl = filings(cik, ua)
        for f in fl:
            folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{f['accession'].replace('-', '')}"
            items = json.loads(get(f"{folder}/index.json", ua))["directory"]["item"]
            docs = [i["name"] for i in items if i["name"].lower().endswith((".htm", ".html"))
                    and not re.match(r"(?i)^R\d+\.htm", i["name"]) and int(i.get("size") or 0) >= MIN_BYTES]
            out = CACHE / "text" / bank / f"{f['report_date']}_{f['form']}_{f['accession']}.txt"
            if not out.exists():
                text = "\n".join(html_to_text(get(f"{folder}/{d}", ua)) for d in docs)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text)
            rows.append({"bank": bank, **f, "documents": " ".join(docs), "text_file": str(out.relative_to(ROOT))})
        print(f"{bank}: {len(fl)} filings", flush=True)
    index = ROOT / "data" / "raw" / "edgar" / "filings.json"
    index.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} filings indexed")


if __name__ == "__main__":
    main()
