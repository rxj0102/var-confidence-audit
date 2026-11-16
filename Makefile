# make test        run the tests
# make reproduce   results/ from data/disclosures.csv, then the README
# make check       README matches results/ (used in CI)
# make sources     rebuild data/disclosures.csv from EDGAR; needs EDGAR_USER_AGENT="Your Name your.address@example.com"
.PHONY: test reproduce check sources

test:
	uv run python -m pytest -q

reproduce:
	uv run python scripts/reproduce.py
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

sources:
	uv run python scripts/fetch_filings.py
	uv run python scripts/build_disclosures.py
