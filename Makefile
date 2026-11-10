# make reproduce   full pipeline: data, EDGAR scrape, tests, placebo, models, paper, README findings
# make check       verify README Key findings match data/processed/ (fast, used in CI)
# make test        run the unit tests
# EDGAR requires a User-Agent: export SEC_USER_AGENT="Name email@example.com"
.PHONY: setup reproduce check test

setup:
	uv sync

reproduce:
	uv run python data/fetch.py
	uv run python data/scrape_edgar.py
	uv run python analysis/violations.py
	uv run python analysis/christoffersen.py
	uv run python analysis/kupiec.py
	uv run python analysis/comparison.py
	uv run python analysis/placebo.py
	uv run python backtest/var_models.py
	MPLBACKEND=Agg uv run python paper/generate_paper.py
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	MPLBACKEND=Agg uv run pytest -q
