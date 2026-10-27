# polymarket-political

Can you spot informed money on Polymarket before the news does? This is a backtest harness built to answer that with a control group, a sealed holdout and trading costs, before writing any trading code.

## What it does

On Polymarket's political markets, prices sometimes move hard on unusual activity before any public headline. This project tests whether those moves carry information:

1. Pull every political market from Polymarket in a 6-month window (8 Oct 2025 to 8 Apr 2026) and keep those with at least $500k lifetime volume and $50k open interest.
2. Flag an event when a market's activity spikes (a multiple of its trailing 7-day level) together with a large one-hour price move. Optionally, skip events with a matching GDELT headline in the 6 hours before.
3. Simulate a $100 trade on every event. Score it at 24, 48 and 72 hours and at market resolution, after fees, spread, slippage and gas.
4. Run the same trades on a control group: random markets at random times in the same week, matched on volume quartile. The signal only counts if it beats that.

The full design is in [`docs/PRD.md`](docs/PRD.md) and [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).

## How it works

- **Ingestion** (`src/polymarket_political/ingest/`): clients for Polymarket's Gamma (markets), CLOB (prices, books) and Data (trades) APIs and GDELT's DOC API. All are public, so no keys are needed. Everything is cached to parquet with typed schemas that the tests assert field by field, so any API drift fails CI immediately.
- **No historical order book.** The week-1 report ([`reports/week1_data_feasibility.md`](reports/week1_data_feasibility.md)) found that Polymarket only serves the current book. Historical depth reconstruction is left as an explicit `NotImplementedError` rather than invented. The backtest uses mid-price history with a fixed-spread cost model and labels every number "provisional".
- **Two signal modes** (`signal/detector.py`). `trades` uses volume spike, price move and taker imbalance. `prices_only` uses realized-volatility spike and price move, because Polymarket's trade endpoint caps history at 3,000 records, so busy markets cannot be fully reconstructed. The detector is a pure function with no I/O.
- **Control group done fairly** (`backtest/control.py`). The control is matched on volume quartile, and the report shows the signal against both the news-filtered control and the raw one. Filtering only the signal side would quietly flatter it.
- **Holdout discipline.** The last 5 weeks are a sealed holdout window, pinned in code (`pipeline.py`), and all tuning is meant to happen on the first 21 weeks.
- **Testable end to end.** Every fetcher is injectable, so the whole pipeline runs on synthetic fixtures (`tests/fixtures/`) with pytest and hypothesis, with no network.

## Status

Paused research, not a finished result.

- Done: PRD and research plan, ingestion, signal, ledger and cost model, control group, metrics, pipeline, CLI, and a threshold and entry-timing sweep (`sweep.py`).
- Early lead, not a finding: a 32-combination sweep on 50 markets in one election week put a *contrarian* entry one hour after the spike on top, a Sharpe of about 1.4 under the provisional cost model. Fading the spike, not following it, is the opposite of the original hypothesis. One week, many combinations and a fixed-spread cost model mean it needs the untouched holdout before it means anything. The raw grids are in `data/processed/sweep_*.parquet`.
- Not done: a depth-aware cost model (subgraph or order-book reconstruction), the week-3 report (`reports/week3_first_run.md` is still the template), and the holdout run.

## Tech stack

Python 3.12, pandas, numpy, pyarrow, httpx, pydantic-settings, matplotlib, pytest, hypothesis, uv.

## Running locally

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"
pytest                         # hermetic, no network

# pull political markets for the research window and apply the liquidity floor
python -m polymarket_political.ingest.markets --apply-floor

# run the backtest on the training window and write a report
python -m polymarket_political.backtest run --window training --out reports/week3_first_run.md
python -m polymarket_political.backtest run --window training --max-markets 20   # quick smoke run
```

Thresholds can be overridden with environment variables (`POLYSIG_*`, `PMI_*`), as defined in `src/polymarket_political/config.py`.
