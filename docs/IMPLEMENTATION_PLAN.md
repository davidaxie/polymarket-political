# Implementation Plan: Polymarket Insider-Flow Backtest

**Companion to:** `docs/PRD.md`, `docs/RESEARCH_PLAN.md`
**Purpose:** Concrete build plan, structured around **parallel-agent dispatch** so independent workstreams run simultaneously in separate git worktrees.

---

## Stack Decisions

- **Language:** Python 3.12 (ecosystem for quant/backtest work is unmatched here)
- **Data:** `pandas` + `pyarrow` (Parquet everywhere — fast, typed, compressible)
- **HTTP:** `httpx` (async for parallel market pulls)
- **Notebooks:** `jupyter` for exploration; all reusable logic extracted into `src/`
- **Testing:** `pytest` — every pure function (signal, cost model, PnL ledger) gets a unit test
- **Config:** `pydantic-settings` for thresholds (so sensitivity sweeps are one-line changes)
- **Package manager:** `uv` (fast, lockfile, reproducible)

## Repo Layout

```
polymarket-insider/
├── docs/                          # PRD, research plan, this file, weekly reports
├── src/polymarket_insider/
│   ├── __init__.py
│   ├── config.py                  # Thresholds, windows, paths (pydantic)
│   ├── ingest/
│   │   ├── markets.py             # Enumerate political markets, apply liquidity floor
│   │   ├── orderbook.py           # Historical orderbook snapshots
│   │   ├── trades.py              # Historical trade tape
│   │   └── news.py                # GDELT headline pulls
│   ├── signal/
│   │   ├── features.py            # Rolling volume, price move, taker imbalance
│   │   └── detector.py            # The rule engine — pure function, heavily tested
│   ├── backtest/
│   │   ├── costs.py               # Fee / spread / slippage / gas model
│   │   ├── ledger.py              # PnL accounting per event
│   │   └── control.py             # Random control-group generator
│   └── report/
│       ├── metrics.py             # Hit rate, PnL, Sharpe, drawdown
│       └── plots.py               # Standard plots used across weekly reports
├── notebooks/
│   ├── 01_data_feasibility.ipynb
│   ├── 02_signal_exploration.ipynb
│   ├── 03_backtest_first_run.ipynb
│   ├── 04_parameter_sweep.ipynb
│   └── 05_holdout_FINAL.ipynb     # Run exactly once
├── data/                          # .gitignored — raw parquet cache
│   ├── raw/                       # Untouched API pulls
│   └── processed/                 # Feature tables, event tables
├── reports/                       # Weekly markdown artifacts per RESEARCH_PLAN
├── tests/                         # pytest — unit tests for src/
├── pyproject.toml
└── .env.example                   # API keys placeholder
```

`data/` is gitignored. Raw pulls are cached so re-running notebooks doesn't re-hit the API.

---

## Parallel Workstream Graph

The build has four workstreams that overlap heavily. Dependencies:

```
        ┌─── A. Ingestion ────┐
        │                     │
        ├─── B. Signal  ──────┼──→  D. Backtest harness  ──→  reports
        │                     │
        └─── C. Cost model ───┘
```

**A, B, C are independent.** They can run in parallel in separate worktrees and be merged to `main` as each completes. **D depends on all three** and is sequenced last. This is the canonical shape for parallel-agent dispatch.

Use the `superpowers:dispatching-parallel-agents` skill (listed in session context) to run A/B/C simultaneously. Each worktree gets a single, well-scoped prompt and returns a merged branch.

---

## Workstream A — Ingestion (parallel)

**Worktree branch:** `claude/ingest`
**Owner:** subagent A

### Tasks
1. Set up `pyproject.toml` with deps, `uv sync`, `src/polymarket_insider/__init__.py`
2. Write `config.py` with pydantic settings for the PRD thresholds (one place to change them)
3. `ingest/markets.py`:
   - `list_political_markets(start, end) -> DataFrame` — hit Polymarket `/markets` endpoint, filter by tag + date range
   - `apply_liquidity_floor(markets, min_volume=500_000, min_oi=50_000) -> DataFrame`
   - Cache raw response to `data/raw/markets_{timestamp}.parquet`
4. `ingest/orderbook.py`: pull historical orderbook snapshots at hourly granularity for each market
5. `ingest/trades.py`: pull full trade tape for each market; derive taker side from aggressor flag
6. `ingest/news.py`: GDELT query by market keywords; return `DataFrame` of headlines with timestamps
7. Tests: mock API responses, verify filters and schemas

**MUST USE CONTEXT7**: Before writing any Polymarket API code, use `mcp__plugin_context7_context7__query-docs` for up-to-date endpoint signatures and rate limits. Training data is likely stale.

### Deliverable
Branch mergeable to `main` with:
- Working `python -m polymarket_insider.ingest.markets` CLI that produces a parquet file
- All tests passing
- Week 1 report draft in `reports/week1_data_feasibility.md`

---

## Workstream B — Signal (parallel)

**Worktree branch:** `claude/signal`
**Owner:** subagent B

### Tasks
1. `signal/features.py`:
   - `rolling_hourly_volume(trades, window='7d')`
   - `compute_1h_price_move(orderbook)`
   - `compute_taker_imbalance(trades, window='1h')`
2. `signal/detector.py`:
   - `detect_events(market_features: DataFrame, config: Config) -> list[Event]`
   - Pure function — no I/O, no globals, config-injected
   - Returns typed `Event` dataclasses
3. Tests:
   - Unit tests with hand-crafted fixture data covering: volume spike but no price move (should not fire), price move from news filter (should not fire), clean signal (should fire)
   - Property-based test: signal is deterministic for fixed input

**This workstream can build and test without real data** — uses fixture DataFrames. That's why it runs in parallel with A.

### Deliverable
Branch mergeable to `main` with `detect_events` callable on fixture data, all tests passing, and a reference example in `notebooks/02_signal_exploration.ipynb` scaffolded (empty cells for when real data arrives).

---

## Workstream C — Cost Model + Metrics (parallel)

**Worktree branch:** `claude/backtest-infra`
**Owner:** subagent C

### Tasks
1. `backtest/costs.py`:
   - `apply_fees(pnl, is_resolved)` — 2% on wins, resolved only
   - `apply_spread(entry_price, orderbook)` — pay full spread on entry, half on exit
   - `apply_slippage(size, top_of_book_depth)` — linear, capped
   - `apply_gas()` — fixed $1 round-trip
2. `backtest/ledger.py`:
   - `Trade` dataclass: entry, exit_24h, exit_48h, exit_72h, exit_resolution, all PnL variants
   - `simulate_trade(event, market_data, config) -> Trade`
3. `backtest/control.py`:
   - `generate_control_group(events, universe, n=5)` — random markets + timestamps, volume-quartile matched
4. `report/metrics.py`:
   - Hit rate, mean/median PnL, Sharpe, max drawdown, long-tail concentration
   - Returns a `MetricsSummary` dataclass
5. Tests: cost model with known inputs (pay $5 fee on $250 winnings, etc.); metrics with toy trade lists

**Also parallel-safe** — all of this takes synthetic `Trade` and `Event` objects as input. Doesn't need real data or the real signal.

### Deliverable
Branch mergeable with cost model, ledger, control-group generator, and metrics module all tested.

---

## Workstream D — Backtest Integration (sequenced)

**Worktree branch:** `claude/backtest-run`
**Owner:** main session after A/B/C merge

Runs after A + B + C are merged to `main`.

### Tasks
1. Wire the pipeline end-to-end: markets → orderbooks/trades → features → events → trades → metrics
2. Write `notebooks/03_backtest_first_run.ipynb` following Research Plan Week 3
3. Add CLI: `python -m polymarket_insider.backtest run --window training` → produces a report
4. Iterate per Research Plan Weeks 3–5

---

## Dispatch Instructions (for executing session)

When you're ready to build:

1. Read `docs/PRD.md` and `docs/RESEARCH_PLAN.md` to refresh context
2. Invoke `superpowers:using-git-worktrees` skill
3. Invoke `superpowers:dispatching-parallel-agents` skill
4. Dispatch three subagents in a **single message** with three tool calls:
   - **Agent A:** prompt = full Workstream A section above + "work in branch `claude/ingest` via a worktree, return when all tests pass and week 1 report draft exists"
   - **Agent B:** prompt = full Workstream B section + same instructions for `claude/signal`
   - **Agent C:** prompt = full Workstream C section + same instructions for `claude/backtest-infra`
5. When all three return, review each diff, merge to `main` one at a time, resolving any conflicts in `config.py` or `pyproject.toml`
6. Begin Workstream D in the main session

## Non-Obvious Gotchas

- **Polymarket rate limits:** unknown at time of writing. Workstream A must discover them and document them in week 1 report. Cache aggressively.
- **Orderbook reconstruction:** if Polymarket doesn't expose historical orderbook snapshots directly, we'll need to reconstruct from the trade tape + depth snapshots. This is a *real risk* for Week 1 and may force the Week 1 gate open.
- **News-filter recall:** GDELT is global but its political keyword matching on Polymarket market titles (which can be stylized) will have gaps. Plan manual audit of top-flagged events in Week 2.
- **The holdout must stay sealed.** Put a comment at the top of `notebooks/05_holdout_FINAL.ipynb` saying so, and don't look at it until Week 5. Consider a pre-commit hook that blocks edits to that file until a specific date flag is set.
- **Parquet schemas drift silently.** Use `pyarrow.Schema` equality checks in ingestion tests so a Polymarket API change blows up loudly instead of corrupting downstream.

## Exit criteria for "implementation plan complete"

- A, B, C merged to `main` with green tests
- `notebooks/01_data_feasibility.ipynb` runs clean on real API data
- Week 1 report drafted
- Ready to enter Research Plan Week 2
