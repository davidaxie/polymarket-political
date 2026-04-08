# Research Plan: Polymarket Political Insider-Flow Backtest

**Companion to:** `docs/PRD.md`
**Purpose:** Defines *how* we will answer the PRD's core question — does the signal have edge after costs?

---

## Guiding Principles

1. **Fail cheap, fail fast.** Every week has a go/no-go checkpoint. If the data isn't there, we stop before building the rest.
2. **No p-hacking.** The final 20% of the time window (most recent ~5 weeks) is a **holdout** — we do not look at it until the final scoring run. All parameter tuning happens on the 80% training window.
3. **Everything reproducible.** Raw data is cached to disk. Every notebook run produces a dated artifact. No "it worked on my machine."
4. **Short feedback loops.** Each week ends with a plot or a number we can show a human.

---

## Data Windows

- **Full window:** 2025-10-08 → 2026-04-08 (6 months, ending today)
- **Training window (80%):** 2025-10-08 → 2026-03-04 (~21 weeks)
- **Holdout window (20%):** 2026-03-05 → 2026-04-08 (~5 weeks) — **sealed**

## Success Gates (per week)

Each week ends with a concrete artifact and a go/no-go. Missing a gate doesn't kill the project; it forces a scope conversation.

### Week 1 — Data feasibility
**Goal:** Can we actually get the data we need?

- [ ] Enumerate all political markets active during the 6-month window (target: 100–500 markets)
- [ ] Confirm Polymarket REST API provides: market metadata, orderbook snapshots, trade history, resolution outcome
- [ ] Confirm historical trade-level data is available (either REST or subgraph) for at least the training window
- [ ] Count markets passing the $500k/$50k liquidity floor (target: ≥30, ideally ≥50)
- [ ] Pull one sample market end-to-end and compute a trial signal

**Gate:** If fewer than 30 markets pass the liquidity floor, we relax it to $250k/$25k and document the change. If still <30, we stop and reconsider.

**Artifact:** `reports/week1_data_feasibility.md` with market count, data completeness %, and one worked example.

### Week 2 — Signal implementation & sanity check
**Goal:** Run the signal over the training window and look at what it catches.

- [ ] Implement signal per PRD §4 as a pure function `market_data → list[Event]`
- [ ] Run over training window, dump all events to `events.parquet`
- [ ] Target: 80+ events in training window (implies ≥100 total — PRD threshold)
- [ ] Manually inspect 10 random events + the top-5 most extreme — do they look real or like data glitches?
- [ ] Spot-check 3 events against contemporaneous news to validate the news-filter

**Gate:** If <50 events in training window, relax thresholds (document every change) and re-run. If manual inspection shows >30% data glitches, fix ingestion before proceeding.

**Artifact:** `reports/week2_signals.md` with event count, distribution plots, 15 annotated examples.

### Week 3 — Backtest harness & control group
**Goal:** Score both the signal group and a random control group.

- [ ] Build PnL ledger: entry price, exit prices (24h/48h/72h/resolution), raw PnL
- [ ] Apply cost model from PRD §7
- [ ] Generate control group: for every signal event, pick N=5 random markets from the universe at random timestamps in the same week, matched on volume quartile
- [ ] Compute all metrics from PRD §6 for both groups

**Gate:** Signal must *at minimum* be directionally better than control on the 48h metric, pre-cost. If it's not, the thresholds are wrong and Week 4 is parameter sweep instead of iteration.

**Artifact:** `reports/week3_backtest.md` — the first real "does this work" document.

### Week 4 — Iteration (training window only)
**Goal:** One round of parameter tuning. One. Not ten.

- [ ] Sweep: volume-multiple (3×–8×), price-move threshold (2pp–5pp), taker imbalance (60%–80%)
- [ ] For each combination, compute 48h hit rate and mean PnL after costs
- [ ] Pick the parameter set that maximizes mean PnL after costs on the training window
- [ ] Document every parameter tried and every result (anti-p-hacking log)

**Gate:** If the best parameter set still fails PRD success criteria, we move to the "fail" branch of the decision gate — no more tuning.

**Artifact:** `reports/week4_tuning.md` with the full grid and final chosen parameters.

### Week 5 — Holdout test (sealed envelope)
**Goal:** Run the final parameters on the holdout window, exactly once.

- [ ] Verify holdout data has not been touched (git-log the analysis files)
- [ ] Run the Week 4 final parameters on 2026-03-05 → 2026-04-08
- [ ] Compute all PRD success criteria
- [ ] Write the final report **before** looking at any output other than the pass/fail on criteria — no re-runs, no "let me try one more thing"

**Gate:** This is the decision gate from PRD §11. Green / yellow / red.

**Artifact:** `reports/week5_holdout_FINAL.md` — the document we'd show anyone asking "does this work?"

---

## Metrics to Compute

Per event and aggregated:

- **Hit rate** (directional): did the price move our way at each horizon?
- **Mean PnL per trade**, pre- and post-cost
- **Median PnL** (resilient to outliers)
- **Sharpe** (mean / stdev of per-trade PnL, × sqrt(events per year))
- **Max drawdown** on the cumulative PnL curve
- **Long-tail contribution:** what % of total PnL comes from the top 5 trades? (high = fragile)
- **Sample size** at each horizon (short-horizon ≫ long-horizon expected)

Compared against the control group on every metric.

---

## Decision Gates — What Each Outcome Means

| Outcome | Action |
|---|---|
| 🟢 All PRD criteria met on holdout | Archive backtest; start PRD v2 for live alerting prototype |
| 🟡 48h criteria met, long-horizon fails | Ship the short-horizon signal only; long-horizon is a future question |
| 🟡 Training window good, holdout worse but directionally positive | Write "inconclusive" report, propose extended data collection (another 3 months), re-test |
| 🔴 Signal fails on training window | Stop. Write lessons-learned. Consider v2 (wallet-level) as a separate project. |
| 🔴 Signal fails on holdout despite training success | **This is the most important result** — it means the training-window tuning overfit. Write this up clearly. |

## What This Plan is Not

- It is not an execution plan. Parallelization, file structure, and library choices live in `IMPLEMENTATION_PLAN.md`.
- It is not a promise of success. It is a structured way to find out.
