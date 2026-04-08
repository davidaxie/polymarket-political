# Week 3 — First Backtest Run (PROVISIONAL) — TEMPLATE

> ⚠️ **PROVISIONAL DATA — DO NOT CIRCULATE AS A RESULT.**
>
> This backtest uses **Polymarket `/prices-history` mid-price only** (no bid/ask
> depth) and a **fixed-spread cost model** (`ProvisionalCostConfig`: 2¢
> round-trip spread, 0.5¢ per-side slippage, 2% fee, $1 gas). All metrics
> below are labeled:
>
> > *provisional — mid-price only, fixed-spread cost model*
>
> Replace with a depth-aware cost model (Workstream E) before treating any
> number here as a real edge estimate.

---

**This file is a template.** The live report is produced by the Workstream-D
CLI:

```bash
python -m polymarket_political.backtest run \
    --window training \
    --out reports/week3_first_run.md
```

The CLI overwrites this file with a fully-rendered report containing the
sections below. Until then, treat this as a scaffold.

## Sections produced by the CLI

1. **Provisional banner** (always first).
2. **Config snapshot** — JSON dump of the signal/cost/backtest config and the
   resolved `(window, start, end)`.
3. **Universe** — markets after liquidity floor, event count, simulated
   trade counts for signal and control.
4. **Signal metrics** — 48h net PnL `MetricsSummary` (PRD §6 default horizon).
5. **Signal vs Control (two views)** — side-by-side table comparing the signal
   against both the news-filtered control (apples-to-apples) and the raw
   unfiltered control (legacy). A warning is emitted if the two controls
   differ by more than 20% of total PnL, since that indicates news-filter
   asymmetry was masking or unmasking apparent edge.
6. **Control metrics — filtered** — 48h net PnL for the
   volume-quartile-matched random control group after applying the same news
   filter used on the signal group.
7. **Control metrics — raw** — legacy unfiltered control group, kept for
   transparency.
8. **Top-5 events** — biggest `abs_price_move` hits, with `volume_ratio`,
   `imbalance`, and timestamp.
9. **Warnings** — any fetcher failures, skipped events, or empty universe.
10. **Caveats** — reminders about mid-price-only, fixed spread, and the holdout
    being sealed.
11. **Week 3 Gate Recommendation** — 🟢 / 🟡 / 🔴 per the Research Plan.

## Week 3 gate logic (from RESEARCH_PLAN.md)

> "Signal must *at minimum* be directionally better than control on the 48h
> metric, pre-cost. If it's not, the thresholds are wrong and Week 4 is a
> parameter sweep instead of iteration."

The CLI implements this as:

- 🟢 **Green** — signal beats control on **both** mean PnL and hit rate.
- 🟡 **Amber** — signal beats control on **one** of {mean, hit-rate}.
- 🔴 **Red** — signal loses on both, OR no events detected.

## Related artifacts (written under `data/processed/` by default)

- `ledger_training.parquet` — every simulated signal trade.
- `control_ledger_training.parquet` — every simulated control trade.
- `metrics_training.parquet` — `signal`, `control_filtered`, and
  `control_raw` `MetricsSummary` rows.
