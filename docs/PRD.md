# PRD: Polymarket Political Insider-Flow Detection

**Status:** Draft v1
**Owner:** @davidaxie
**Date:** 2026-04-08
**Phase:** Research / Backtest (no live trading)

---

## 1. Problem

Polymarket's political markets occasionally exhibit sudden, large directional inflows that front-run public news — the canonical example being wallet-level activity around the 2024 US election. These flows *may* contain information (insider knowledge, well-sourced political operatives, early wire reports) that has not yet been priced in by the broader market.

If such flows can be identified in near-real-time from public CLOB data, they represent a potential source of trading edge: a follower strategy that enters on detection and exits on either short-horizon mean-revert/continuation or final market resolution.

**We do not yet know if this edge exists, survives trading costs, or is statistically distinguishable from noise.** This project answers that question before any execution system is built.

## 2. Goal

Determine, with statistical rigor, whether a signal based on **volume spikes combined with price/volume divergence** on Polymarket political markets predicts:

- **(a) Short-horizon** price movement (24h / **48h headline** / 72h sensitivity), and
- **(b) Long-horizon** final market resolution,

on markets meeting a strict liquidity floor, **after** realistic trading costs (fees, spread, slippage, gas).

## 3. Non-Goals (Explicit)

- ❌ No live trading, paper or real, in this phase.
- ❌ No wallet-level / on-chain analysis (Polygon tx indexing). Deferred to a possible v2 if price-based signal shows edge.
- ❌ No non-political markets. Crypto, sports, culture are excluded.
- ❌ No US-only restriction — global politics is in scope (foreign elections, geopolitics, policy).
- ❌ No ML models in v1. The signal is rule-based and interpretable. ML is deferred until we know a signal exists worth modeling.

## 4. Signal Definition (v1)

A **suspicious-flow event** fires on a market at time `T` when **all** of the following hold in a rolling 1-hour window ending at `T`:

1. **Liquidity gate (market-level, static):**
   - Cumulative lifetime volume ≥ **$500k**
   - Open interest at `T` ≥ **$50k**
   - Market is tagged `Politics` by Polymarket
   - Market is not within 24h of resolution (avoid settlement noise)

2. **Volume spike:**
   - 1-hour traded volume ≥ **5× the trailing 7-day hourly average** for this market
   - Absolute 1-hour volume ≥ **$25k** (prevents trivial spikes on quiet markets)

3. **Price/volume divergence:**
   - Mid-price move in the 1-hour window ≥ **3 percentage points** (e.g., 42¢ → 45¢ on a YES share)
   - Move is directional (not round-trip): |close − open| ≥ 0.6 × (high − low)
   - Taker-side imbalance: ≥ **70% of volume** crossed the spread in the direction of the price move

4. **News filter (v1 = simple):**
   - No major headline matching the market's keywords published in the 6 hours before `T`
   - Implemented via a headline-feed check (source TBD in research plan; GDELT is the default)

When fired, the event records: `{market_id, timestamp T, direction (YES|NO), entry_mid_price, flagged_volume, z_score}`.

### Rationale
- **5× + $25k floor** filters fake spikes on dead markets and Sybil noise.
- **Directional + taker-imbalance** distinguishes "someone aggressively buying a side" from "two LPs churning inventory."
- **News filter** is the crude version of "could this be public information." It will be imperfect and we will iterate.

## 5. Strategy Under Test (Backtest Only)

For each fired event, simulate entering a **$1,000 notional position** at the prevailing best ask (YES) or best bid (NO) in the signal direction, then:

- **Exit A (short-horizon):** Close at mid-price exactly 48h after entry. Also record PnL at 24h and 72h.
- **Exit B (resolution):** Hold to market resolution. Score PnL at final outcome.

Both exits are scored independently — a single signal produces two PnL rows.

Position sizing is fixed at $1k in v1. No Kelly, no conviction-scaling. Those are v2 questions.

## 6. Success Criteria

The signal "has edge" for purposes of proceeding to a live prototype if **all** of the following hold:

| Metric | Threshold |
|---|---|
| Short-horizon hit rate (48h) | ≥ 55% (vs. 50% random baseline) |
| Short-horizon mean PnL per trade, **after costs** | > $0 with p < 0.05 |
| Long-horizon resolution accuracy (where resolved) | ≥ 60% |
| Signal count over 6 months | ≥ 100 events (statistical power) |
| Sharpe of signal-following strategy | ≥ 1.0 (annualized, conservative) |
| Performance vs. random control group | Outperforms by ≥ 3% mean PnL with p < 0.05 |

If fewer than 100 signals fire in 6 months, we relax signal thresholds and re-run — but document every parameter change to avoid p-hacking.

If the signal fails, we document the failure, consider whether a wallet-level v2 is worth pursuing, and stop.

## 7. Cost Model

Applied to every simulated trade:

- **Polymarket fee:** 2% of winnings (only applies on resolved trades)
- **Spread cost:** Pay full spread on entry (buy ask, sell bid) and half spread on exit (mid)
- **Slippage:** Linear model, `0.5% * (order_size / top_of_book_depth)`, capped at 2%
- **Gas:** Fixed $0.50 per trade (entry + exit = $1.00) — Polygon costs in practice

Costs are deducted per-trade in the PnL ledger. A signal that is profitable pre-cost and unprofitable post-cost is reported as "no edge."

## 8. Data Sources

- **Polymarket CLOB REST API** — markets list, orderbook snapshots, trade history
  - Docs: https://docs.polymarket.com
  - Rate limits: TBD, research plan addresses caching/batching
- **Polymarket Subgraph (Polygon)** — backup for historical trade data if REST doesn't expose enough history
- **GDELT Project** — news headline feed for the news filter (free, global, good political coverage)
- **(Deferred)** On-chain Polygon indexer for wallet analysis — not in v1

## 9. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| 6-month lookback may not contain enough *resolved* markets for long-horizon scoring | Report short-horizon and long-horizon sample sizes separately; long-horizon is advisory in v1 |
| Polymarket REST API historical depth may be insufficient | Subgraph fallback; document API limits in research plan week 1 |
| News filter false negatives (we miss a headline, flag real news as insider) | Manually audit top-20 flagged events for news contamination |
| Signal p-hacking from iterating thresholds | Reserve final 20% of data as holdout, never touched until final report |
| Survivorship bias (only look at markets still accessible via API) | Use subgraph to enumerate all historical markets, including delisted ones |
| Regulatory — CFTC has scrutinized Polymarket | This phase is research only; go/no-go on live trading is a separate decision with legal review |

## 10. Out of Scope for v1 (Candidates for v2)

- Wallet-level analysis ("fresh wallet" / "whale tracking" signals)
- ML models on the flagged events to improve precision
- Auto-execution infrastructure
- Portfolio construction / multi-signal combination
- Non-political markets
- Real-time alerting pipeline

## 11. Decision Gate

At the end of the backtest, we make one of three calls:

- 🟢 **Go:** Signal meets all success criteria → write PRD v2 for a live alerting prototype
- 🟡 **Iterate:** Signal shows edge but fails 1–2 criteria → one round of parameter tuning on non-holdout data, then re-test on holdout
- 🔴 **Stop:** Signal fails → document findings, archive project, decide if wallet-level v2 is worth attempting
