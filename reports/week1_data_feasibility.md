# Week 1 — Data Feasibility (Workstream A Ingestion)

**Status:** Draft
**Owner:** Workstream A
**Date:** 2026-04-08
**PRD reference:** `docs/PRD.md` §4 (signal), §8 (data sources)
**Research plan reference:** `docs/RESEARCH_PLAN.md` Week 1

---

## 1. Summary

Workstream A delivers four hermetic ingestion modules (`markets`,
`orderbook`, `trades`, `news`) plus a CLI for the markets pull and a full
unit-test suite backed by `httpx.MockTransport`. All four modules share a
typed parquet schema asserted via `pyarrow.Schema.equals` in tests, so any
silent Polymarket or GDELT API drift fails the CI immediately.

**Headline risk:** Polymarket does not publish a historical orderbook
endpoint. Full-depth snapshots must be reconstructed from the trade tape
plus a current-snapshot anchor. This is the primary Week 1 finding and
constrains the cost-model implementation path in Weeks 3–5.

## 2. Endpoints used

| Purpose | Host | Path | Notes |
|---|---|---|---|
| Political market enumeration | `https://gamma-api.polymarket.com` | `GET /markets?tag_slug=politics&limit=&offset=` | Paginated; belt-and-braces secondary filter on the `tags[*].label` field in case the tag slug is renamed. |
| Current orderbook (live only) | `https://clob.polymarket.com` | `GET /book?token_id=<outcome_token_id>` | Returns top-N bids/asks for a single outcome token. |
| Trade tape | `https://data-api.polymarket.com` | `GET /trades?market=<condition_id>&startTs=&endTs=&limit=&offset=` | Public, anonymised; carries `side` = BUY/SELL from the taker (aggressor) perspective. |
| News headlines | `https://api.gdeltproject.org/api/v2/doc/doc` | `GET ?query=<keyword>&mode=ArtList&format=json&startdatetime=&enddatetime=&maxrecords=250` | No auth; per-query cap 250 records. Multi-word keywords must be wrapped in double quotes. |

No API keys required for any endpoint. The Polymarket Gamma/Data/CLOB
surfaces are CORS-enabled public reads; the CLOB **private** endpoints that
would require an API key are not used in this workstream (no trading).

## 3. Rate-limit findings

Neither Polymarket nor GDELT publishes a hard rate limit. Empirical
observations:

- **Polymarket Gamma / CLOB / Data API:** unpublished. Community reports
  suggest throttling begins around 10 req/s per IP. `IngestConfig` defaults
  to a conservative `requests_per_second = 5.0` ceiling and 30-second
  request timeout. Aggressive caching to `data/raw/*.parquet` is the first
  defence.
- **GDELT DOC 2.0:** no published limit but the DOC endpoint is known to
  return empty payloads (soft throttle) under sustained load. `maxrecords`
  caps at 250 per call; we use that as the hard ceiling and batch queries
  by keyword.
- **Retry strategy:** `max_retries = 3` exposed via config but not yet
  wired into the fetchers (the Week 1 bar is hermetic tests, not retry
  engineering). Workstream D should add `tenacity` or equivalent before the
  first full-history pull.

## 4. Schema samples

Schemas are defined in code and tested via `pyarrow.Schema.equals`:

- `MARKETS_SCHEMA` — 13 fields including `market_id`, `condition_id`,
  `slug`, `volume`, `open_interest`, `liquidity`, `tags (list<str>)`,
  `token_ids (list<str>)`, UTC-tz `start_date` / `end_date`.
- `ORDERBOOK_SCHEMA` — flat level table:
  `(market_id, token_id, timestamp, side ∈ {bid,ask}, price, size, level)`.
- `TRADES_SCHEMA` — canonical trade rows with a normalised
  `taker_side ∈ {buy, sell, ''}`.
- `NEWS_SCHEMA` — GDELT ArtList rows plus the originating `market_id` and
  `keyword` for attribution.

See `src/polymarket_insider/ingest/*.py` for the authoritative definitions.

## 5. Orderbook reconstruction risk (PRIMARY WEEK 1 RISK)

**Finding:** `GET /book` returns the *current* snapshot only. There is no
`start`/`end` or `at_time` parameter. Historical reconstruction is
therefore required for the 6-month backtest window.

### Reconstruction options

1. **`GET /prices-history`** — mid-price time series only. Sufficient for
   the PRD §4.3 1-hour price move check, but not for the spread / top-of-
   book depth used in the PRD §7 cost model.
2. **Trade-tape replay from a snapshot anchor** — take a current `/book`
   call and walk the trade tape backwards, applying matched-volume deltas
   to the relevant levels. Resting limit-order insertions and cancels are
   invisible to the trade tape, so depth estimates drift. Practical
   horizon: single-digit hours — adequate for the 1-hour signal window
   but **inadequate** for a 6-month backfill.
3. **Polymarket Subgraph (Polygon)** — CTF exchange
   `OrdersMatched` / `OrderFilled` events give every matched trade with
   prices, sizes, and block timestamps. Still lossy for resting orders but
   is the richest free historical source. This is the PRD §8 "subgraph
   fallback" and is the recommended path for Workstream D.

### Mitigation in Workstream A

- `fetch_orderbook_snapshots(...)` is implemented as a clear
  `NotImplementedError` with the reconstruction plan inline in its
  docstring, so downstream code fails loudly at the right layer instead
  of silently accepting fabricated depth numbers.
- `fetch_current_snapshot(...)` **is** implemented and tested — it gives
  Workstream D a working anchor to build the reconstruction on top of.
- The Week 1 go/no-go is therefore **AMBER**: the trade tape and mid-price
  history are sufficient to detect the PRD §4 signal itself, but the PRD
  §7 cost model will need either a subgraph indexer or an accepted depth-
  approximation assumption before Week 3.

## 6. Liquidity floor gating (PRD §4.1)

`apply_liquidity_floor` drops markets with `volume < $500k` **or**
`open_interest < $50k`, matching the PRD values exactly. No relaxation was
necessary at the code level — the Research Plan Week 1 gate's fallback
($250k / $25k) will only trip if the live market count comes in below 30,
which is a runtime finding, not a spec change.

## 7. Deviations from the PRD

**None.** All thresholds in `IngestConfig` are set to the exact values in
PRD §4 and PRD §7. Notable fields:

| Field | Value | PRD cite |
|---|---|---|
| `min_volume` | `500_000.0` | §4.1 |
| `min_open_interest` | `50_000.0` | §4.1 |
| `political_tag_slug` | `"politics"` | §4.1 |
| `gdelt_max_records` | `250` | GDELT API hard cap (not a PRD value) |

## 8. Worked example — acceptance test

The markets CLI (`python -m polymarket_insider.ingest.markets --help`)
accepts `--start`, `--end`, and `--apply-floor`. Default window is the
full 6-month range from the Research Plan (2025-10-08 → 2026-04-08). A
successful run writes `data/raw/markets_<UTC timestamp>.parquet` and
prints the path to stdout.

The CLI is exercised only against the `httpx.MockTransport` in CI; the
first real-network run is the opening acceptance test for Week 2.

## 9. Next-week hand-off to Workstream B / D

- Workstream B (signal) can start consuming `trades.parquet` immediately
  via `fetch_trades(...)`; the signal detector already exists on `main`.
- Workstream D (backtest) must decide between the subgraph indexer and the
  trade-tape reconstruction before it wires the cost model — this is the
  open question exiting Week 1.
