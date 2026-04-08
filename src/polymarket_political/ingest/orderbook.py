"""Historical orderbook snapshots for Polymarket markets.

⚠️  RECONSTRUCTION RISK — see ``reports/week1_data_feasibility.md``.

The Polymarket CLOB REST API exposes a *current* orderbook snapshot via
``GET /book?token_id=<outcome_token_id>`` but **does not** publish a direct
historical snapshot endpoint. The documented alternatives are:

1.  **Price history** (``GET /prices-history?market=<token>&interval=...``) —
    mid-price time series only. Sufficient for the 1-hour mid-price move in the
    signal definition, but *not* for spread / depth / top-of-book-depth used by
    the cost model.

2.  **Reconstruction from the trade tape + snapshot anchor** — start from a
    known snapshot (most recent ``/book`` call), then replay the trade tape
    backwards in time. This is lossy: resting limit-order changes are not on
    the tape, so depth numbers drift the further back you go. Practical horizon
    is on the order of a few hours, which is adequate for the 1-hour signal
    window but NOT for 6 months of backfill.

3.  **Polymarket Subgraph (Polygon)** — the ``OrdersMatched`` / ``OrderFilled``
    events on the CTF exchange contract give every matched order with full
    prices, sizes, and block timestamps. Reconstruction is still lossy for
    resting orders, but is the richest free source. This is the PRD §8
    "subgraph fallback."

This module therefore:

* Implements the **current-snapshot** fetch against ``/book`` (working).
* Exposes ``fetch_orderbook_snapshots(...)`` as the historical path, which
  raises ``NotImplementedError`` with the documented reconstruction plan in the
  docstring so Workstream D knows what to do before it wires the backtest.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable

import httpx
import pandas as pd
import pyarrow as pa

from ..config import IngestConfig

log = logging.getLogger(__name__)


ORDERBOOK_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("market_id", pa.string()),
        pa.field("token_id", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("side", pa.string()),  # 'bid' | 'ask'
        pa.field("price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("level", pa.int32()),  # 0 = top of book
    ]
)


def _levels_to_rows(
    levels: Iterable[dict[str, Any]],
    *,
    market_id: str,
    token_id: str,
    timestamp: pd.Timestamp,
    side: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, lvl in enumerate(levels):
        price = lvl.get("price")
        size = lvl.get("size") or lvl.get("quantity")
        if price is None or size is None:
            continue
        out.append(
            {
                "market_id": market_id,
                "token_id": token_id,
                "timestamp": timestamp,
                "side": side,
                "price": float(price),
                "size": float(size),
                "level": i,
            }
        )
    return out


def fetch_current_snapshot(
    token_id: str,
    *,
    market_id: str | None = None,
    config: IngestConfig | None = None,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Fetch the current orderbook snapshot for one outcome token.

    Uses ``GET <clob_base>/book?token_id=<id>``. This is the only endpoint
    Polymarket exposes for live depth; historical snapshots must be
    reconstructed (see module docstring / Week 1 report).
    """
    cfg = config or IngestConfig()
    owns_client = client is None
    client = client or httpx.Client(
        base_url=cfg.clob_base_url, timeout=cfg.request_timeout_sec
    )
    try:
        resp = client.get("/book", params={"token_id": token_id})
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns_client:
            client.close()

    ts = pd.Timestamp.now(tz="UTC")
    mid = market_id or str(payload.get("market") or "")
    tid = str(payload.get("asset_id") or payload.get("token_id") or token_id)

    # Polymarket returns bids lowest→highest and asks lowest→highest; callers
    # typically want best-first, so reverse bids to descend.
    bids = list(reversed(payload.get("bids") or []))
    asks = list(payload.get("asks") or [])
    rows = _levels_to_rows(
        bids, market_id=mid, token_id=tid, timestamp=ts, side="bid"
    ) + _levels_to_rows(
        asks, market_id=mid, token_id=tid, timestamp=ts, side="ask"
    )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            {name: pd.Series(dtype=object) for name in ORDERBOOK_SCHEMA.names}
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["level"] = df["level"].astype("int32")
    return df


def fetch_orderbook_snapshots(
    token_id: str,
    start: datetime | str,
    end: datetime | str,
    *,
    frequency: str = "1h",
    config: IngestConfig | None = None,
) -> pd.DataFrame:
    """Historical orderbook snapshots at ``frequency`` granularity.

    **Not implemented.** Polymarket's public API does not expose historical
    full-depth snapshots. See module docstring for the three reconstruction
    paths. The recommended path for Week 2 is:

    1. Use ``ingest.trades.fetch_trades`` to pull the full trade tape for the
       market.
    2. Use ``GET /prices-history`` to pull the mid-price time series at the
       requested frequency.
    3. Anchor depth with ``fetch_current_snapshot`` at ``end`` and walk
       backwards, applying trade-tape deltas. Document the recon error band in
       the Week 1 report (drift accelerates after ~6h).

    This stub exists so downstream modules fail loudly at the right layer
    instead of silently accepting fabricated data.
    """
    raise NotImplementedError(
        "Polymarket does not publish historical orderbook snapshots. "
        "Reconstruct from /prices-history + /data/trades + a current /book "
        "anchor — see ingest.orderbook module docstring and Week 1 report."
    )
