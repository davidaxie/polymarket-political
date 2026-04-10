"""Historical trade tape for Polymarket markets.

Endpoint: ``GET <data_api>/trades?market=<condition_id>`` on the Polymarket
Data API (``https://data-api.polymarket.com``). The CLOB also exposes
``GET /trades`` for authenticated user history, but the Data API variant is
public and returns anonymized trades. Each trade carries a ``side`` (BUY/SELL)
from the *taker's* perspective — this is the "aggressor flag" we use to derive
taker side for the PRD §4.3 imbalance check.

Schema is asserted in tests via ``pyarrow.Schema`` equality so silent API drift
fails loud.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import IngestConfig

log = logging.getLogger(__name__)


TRADES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("trade_id", pa.string()),
        pa.field("market_id", pa.string()),
        pa.field("token_id", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("taker_side", pa.string()),  # 'buy' | 'sell'
        pa.field("outcome", pa.string()),  # 'YES' | 'NO' | ''
    ]
)


def derive_taker_side(trade: dict[str, Any]) -> str:
    """Extract taker side from a Polymarket trade dict.

    The Data API uses ``side`` (``BUY``/``SELL``). Older CLOB responses use
    ``takerSide`` or ``aggressor``. Normalise all of them to ``'buy'``/``'sell'``.
    Unknown inputs return ``''``.
    """
    raw = trade.get("side") or trade.get("takerSide") or trade.get("aggressor") or ""
    s = str(raw).strip().lower()
    if s in ("buy", "bid", "b"):
        return "buy"
    if s in ("sell", "ask", "s"):
        return "sell"
    return ""


def _as_utc_ts(value: Any) -> pd.Timestamp:
    """Coerce naive or tz-aware input to a UTC ``pd.Timestamp``."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, "", 0):
        return None
    try:
        # Polymarket returns unix seconds as int/str.
        if isinstance(value, (int, float)) or (
            isinstance(value, str) and value.isdigit()
        ):
            return pd.Timestamp(int(value), unit="s", tz="UTC")
        return pd.Timestamp(value, tz="UTC")
    except Exception:
        return None


def _row_from_trade(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_id": str(raw.get("id") or raw.get("transactionHash") or ""),
        "market_id": str(raw.get("market") or raw.get("conditionId") or ""),
        "token_id": str(raw.get("asset") or raw.get("token_id") or raw.get("tokenId") or ""),
        "timestamp": _parse_ts(raw.get("timestamp") or raw.get("time")),
        "price": float(raw.get("price") or 0.0),
        "size": float(raw.get("size") or raw.get("amount") or 0.0),
        "taker_side": derive_taker_side(raw),
        "outcome": str(raw.get("outcome") or ""),
    }


def _to_trades_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(
            {name: pd.Series(dtype=object) for name in TRADES_SCHEMA.names}
        )
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    for col in ("price", "size"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0).astype(float)
    for col in ("trade_id", "market_id", "token_id", "taker_side", "outcome"):
        df[col] = df.get(col, "").astype(str)
    return df[
        [
            "trade_id",
            "market_id",
            "token_id",
            "timestamp",
            "price",
            "size",
            "taker_side",
            "outcome",
        ]
    ]


def fetch_trades(
    market_id: str,
    start: datetime | str,
    end: datetime | str,
    *,
    config: IngestConfig | None = None,
    client: httpx.Client | None = None,
    base_url: str = "https://data-api.polymarket.com",
) -> pd.DataFrame:
    """Pull the full trade tape for ``market_id`` between ``start`` and ``end``.

    Returns a frame matching ``TRADES_SCHEMA``.

    IMPORTANT: Polymarket's /trades endpoint caps offset-based pagination at
    3000 records per query. For high-volume markets (Trump 2024 election had
    hundreds of thousands of trades) offset paging alone is insufficient —
    the server returns ``{"error":"max historical activity offset of 3000
    exceeded"}`` with HTTP 400 once you page past it.

    Strategy: fixed-width chunking with cap-tolerant pagination.
    1. Slice ``[start, end]`` into fixed chunks of ``cfg.trades_chunk_seconds``
       (default 30 min).
    2. Paginate each chunk offset 0..3000. If a chunk hits the offset cap,
       **keep the ~3000 trades we got** (a statistically robust sample for
       taker-imbalance and volume-spike detection) and move on to the next
       chunk. No halving — halving creates O(thousands) of redundant HTTP
       calls on ultra-dense markets like Trump 2024 election night.
    3. The ``truncated`` flag on the returned DataFrame (attr) indicates
       whether any chunk was capped.
    """
    cfg = config or IngestConfig()
    owns_client = client is None
    client = client or httpx.Client(base_url=base_url, timeout=cfg.request_timeout_sec)
    start_ts = int(_as_utc_ts(start).timestamp())
    end_ts = int(_as_utc_ts(end).timestamp())

    MAX_OFFSET = 3000  # Polymarket hard cap
    rows: list[dict[str, Any]] = []
    any_truncated = False

    def _drain_chunk(lo_ts: int, hi_ts: int) -> None:
        """Paginate [lo, hi] keeping up to 3000 trades on cap-hit."""
        nonlocal any_truncated
        if hi_ts <= lo_ts:
            return
        offset = 0
        chunk_rows: list[dict[str, Any]] = []
        while True:
            params = {
                "market": market_id,
                "startTs": lo_ts,
                "endTs": hi_ts,
                "limit": cfg.page_size,
                "offset": offset,
            }
            try:
                resp = client.get("/trades", params=params)
            except httpx.ReadTimeout:
                log.warning(
                    "trades: timeout on %s [%d,%d] offset=%d, keeping %d rows",
                    market_id, lo_ts, hi_ts, offset, len(chunk_rows),
                )
                any_truncated = True
                break
            if resp.status_code == 400 and "offset" in resp.text.lower():
                log.info(
                    "trades: offset cap on %s [%d,%d], keeping %d rows",
                    market_id, lo_ts, hi_ts, len(chunk_rows),
                )
                any_truncated = True
                break
            resp.raise_for_status()
            payload = resp.json()
            batch = payload if isinstance(payload, list) else payload.get("data", [])
            if not batch:
                break
            chunk_rows.extend(batch)
            if len(batch) < cfg.page_size:
                break
            offset += cfg.page_size
            if offset >= MAX_OFFSET:
                log.info(
                    "trades: proactive cap on %s [%d,%d], keeping %d rows",
                    market_id, lo_ts, hi_ts, len(chunk_rows),
                )
                any_truncated = True
                break
        rows.extend(chunk_rows)

    try:
        step = max(1, int(cfg.trades_chunk_seconds))
        cursor = start_ts
        while cursor < end_ts:
            nxt = min(cursor + step, end_ts)
            _drain_chunk(cursor, nxt)
            cursor = nxt
    finally:
        if owns_client:
            client.close()

    df = _to_trades_frame(_row_from_trade(r) for r in rows)
    if not df.empty:
        lo = _as_utc_ts(start)
        hi = _as_utc_ts(end)
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].reset_index(
            drop=True
        )
        df = df.drop_duplicates(subset=["trade_id"]).reset_index(drop=True)
    df.attrs["truncated"] = any_truncated
    return df


def _safe_stamp(value: datetime | str) -> str:
    """Compact UTC stamp for building cache filenames (e.g. 20241104T120000Z)."""
    return _as_utc_ts(value).strftime("%Y%m%dT%H%M%SZ")


def cache_trades_parquet(
    df: pd.DataFrame,
    market_id: str,
    start: datetime | str,
    end: datetime | str,
    *,
    config: IngestConfig | None = None,
) -> Path:
    """Write a trades frame to ``data/raw/trades_{market}_{start}_{end}.parquet``.

    Cache key is keyed on ``(market_id, start, end)`` — same window, same cache.
    """
    cfg = config or IngestConfig()
    out_dir = Path(cfg.data_root) / cfg.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"trades_{market_id}_{_safe_stamp(start)}_{_safe_stamp(end)}.parquet"
    )
    table = pa.Table.from_pandas(df, schema=TRADES_SCHEMA, preserve_index=False)
    pq.write_table(table, path)
    return path


def load_or_fetch_trades(
    market_id: str,
    start: datetime | str,
    end: datetime | str,
    *,
    config: IngestConfig | None = None,
    client: httpx.Client | None = None,
    base_url: str = "https://data-api.polymarket.com",
) -> pd.DataFrame:
    """Return a cached parquet if present, else fetch via bisection and cache.

    This is the preferred entry point for downstream pipeline code because
    ``fetch_trades`` can be very expensive on high-volume markets (hundreds of
    HTTP requests via the bisection fallback once Polymarket's 3000-offset cap
    kicks in). The cache turns repeated pipeline runs over the same window
    into a single parquet read.

    NOTE: Cache is keyed only on ``(market_id, start, end)``. For *resolved*
    markets the trade tape is immutable, so this is always safe. For *live*
    markets where ``end`` is close to "now", new trades may have landed since
    the cache was written — delete the parquet to force a refresh. See
    IngestConfig for a future TTL knob if this becomes painful.
    """
    cfg = config or IngestConfig()
    out_dir = Path(cfg.data_root) / cfg.raw_subdir
    path = out_dir / (
        f"trades_{market_id}_{_safe_stamp(start)}_{_safe_stamp(end)}.parquet"
    )
    if path.exists():
        table = pq.read_table(path)
        df = table.to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    df = fetch_trades(
        market_id, start, end, config=cfg, client=client, base_url=base_url
    )
    try:
        cache_trades_parquet(df, market_id, start, end, config=cfg)
    except Exception as exc:  # pragma: no cover — best-effort cache
        log.warning("Failed to cache trades parquet for %s: %s", market_id, exc)
    return df
