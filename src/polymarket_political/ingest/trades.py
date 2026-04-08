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
from typing import Any, Iterable

import httpx
import pandas as pd
import pyarrow as pa

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

    Returns a frame matching ``TRADES_SCHEMA``. Pagination uses the offset
    pattern advertised by the Data API; we stop once the API returns fewer
    records than ``page_size``.
    """
    cfg = config or IngestConfig()
    owns_client = client is None
    client = client or httpx.Client(base_url=base_url, timeout=cfg.request_timeout_sec)
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())

    rows: list[dict[str, Any]] = []
    offset = 0
    try:
        while True:
            params = {
                "market": market_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "limit": cfg.page_size,
                "offset": offset,
            }
            resp = client.get("/trades", params=params)
            resp.raise_for_status()
            payload = resp.json()
            batch = payload if isinstance(payload, list) else payload.get("data", [])
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < cfg.page_size:
                break
            offset += cfg.page_size
    finally:
        if owns_client:
            client.close()

    df = _to_trades_frame(_row_from_trade(r) for r in rows)
    if not df.empty:
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC")
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].reset_index(
            drop=True
        )
    return df
