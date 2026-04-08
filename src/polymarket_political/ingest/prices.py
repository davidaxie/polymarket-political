"""Historical price series from Polymarket's CLOB ``/prices-history`` endpoint.

API reference (confirmed via Context7, April 2026 — see
``docs/RESEARCH_PLAN.md`` Week 3 note on the Option-1 mid-price path):

    GET https://clob.polymarket.com/prices-history
        query params:
            - market     (required) — asset/token id (NOT condition_id)
            - startTs    (optional) — unix seconds
            - endTs      (optional) — unix seconds
            - interval   (optional) — one of {max, all, 1m, 1w, 1d, 6h, 1h}
            - fidelity   (optional) — minutes per sample
        response: {"history": [{"t": <unix_seconds>, "p": <float>}, ...]}

This endpoint returns a best-available mid-price series for a single outcome
token. We deliberately use only the mid for the Week-3 backtest (Option 1 in
the research plan): the insider-flow signal detector only needs price moves,
and any real bid/ask depth would require the Polygon subgraph (Workstream E).
ALL downstream metrics derived from this module MUST be labeled
"provisional — mid-price only".

Schema is enforced via ``pyarrow.Schema`` equality so API drift fails loud.
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


PRICES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("market_id", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("mid_price", pa.float64()),
    ]
)


_ALLOWED_INTERVALS = {"max", "all", "1m", "1w", "1d", "6h", "1h"}


def _row_from_point(market_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    t_val = raw.get("t") or raw.get("timestamp") or raw.get("time")
    p_val = raw.get("p") if "p" in raw else raw.get("price")
    ts: pd.Timestamp | None = None
    if t_val not in (None, ""):
        try:
            ts = pd.Timestamp(int(t_val), unit="s", tz="UTC")
        except Exception:
            try:
                ts = pd.Timestamp(t_val, tz="UTC")
            except Exception:
                ts = None
    try:
        price = float(p_val) if p_val is not None else float("nan")
    except Exception:
        price = float("nan")
    return {"market_id": market_id, "timestamp": ts, "mid_price": price}


def _to_prices_frame(
    rows: Iterable[dict[str, Any]], market_id: str
) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(
            {name: pd.Series(dtype=object) for name in PRICES_SCHEMA.names}
        )
    df["market_id"] = df.get("market_id", market_id).astype(str)
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    df["mid_price"] = pd.to_numeric(
        df.get("mid_price"), errors="coerce"
    ).astype(float)
    df = df.dropna(subset=["timestamp", "mid_price"]).reset_index(drop=True)
    return df[["market_id", "timestamp", "mid_price"]]


def fetch_price_history(
    market_id: str,
    start: datetime | str,
    end: datetime | str,
    interval: str = "1h",
    *,
    fidelity: int | None = None,
    config: IngestConfig | None = None,
    client: httpx.Client | None = None,
    base_url: str | None = None,
) -> pd.DataFrame:
    """Pull the mid-price series for ``market_id`` between ``start`` and ``end``.

    Parameters
    ----------
    market_id:
        Polymarket asset/token id (clob token id). NOTE: The ``/prices-history``
        endpoint expects the outcome token id, NOT the condition id. The
        pipeline module is responsible for picking the correct YES-token.
    start, end:
        Inclusive bounds; accepted as ISO strings or ``datetime`` / pandas
        ``Timestamp``.
    interval:
        One of ``{max, all, 1m, 1w, 1d, 6h, 1h}``. Default ``'1h'`` so the
        resulting series lines up with the PRD §4.3 1-hour windowing.
    fidelity:
        Optional — sample spacing in minutes. Leave ``None`` to let Polymarket
        pick (the API default is 1 minute).

    Returns
    -------
    DataFrame matching :data:`PRICES_SCHEMA` with columns
    ``[market_id, timestamp, mid_price]``.
    """
    if interval not in _ALLOWED_INTERVALS:
        raise ValueError(
            f"interval must be one of {_ALLOWED_INTERVALS}, got {interval!r}"
        )

    cfg = config or IngestConfig()
    owns_client = client is None
    url = base_url or cfg.clob_base_url
    client = client or httpx.Client(base_url=url, timeout=cfg.request_timeout_sec)

    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())

    params: dict[str, Any] = {
        "market": market_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "interval": interval,
    }
    if fidelity is not None:
        params["fidelity"] = int(fidelity)

    try:
        resp = client.get("/prices-history", params=params)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns_client:
            client.close()

    history = payload.get("history", []) if isinstance(payload, dict) else []
    df = _to_prices_frame((_row_from_point(market_id, p) for p in history), market_id)

    if not df.empty:
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC")
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].reset_index(
            drop=True
        )
    return df


def _safe_stamp(value: datetime | str) -> str:
    return pd.Timestamp(value, tz="UTC").strftime("%Y%m%dT%H%M%SZ")


def cache_price_history_parquet(
    df: pd.DataFrame,
    market_id: str,
    start: datetime | str,
    end: datetime | str,
    *,
    config: IngestConfig | None = None,
) -> Path:
    """Write a prices frame to ``data/raw/prices_{market}_{start}_{end}.parquet``."""
    cfg = config or IngestConfig()
    out_dir = Path(cfg.data_root) / cfg.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"prices_{market_id}_{_safe_stamp(start)}_{_safe_stamp(end)}.parquet"
    )
    table = pa.Table.from_pandas(df, schema=PRICES_SCHEMA, preserve_index=False)
    pq.write_table(table, path)
    return path


def load_or_fetch_price_history(
    market_id: str,
    start: datetime | str,
    end: datetime | str,
    *,
    interval: str = "1h",
    config: IngestConfig | None = None,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Return a cached parquet if present, else fetch and cache it.

    Used by the pipeline to keep re-runs cheap. Cache key is exactly
    ``prices_{market_id}_{start}_{end}.parquet`` so re-running with the same
    window returns the same file.
    """
    cfg = config or IngestConfig()
    out_dir = Path(cfg.data_root) / cfg.raw_subdir
    path = out_dir / (
        f"prices_{market_id}_{_safe_stamp(start)}_{_safe_stamp(end)}.parquet"
    )
    if path.exists():
        table = pq.read_table(path)
        df = table.to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    df = fetch_price_history(
        market_id, start, end, interval=interval, config=cfg, client=client
    )
    try:
        cache_price_history_parquet(df, market_id, start, end, config=cfg)
    except Exception as exc:  # pragma: no cover — best-effort cache
        log.warning("Failed to cache prices parquet for %s: %s", market_id, exc)
    return df
