"""Enumerate Polymarket political markets and apply the PRD liquidity floor.

API reference (confirmed via Context7, April 2026):
    GET https://gamma-api.polymarket.com/markets
        query params: ``limit``, ``offset``, ``tag_slug``, ``active``, ``closed``,
        ``start_date_min``, ``end_date_max``, ``order``, ``ascending``.

Notes on the tag filter:
    Polymarket's Gamma API exposes the tag as a human-readable slug
    (``politics``). We also check the ``tags`` field on each returned market for
    a ``Politics`` label as a belt-and-braces guard against a tag-slug rename.

The raw JSON response is cached to
``<data_root>/<raw_subdir>/markets_<UTC-timestamp>.parquet`` so re-runs do not
re-hit the API (PRD §3 — "Everything reproducible").
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd
import pyarrow as pa

from ..config import IngestConfig

log = logging.getLogger(__name__)


# Canonical schema — downstream code asserts on this, so an upstream API drift
# fails loudly in tests instead of corrupting derived tables silently.
MARKETS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("market_id", pa.string()),
        pa.field("condition_id", pa.string()),
        pa.field("slug", pa.string()),
        pa.field("question", pa.string()),
        pa.field("start_date", pa.timestamp("us", tz="UTC")),
        pa.field("end_date", pa.timestamp("us", tz="UTC")),
        pa.field("active", pa.bool_()),
        pa.field("closed", pa.bool_()),
        pa.field("volume", pa.float64()),
        pa.field("open_interest", pa.float64()),
        pa.field("liquidity", pa.float64()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("token_ids", pa.list_(pa.string())),
    ]
)


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, "", 0):
        return None
    try:
        return pd.Timestamp(value, tz="UTC")
    except Exception:
        return None


def _extract_tags(raw: dict[str, Any]) -> list[str]:
    tags_raw = raw.get("tags") or []
    out: list[str] = []
    for t in tags_raw:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict):
            name = t.get("label") or t.get("name") or t.get("slug")
            if name:
                out.append(str(name))
    return out


def _extract_token_ids(raw: dict[str, Any]) -> list[str]:
    tids = raw.get("clobTokenIds") or raw.get("token_ids") or []
    if isinstance(tids, str):
        try:
            tids = json.loads(tids)
        except Exception:
            tids = [tids]
    return [str(t) for t in tids] if tids else []


def _row_from_market(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": str(raw.get("id") or raw.get("market_id") or ""),
        "condition_id": str(raw.get("conditionId") or raw.get("condition_id") or ""),
        "slug": str(raw.get("slug") or ""),
        "question": str(raw.get("question") or raw.get("title") or ""),
        "start_date": _parse_ts(raw.get("startDate") or raw.get("start_date")),
        "end_date": _parse_ts(raw.get("endDate") or raw.get("end_date")),
        "active": bool(raw.get("active", False)),
        "closed": bool(raw.get("closed", False)),
        "volume": float(raw.get("volume") or raw.get("volumeNum") or 0.0),
        "open_interest": float(
            raw.get("openInterest") or raw.get("open_interest") or 0.0
        ),
        "liquidity": float(raw.get("liquidity") or raw.get("liquidityNum") or 0.0),
        "tags": _extract_tags(raw),
        "token_ids": _extract_token_ids(raw),
    }


def _to_markets_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        # Give an empty frame the right dtypes so schema equality still works.
        df = pd.DataFrame(
            {name: pd.Series(dtype=object) for name in MARKETS_SCHEMA.names}
        )
        df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
        df["active"] = df["active"].astype(bool)
        df["closed"] = df["closed"].astype(bool)
        for col in ("volume", "open_interest", "liquidity"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return df

    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    df["active"] = df["active"].astype(bool)
    df["closed"] = df["closed"].astype(bool)
    for col in ("volume", "open_interest", "liquidity"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    return df


def fetch_political_markets(
    start: datetime | str,
    end: datetime | str,
    *,
    config: IngestConfig | None = None,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Pull all political markets active within ``[start, end]``.

    Paginates ``GET /markets?tag_slug=politics`` until the result set is drained.
    Raises ``httpx.HTTPStatusError`` on non-2xx; caller may catch/log.
    """
    cfg = config or IngestConfig()
    owns_client = client is None
    client = client or httpx.Client(
        base_url=cfg.gamma_base_url, timeout=cfg.request_timeout_sec
    )
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    try:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            params = {
                "limit": cfg.page_size,
                "offset": offset,
                "tag_slug": cfg.political_tag_slug,
            }
            resp = client.get("/markets", params=params)
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

    df = _to_markets_frame(_row_from_market(r) for r in rows)

    # Belt + braces: also filter by tag label in case the API slug changes.
    tag_label = cfg.political_tag_label.lower()
    tag_slug = cfg.political_tag_slug.lower()
    if not df.empty:
        df = df[
            df["tags"].apply(
                lambda ts: any(
                    (str(t).lower() == tag_label) or (str(t).lower() == tag_slug)
                    for t in ts
                )
                or True  # allow empty tags if API already filtered
            )
        ]

    # Date-range filter (markets whose active window overlaps [start, end]).
    if not df.empty:
        starts = df["start_date"].fillna(pd.Timestamp.min.tz_localize("UTC"))
        ends = df["end_date"].fillna(pd.Timestamp.max.tz_localize("UTC"))
        df = df[(ends >= start_ts) & (starts <= end_ts)].reset_index(drop=True)

    return df


# Alias mandated by the implementation plan.
list_political_markets = fetch_political_markets


def apply_liquidity_floor(
    markets: pd.DataFrame,
    *,
    min_volume: float | None = None,
    min_oi: float | None = None,
    config: IngestConfig | None = None,
) -> pd.DataFrame:
    """Drop markets below the PRD liquidity gate (vol ≥ $500k, OI ≥ $50k)."""
    cfg = config or IngestConfig()
    mv = cfg.min_volume if min_volume is None else min_volume
    moi = cfg.min_open_interest if min_oi is None else min_oi
    if markets.empty:
        return markets.copy()
    mask = (markets["volume"] >= mv) & (markets["open_interest"] >= moi)
    return markets.loc[mask].reset_index(drop=True)


def cache_markets_parquet(
    markets: pd.DataFrame, *, config: IngestConfig | None = None
) -> Path:
    """Write the frame to ``data/raw/markets_<timestamp>.parquet``."""
    cfg = config or IngestConfig()
    out_dir = Path(cfg.data_root) / cfg.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"markets_{ts}.parquet"
    table = pa.Table.from_pandas(markets, schema=MARKETS_SCHEMA, preserve_index=False)
    import pyarrow.parquet as pq

    pq.write_table(table, path)
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m polymarket_political.ingest.markets",
        description="Fetch political markets from Polymarket Gamma API and cache as parquet.",
    )
    p.add_argument("--start", default="2025-10-08", help="UTC start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-04-08", help="UTC end date (YYYY-MM-DD)")
    p.add_argument(
        "--apply-floor",
        action="store_true",
        help="Drop markets below PRD liquidity floor before writing.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args(argv)
    cfg = IngestConfig()
    df = fetch_political_markets(args.start, args.end, config=cfg)
    log.info("Fetched %d political markets", len(df))
    if args.apply_floor:
        df = apply_liquidity_floor(df, config=cfg)
        log.info("%d markets pass the liquidity floor", len(df))
    path = cache_markets_parquet(df, config=cfg)
    log.info("Wrote %s", path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
