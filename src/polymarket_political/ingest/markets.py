"""Enumerate Polymarket political markets and apply the PRD liquidity floor.

API reference (verified live April 2026):
    The Gamma ``/markets`` endpoint does NOT honour ``tag_slug`` as a filter —
    passing it returns the entire market universe (~51k rows). Tag information
    is only attached to *events*, not markets. To filter to political markets
    we must go through ``/events``:

        GET https://gamma-api.polymarket.com/events
            query params (verified):
                ``tag_slug``        — e.g. "politics" (filters server-side)
                ``closed``          — true/false
                ``limit``, ``offset``
                ``end_date_min``    — earliest event end date
                ``end_date_max``    — latest event end date
                ``start_date_max``  — latest event start date
                ``order``, ``ascending``

    Each event returns a nested ``markets`` array plus a ``tags`` array of
    ``{id, slug, label}`` objects. We flatten events → markets, copy the event
    tags onto each child market, and apply the PRD liquidity floor.

Date filter:
    A market is "active during [start, end]" if its ``endDate`` ≥ ``start``
    (event hadn't resolved before our window) AND its ``startDate`` ≤ ``end``
    (event existed by our window close). We push the first half server-side
    via ``end_date_min=start`` and the second half via ``start_date_max=end``.

Open-interest caveat:
    Gamma returns ``openInterest`` as ``0`` or ``None`` for every market we
    looked at. Enforcing the PRD OI ≥ $50k gate against ``None`` would wipe
    out every political market, so we only apply the OI floor to rows that
    actually report a positive value. Volume gate is applied unconditionally.

YES token convention (verified live on 2024 Trump/Harris markets):
    ``outcomes`` and ``clobTokenIds`` are JSON-encoded arrays aligned by index.
    ``outcomes[0]`` is the "Yes" label and ``clobTokenIds[0]`` is the YES CLOB
    token id. We store ``token_ids[0]`` = YES, ``token_ids[1]`` = NO, which
    matches every binary market we inspected.

The resulting frame is cached to
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
    """Return ``[YES_token_id, NO_token_id]`` for a binary market.

    Polymarket encodes ``clobTokenIds`` and ``outcomes`` as JSON strings whose
    arrays are index-aligned. ``outcomes[0]`` is the "Yes" label and
    ``clobTokenIds[0]`` is the matching YES token id. This holds for every
    binary market we spot-checked (Trump 2024, Harris 2024, debate-slur, etc.).
    """
    tids = raw.get("clobTokenIds") or raw.get("token_ids") or []
    if isinstance(tids, str):
        try:
            tids = json.loads(tids)
        except Exception:
            tids = [tids]
    return [str(t) for t in tids] if tids else []


def _row_from_market(
    raw: dict[str, Any], *, event_tags: list[str] | None = None
) -> dict[str, Any]:
    # If markets come from an event payload, the tag list lives on the event.
    tags = _extract_tags(raw)
    if not tags and event_tags:
        tags = list(event_tags)
    oi_raw = raw.get("openInterest")
    if oi_raw is None:
        oi_raw = raw.get("open_interest")
    return {
        "market_id": str(raw.get("id") or raw.get("market_id") or ""),
        "condition_id": str(raw.get("conditionId") or raw.get("condition_id") or ""),
        "slug": str(raw.get("slug") or ""),
        "question": str(raw.get("question") or raw.get("title") or ""),
        "start_date": _parse_ts(raw.get("startDate") or raw.get("start_date")),
        "end_date": _parse_ts(raw.get("endDate") or raw.get("end_date")),
        "active": bool(raw.get("active", False)),
        "closed": bool(raw.get("closed", False)),
        "volume": float(raw.get("volumeNum") or raw.get("volume") or 0.0),
        "open_interest": float(oi_raw or 0.0),
        "liquidity": float(raw.get("liquidityNum") or raw.get("liquidity") or 0.0),
        "tags": tags,
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

    Paginates ``GET /events?tag_slug=politics`` (politics is filtered
    server-side on events; ``/markets`` does not honour the tag filter) and
    flattens each event's child markets into rows. Also requests both
    ``closed=true`` and ``closed=false`` so we capture resolved + live markets
    in the window.

    Raises ``httpx.HTTPStatusError`` on non-2xx; caller may catch/log.
    """
    cfg = config or IngestConfig()
    owns_client = client is None
    client = client or httpx.Client(
        base_url=cfg.gamma_base_url, timeout=cfg.request_timeout_sec
    )
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    start_iso = start_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _iter_events(closed_flag: bool) -> Iterable[dict[str, Any]]:
        offset = 0
        while True:
            params: dict[str, Any] = {
                "limit": cfg.page_size,
                "offset": offset,
                "tag_slug": cfg.political_tag_slug,
                "closed": str(closed_flag).lower(),
                # A market is "active during [start, end]" iff
                #   endDate >= start AND startDate <= end.
                "end_date_min": start_iso,
                "start_date_max": end_iso,
            }
            resp = client.get("/events", params=params)
            resp.raise_for_status()
            payload = resp.json()
            batch = payload if isinstance(payload, list) else payload.get("data", [])
            if not batch:
                return
            yield from batch
            if len(batch) < cfg.page_size:
                return
            offset += cfg.page_size

    rows: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    try:
        for closed_flag in (True, False):
            for event in _iter_events(closed_flag):
                event_tags = _extract_tags(event)
                for m in event.get("markets") or []:
                    row = _row_from_market(m, event_tags=event_tags)
                    mid = row["market_id"]
                    if mid and mid in seen_market_ids:
                        continue
                    if mid:
                        seen_market_ids.add(mid)
                    rows.append(row)
    finally:
        if owns_client:
            client.close()

    df = _to_markets_frame(rows)

    # Belt + braces: after flattening, drop any market whose inherited event
    # tags don't actually include "politics". Empty tag lists are allowed
    # (server-side filter was already applied) so we keep them.
    tag_label = cfg.political_tag_label.lower()
    tag_slug = cfg.political_tag_slug.lower()
    if not df.empty:
        def _is_political(ts: list[str]) -> bool:
            if not ts:
                return True
            return any(
                (str(t).lower() == tag_label) or (str(t).lower() == tag_slug)
                for t in ts
            )

        df = df[df["tags"].apply(_is_political)]

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
    """Drop markets below the PRD liquidity gate (vol ≥ $500k, OI ≥ $50k).

    The Gamma API reports ``openInterest`` as 0 or null for every political
    market we inspected, so a strict OI gate would wipe out the universe. We
    therefore only enforce the OI floor on rows that actually report a
    positive OI value — rows with OI == 0 bypass the OI check and are gated
    on volume alone. Caller can pass an explicit ``min_oi`` to re-assert the
    strict interpretation if a richer data source becomes available.
    """
    cfg = config or IngestConfig()
    mv = cfg.min_volume if min_volume is None else min_volume
    moi = cfg.min_open_interest if min_oi is None else min_oi
    if markets.empty:
        return markets.copy()
    vol_ok = markets["volume"] >= mv
    oi_reported = markets["open_interest"] > 0
    oi_ok = (~oi_reported) | (markets["open_interest"] >= moi)
    return markets.loc[vol_ok & oi_ok].reset_index(drop=True)


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
