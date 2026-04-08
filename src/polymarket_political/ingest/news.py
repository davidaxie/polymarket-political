"""GDELT 2.0 DOC API headline pulls for the PRD news filter.

Endpoint (confirmed via Context7, April 2026):

    GET https://api.gdeltproject.org/api/v2/doc/doc
        query parameters:
            query=<boolean keyword expression>
            mode=ArtList
            format=json
            maxrecords=<1..250>
            startdatetime=YYYYMMDDHHMMSS
            enddatetime=YYYYMMDDHHMMSS
            sort=DateDesc

No authentication, no published rate limit — but heavy callers are throttled
server-side. Be polite: cache aggressively and avoid hammering on loops.
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


NEWS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("market_id", pa.string()),
        pa.field("keyword", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("title", pa.string()),
        pa.field("url", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("language", pa.string()),
        pa.field("source_country", pa.string()),
    ]
)


def _fmt_dt(value: datetime | str) -> str:
    ts = pd.Timestamp(value, tz="UTC")
    return ts.strftime("%Y%m%d%H%M%S")


def _parse_gdelt_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, "", 0):
        return None
    try:
        return pd.to_datetime(str(value), format="%Y%m%dT%H%M%SZ", utc=True)
    except Exception:
        try:
            return pd.Timestamp(value, tz="UTC")
        except Exception:
            return None


def _row_from_article(
    raw: dict[str, Any], *, market_id: str, keyword: str
) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "keyword": keyword,
        "timestamp": _parse_gdelt_ts(raw.get("seendate")),
        "title": str(raw.get("title") or ""),
        "url": str(raw.get("url") or ""),
        "domain": str(raw.get("domain") or ""),
        "language": str(raw.get("language") or ""),
        "source_country": str(raw.get("sourcecountry") or ""),
    }


def _to_news_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame({name: pd.Series(dtype=object) for name in NEWS_SCHEMA.names})
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    for col in (
        "market_id",
        "keyword",
        "title",
        "url",
        "domain",
        "language",
        "source_country",
    ):
        df[col] = df.get(col, "").astype(str)
    return df[
        [
            "market_id",
            "keyword",
            "timestamp",
            "title",
            "url",
            "domain",
            "language",
            "source_country",
        ]
    ]


def _quote(kw: str) -> str:
    kw = kw.strip()
    if not kw:
        return ""
    if " " in kw and not (kw.startswith('"') and kw.endswith('"')):
        return f'"{kw}"'
    return kw


def fetch_gdelt_headlines(
    keywords: list[str],
    start: datetime | str,
    end: datetime | str,
    *,
    market_id: str = "",
    config: IngestConfig | None = None,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Query GDELT DOC 2.0 for headlines matching ``keywords`` in the window.

    Each keyword is queried independently (GDELT truncates at 250 results per
    query) so the news filter can attribute hits back to the originating term.
    """
    cfg = config or IngestConfig()
    owns_client = client is None
    client = client or httpx.Client(timeout=cfg.request_timeout_sec)
    start_str = _fmt_dt(start)
    end_str = _fmt_dt(end)

    rows: list[dict[str, Any]] = []
    try:
        for kw in keywords:
            q = _quote(kw)
            if not q:
                continue
            original_kw = kw
            params = {
                "query": q,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": cfg.gdelt_max_records,
                "startdatetime": start_str,
                "enddatetime": end_str,
                "sort": "DateDesc",
            }
            resp = client.get(cfg.gdelt_base_url, params=params)
            resp.raise_for_status()
            try:
                payload = resp.json()
            except Exception:
                log.warning("GDELT returned non-JSON for %r", kw)
                continue
            articles = payload.get("articles") or []
            for art in articles:
                rows.append(
                    _row_from_article(art, market_id=market_id, keyword=original_kw)
                )
    finally:
        if owns_client:
            client.close()

    df = _to_news_frame(rows)
    if not df.empty:
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC")
        df = df[(df["timestamp"].isna()) | ((df["timestamp"] >= lo) & (df["timestamp"] <= hi))]
        df = df.reset_index(drop=True)
    return df
