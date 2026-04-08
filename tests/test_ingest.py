"""Unit tests for Workstream A — ingestion.

Uses ``httpx.MockTransport`` to stub API calls so tests are hermetic. Schema
equality is asserted on every fetcher output via ``pyarrow.Schema.equals`` so
upstream API drift fails loud instead of silently corrupting downstream tables.
"""
from __future__ import annotations

import json

import httpx
import pandas as pd
import pyarrow as pa
import pytest

from polymarket_insider.config import IngestConfig
from polymarket_insider.ingest import (
    MARKETS_SCHEMA,
    NEWS_SCHEMA,
    ORDERBOOK_SCHEMA,
    TRADES_SCHEMA,
    apply_liquidity_floor,
    derive_taker_side,
    fetch_gdelt_headlines,
    fetch_orderbook_snapshots,
    fetch_political_markets,
    fetch_trades,
)
from polymarket_insider.ingest.orderbook import fetch_current_snapshot


# -------------------------- markets -----------------------------------------


def _fake_market(
    *,
    mid: str,
    vol: float,
    oi: float,
    start: str = "2025-11-01T00:00:00Z",
    end: str = "2026-02-01T00:00:00Z",
    tag: str = "Politics",
) -> dict:
    return {
        "id": mid,
        "conditionId": f"0x{mid}",
        "slug": f"market-{mid}",
        "question": f"Will thing {mid} happen?",
        "startDate": start,
        "endDate": end,
        "active": True,
        "closed": False,
        "volume": vol,
        "openInterest": oi,
        "liquidity": 12345.6,
        "tags": [{"label": tag, "slug": tag.lower()}],
        "clobTokenIds": json.dumps([f"tok-{mid}-yes", f"tok-{mid}-no"]),
    }


def test_fetch_political_markets_filters_and_schema():
    page1 = [_fake_market(mid="A", vol=600_000, oi=75_000)] * 2
    page2 = [_fake_market(mid="B", vol=100_000, oi=10_000)]
    responses = iter([page1, page2, []])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets"
        assert request.url.params.get("tag_slug") == "politics"
        return httpx.Response(200, json=next(responses))

    transport = httpx.MockTransport(handler)
    cfg = IngestConfig(page_size=2)
    with httpx.Client(
        transport=transport, base_url=cfg.gamma_base_url
    ) as client:
        df = fetch_political_markets(
            "2025-10-08", "2026-04-08", config=cfg, client=client
        )

    assert len(df) == 3
    table = pa.Table.from_pandas(df, schema=MARKETS_SCHEMA, preserve_index=False)
    assert table.schema.equals(MARKETS_SCHEMA)


def test_apply_liquidity_floor_drops_below_thresholds():
    cfg = IngestConfig()
    df = pd.DataFrame(
        {
            "market_id": ["A", "B", "C"],
            "volume": [600_000.0, 100_000.0, 500_000.0],
            "open_interest": [75_000.0, 80_000.0, 49_999.0],
        }
    )
    # Fill required columns to satisfy downstream code even though we only
    # assert on volume/OI here.
    for col in (
        "condition_id",
        "slug",
        "question",
        "active",
        "closed",
        "liquidity",
        "tags",
        "token_ids",
    ):
        df[col] = "" if col in ("condition_id", "slug", "question") else 0
    df["active"] = True
    df["closed"] = False
    df["tags"] = [[], [], []]
    df["token_ids"] = [[], [], []]
    df["start_date"] = pd.Timestamp("2025-11-01", tz="UTC")
    df["end_date"] = pd.Timestamp("2026-02-01", tz="UTC")

    kept = apply_liquidity_floor(df, config=cfg)
    assert list(kept["market_id"]) == ["A"]


# -------------------------- orderbook ---------------------------------------


def test_fetch_current_snapshot_schema():
    payload = {
        "market": "0xabc",
        "asset_id": "tok-1",
        "bids": [
            {"price": "0.41", "size": "100"},
            {"price": "0.42", "size": "200"},
        ],
        "asks": [
            {"price": "0.43", "size": "150"},
            {"price": "0.44", "size": "50"},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/book"
        assert request.url.params.get("token_id") == "tok-1"
        return httpx.Response(200, json=payload)

    cfg = IngestConfig()
    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url=cfg.clob_base_url
    ) as client:
        df = fetch_current_snapshot("tok-1", config=cfg, client=client)

    assert len(df) == 4
    # bid side should be sorted best-first (highest price first).
    bids = df[df["side"] == "bid"].reset_index(drop=True)
    assert bids.loc[0, "price"] == 0.42
    table = pa.Table.from_pandas(df, schema=ORDERBOOK_SCHEMA, preserve_index=False)
    assert table.schema.equals(ORDERBOOK_SCHEMA)


def test_fetch_orderbook_snapshots_is_not_implemented():
    with pytest.raises(NotImplementedError, match="(?i)reconstruct"):
        fetch_orderbook_snapshots("tok-1", "2025-11-01", "2025-11-02")


# -------------------------- trades ------------------------------------------


def test_fetch_trades_schema_and_taker_side():
    trades = [
        {
            "id": "t1",
            "market": "0xabc",
            "asset": "tok-1",
            "timestamp": 1700000100,
            "price": 0.42,
            "size": 500,
            "side": "BUY",
            "outcome": "YES",
        },
        {
            "id": "t2",
            "market": "0xabc",
            "asset": "tok-1",
            "timestamp": 1700000200,
            "price": 0.43,
            "size": 250,
            "side": "SELL",
            "outcome": "YES",
        },
    ]

    responses = iter([trades, []])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trades"
        assert request.url.params.get("market") == "0xabc"
        return httpx.Response(200, json=next(responses))

    cfg = IngestConfig(page_size=2)
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://data-api.polymarket.com",
    ) as client:
        df = fetch_trades(
            "0xabc",
            "2023-11-14T00:00:00Z",
            "2023-11-16T00:00:00Z",
            config=cfg,
            client=client,
        )

    assert list(df["taker_side"]) == ["buy", "sell"]
    assert df["timestamp"].dt.tz is not None
    table = pa.Table.from_pandas(df, schema=TRADES_SCHEMA, preserve_index=False)
    assert table.schema.equals(TRADES_SCHEMA)


def test_derive_taker_side_variants():
    assert derive_taker_side({"side": "BUY"}) == "buy"
    assert derive_taker_side({"takerSide": "sell"}) == "sell"
    assert derive_taker_side({"aggressor": "bid"}) == "buy"
    assert derive_taker_side({"aggressor": "ask"}) == "sell"
    assert derive_taker_side({"side": "???"}) == ""
    assert derive_taker_side({}) == ""


# -------------------------- news / GDELT ------------------------------------


def test_fetch_gdelt_headlines_maps_articles_and_keyword():
    payload = {
        "articles": [
            {
                "url": "https://example.com/1",
                "title": "Election headline",
                "seendate": "20251115T120000Z",
                "domain": "example.com",
                "language": "English",
                "sourcecountry": "United States",
            },
            {
                "url": "https://example.com/2",
                "title": "Another headline",
                "seendate": "20251116T090000Z",
                "domain": "example.com",
                "language": "English",
                "sourcecountry": "United States",
            },
        ]
    }
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.url.params))
        assert request.url.params.get("mode") == "ArtList"
        assert request.url.params.get("format") == "json"
        return httpx.Response(200, json=payload)

    cfg = IngestConfig()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        df = fetch_gdelt_headlines(
            ["trump", "election results"],
            "2025-11-01T00:00:00Z",
            "2025-12-01T00:00:00Z",
            market_id="0xabc",
            config=cfg,
            client=client,
        )

    # Two keywords × two articles = 4 rows; multi-word keyword must be quoted.
    assert len(df) == 4
    assert set(df["keyword"]) == {"trump", "election results"}
    # Multi-word keyword must be quoted in the outgoing GDELT query.
    sent_queries = {params["query"] for params in captured}
    assert sent_queries == {"trump", '"election results"'}
    assert (df["market_id"] == "0xabc").all()
    table = pa.Table.from_pandas(df, schema=NEWS_SCHEMA, preserve_index=False)
    assert table.schema.equals(NEWS_SCHEMA)
    # Verify the datetime range params use GDELT's compact format.
    assert captured[0]["startdatetime"] == "20251101000000"
    assert captured[0]["enddatetime"] == "20251201000000"
