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

from polymarket_political.config import IngestConfig
from polymarket_political.ingest import (
    MARKETS_SCHEMA,
    NEWS_SCHEMA,
    ORDERBOOK_SCHEMA,
    PRICES_SCHEMA,
    TRADES_SCHEMA,
    apply_liquidity_floor,
    derive_taker_side,
    fetch_gdelt_headlines,
    fetch_orderbook_snapshots,
    fetch_political_markets,
    fetch_price_history,
    fetch_trades,
)
from polymarket_political.ingest.orderbook import fetch_current_snapshot
from polymarket_political.ingest.prices import (
    cache_price_history_parquet,
    load_or_fetch_price_history,
)
from polymarket_political.ingest.trades import (
    cache_trades_parquet,
    load_or_fetch_trades,
)


# -------------------------- markets -----------------------------------------


def _fake_market(
    *,
    mid: str,
    vol: float,
    oi: float,
    start: str = "2025-11-01T00:00:00Z",
    end: str = "2026-02-01T00:00:00Z",
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
        "volumeNum": vol,
        "openInterest": oi,
        "liquidityNum": 12345.6,
        "clobTokenIds": json.dumps([f"tok-{mid}-yes", f"tok-{mid}-no"]),
    }


def _fake_event(
    *,
    eid: str,
    markets: list[dict],
    tag: str = "Politics",
    slug: str | None = None,
) -> dict:
    return {
        "id": eid,
        "title": f"Event {eid}",
        "slug": slug or f"event-{eid}",
        "tags": [{"label": tag, "slug": tag.lower()}],
        "markets": markets,
    }


def test_fetch_political_markets_filters_and_schema():
    event1 = _fake_event(
        eid="E1",
        markets=[
            _fake_market(mid="A", vol=600_000, oi=75_000),
            _fake_market(mid="B", vol=100_000, oi=10_000),
        ],
    )
    event2 = _fake_event(
        eid="E2", markets=[_fake_market(mid="C", vol=700_000, oi=0)]
    )
    # With page_size=500 and 1 event per batch, each iterator exits after the
    # first short page, so we only need one response per closed flag.
    responses = iter([[event1], [event2]])
    seen_closed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/events"
        assert request.url.params.get("tag_slug") == "politics"
        assert "end_date_min" in request.url.params
        assert "start_date_max" in request.url.params
        seen_closed.append(request.url.params.get("closed", ""))
        return httpx.Response(200, json=next(responses))

    transport = httpx.MockTransport(handler)
    cfg = IngestConfig(page_size=500)
    with httpx.Client(
        transport=transport, base_url=cfg.gamma_base_url
    ) as client:
        df = fetch_political_markets(
            "2025-10-08", "2026-04-08", config=cfg, client=client
        )

    # 3 markets flattened out of 2 events
    assert len(df) == 3
    assert set(df["market_id"]) == {"A", "B", "C"}
    # Tags inherited from the event's tag list
    assert all("Politics" in tags for tags in df["tags"])
    # Both closed=true and closed=false were queried
    assert "true" in seen_closed and "false" in seen_closed
    table = pa.Table.from_pandas(df, schema=MARKETS_SCHEMA, preserve_index=False)
    assert table.schema.equals(MARKETS_SCHEMA)


def test_fetch_political_markets_client_side_tag_filter():
    """Markets whose inherited tags omit 'politics' must be dropped client-side."""
    political_event = _fake_event(
        eid="POL",
        markets=[_fake_market(mid="keep", vol=1_000_000, oi=0)],
        tag="Politics",
    )
    sports_event = _fake_event(
        eid="SPO",
        markets=[_fake_market(mid="drop", vol=1_000_000, oi=0)],
        tag="Sports",
    )
    responses = iter([[political_event, sports_event], []])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    cfg = IngestConfig(page_size=500)
    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url=cfg.gamma_base_url
    ) as client:
        df = fetch_political_markets(
            "2025-10-08", "2026-04-08", config=cfg, client=client
        )

    assert list(df["market_id"]) == ["keep"]


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

    cfg = IngestConfig(page_size=2, trades_chunk_seconds=7 * 24 * 3600)
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


def test_fetch_trades_keeps_partial_on_offset_cap():
    """When offset cap is hit, fetch_trades keeps the partial pull rather than
    discarding or halving. The truncated attr flag is set.
    """
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        offset = int(request.url.params.get("offset", 0))
        # First page returns 2 trades (== page_size), second page triggers cap.
        if offset == 0:
            return httpx.Response(
                200,
                json=[
                    {
                        "id": f"t-{call_count}-a",
                        "market": "0xabc",
                        "asset": "tok",
                        "timestamp": 1704067200,
                        "price": 0.5,
                        "size": 1,
                        "side": "BUY",
                        "outcome": "YES",
                    },
                    {
                        "id": f"t-{call_count}-b",
                        "market": "0xabc",
                        "asset": "tok",
                        "timestamp": 1704067201,
                        "price": 0.5,
                        "size": 1,
                        "side": "SELL",
                        "outcome": "YES",
                    },
                ],
            )
        return httpx.Response(
            400, json={"error": "max historical activity offset of 3000 exceeded"}
        )

    # Single 120s chunk; page_size=2 so first page is full → second page hits cap.
    cfg = IngestConfig(page_size=2, trades_chunk_seconds=120)
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://data-api.polymarket.com",
    ) as client:
        df = fetch_trades(
            "0xabc",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:02:00Z",
            config=cfg,
            client=client,
        )

    # Should keep the 2 trades from the first page despite cap on second.
    assert len(df) == 2
    assert df["trade_id"].is_unique
    assert df.attrs.get("truncated") is True


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


# -------------------------- prices / prices-history -------------------------


def test_fetch_price_history_schema_and_params(tmp_path):
    history_payload = {
        "history": [
            {"t": 1700000000, "p": 0.41},
            {"t": 1700003600, "p": 0.42},
            {"t": 1700007200, "p": 0.45},
        ]
    }
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prices-history"
        captured.append(dict(request.url.params))
        return httpx.Response(200, json=history_payload)

    cfg = IngestConfig(data_root=str(tmp_path))
    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url=cfg.clob_base_url
    ) as client:
        df = fetch_price_history(
            "tok-yes-1",
            "2023-11-14T00:00:00Z",
            "2023-11-16T00:00:00Z",
            interval="1h",
            config=cfg,
            client=client,
        )

    assert len(df) == 3
    # correct outbound params (market id + required ts bounds + interval)
    assert captured[0]["market"] == "tok-yes-1"
    # With a time range set, Polymarket treats `interval` and `startTs/endTs`
    # as mutually exclusive, so we send `fidelity` (minutes) instead.
    assert "interval" not in captured[0]
    assert captured[0]["fidelity"] == "60"
    assert "startTs" in captured[0] and "endTs" in captured[0]
    # schema contract
    assert (df["market_id"] == "tok-yes-1").all()
    assert df["timestamp"].dt.tz is not None
    table = pa.Table.from_pandas(df, schema=PRICES_SCHEMA, preserve_index=False)
    assert table.schema.equals(PRICES_SCHEMA)


def test_fetch_price_history_rejects_bad_interval():
    with pytest.raises(ValueError, match="interval"):
        fetch_price_history(
            "tok-yes-1",
            "2023-11-14",
            "2023-11-16",
            interval="bogus",
        )


def test_load_or_fetch_price_history_uses_cache(tmp_path):
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200, json={"history": [{"t": 1700000000, "p": 0.5}]}
        )

    cfg = IngestConfig(data_root=str(tmp_path))
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url=cfg.clob_base_url) as client:
        df1 = load_or_fetch_price_history(
            "tok-yes-1",
            "2023-11-14T00:00:00Z",
            "2023-11-16T00:00:00Z",
            config=cfg,
            client=client,
        )
    assert call_count["n"] == 1
    assert len(df1) == 1

    # Second call: cache hit, no new HTTP call even if we pass a client
    with httpx.Client(transport=transport, base_url=cfg.clob_base_url) as client:
        df2 = load_or_fetch_price_history(
            "tok-yes-1",
            "2023-11-14T00:00:00Z",
            "2023-11-16T00:00:00Z",
            config=cfg,
            client=client,
        )
    assert call_count["n"] == 1  # still 1: cache served
    assert len(df2) == 1


def test_cache_price_history_parquet_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "market_id": ["tok-1", "tok-1"],
            "timestamp": pd.to_datetime(
                ["2023-11-14T00:00:00Z", "2023-11-14T01:00:00Z"], utc=True
            ),
            "mid_price": [0.41, 0.42],
        }
    )
    cfg = IngestConfig(data_root=str(tmp_path))
    path = cache_price_history_parquet(
        df, "tok-1", "2023-11-14T00:00:00Z", "2023-11-14T02:00:00Z", config=cfg
    )
    assert path.exists()
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.schema.equals(PRICES_SCHEMA)


def test_load_or_fetch_trades_uses_cache(tmp_path):
    """Second call to load_or_fetch_trades for the same window is a cache hit.

    This is the key property that makes Week 3 backtests affordable: once the
    bisecting pagination has drained a window and written parquet, re-running
    the pipeline is O(file_read) instead of O(dozens of HTTP calls).
    """
    call_count = {"n": 0}
    trades_page = [
        {
            "id": "t1",
            "market": "0xabc",
            "asset": "tok-1",
            "timestamp": 1700000100,
            "price": 0.42,
            "size": 500,
            "side": "BUY",
            "outcome": "YES",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        # First response: 1 trade. Subsequent pages: empty to stop pagination.
        if call_count["n"] == 1:
            return httpx.Response(200, json=trades_page)
        return httpx.Response(200, json=[])

    cfg = IngestConfig(data_root=str(tmp_path), page_size=2)
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://data-api.polymarket.com",
    ) as client:
        df1 = load_or_fetch_trades(
            "0xabc",
            "2023-11-14T00:00:00Z",
            "2023-11-16T00:00:00Z",
            config=cfg,
            client=client,
        )
    first_n = call_count["n"]
    assert first_n >= 1
    assert len(df1) == 1

    # Second call with a fresh client — cache should serve without HTTP
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://data-api.polymarket.com",
    ) as client:
        df2 = load_or_fetch_trades(
            "0xabc",
            "2023-11-14T00:00:00Z",
            "2023-11-16T00:00:00Z",
            config=cfg,
            client=client,
        )
    assert call_count["n"] == first_n  # no new HTTP calls
    assert len(df2) == 1
    # Cached frame preserves tz-aware UTC timestamps
    assert df2["timestamp"].dt.tz is not None


def test_cache_trades_parquet_roundtrip(tmp_path):
    """Round-tripping a trades frame through parquet preserves TRADES_SCHEMA."""
    df = pd.DataFrame(
        {
            "trade_id": ["t1", "t2"],
            "market_id": ["0xabc", "0xabc"],
            "token_id": ["tok-1", "tok-1"],
            "timestamp": pd.to_datetime(
                ["2023-11-14T00:00:00Z", "2023-11-14T01:00:00Z"], utc=True
            ),
            "price": [0.41, 0.42],
            "size": [100.0, 250.0],
            "taker_side": ["buy", "sell"],
            "outcome": ["YES", "YES"],
        }
    )
    cfg = IngestConfig(data_root=str(tmp_path))
    path = cache_trades_parquet(
        df, "0xabc", "2023-11-14T00:00:00Z", "2023-11-14T02:00:00Z", config=cfg
    )
    assert path.exists()
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.schema.equals(TRADES_SCHEMA)
