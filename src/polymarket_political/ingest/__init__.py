"""Workstream A — data ingestion for the Polymarket insider-flow backtest.

Modules:
- ``markets``: enumerate political markets via Gamma API + liquidity floor
- ``orderbook``: historical orderbook snapshots (reconstruction path documented)
- ``trades``: full trade tape, taker side derived from aggressor flag
- ``news``: GDELT DOC 2.0 headline pulls for the news filter
"""
from __future__ import annotations

from .markets import (
    MARKETS_SCHEMA,
    apply_liquidity_floor,
    fetch_political_markets,
    list_political_markets,
)
from .news import NEWS_SCHEMA, fetch_gdelt_headlines
from .orderbook import ORDERBOOK_SCHEMA, fetch_orderbook_snapshots
from .prices import (
    PRICES_SCHEMA,
    cache_price_history_parquet,
    fetch_price_history,
    load_or_fetch_price_history,
)
from .trades import TRADES_SCHEMA, derive_taker_side, fetch_trades

__all__ = [
    "MARKETS_SCHEMA",
    "NEWS_SCHEMA",
    "ORDERBOOK_SCHEMA",
    "PRICES_SCHEMA",
    "TRADES_SCHEMA",
    "apply_liquidity_floor",
    "cache_price_history_parquet",
    "derive_taker_side",
    "fetch_gdelt_headlines",
    "fetch_orderbook_snapshots",
    "fetch_political_markets",
    "fetch_price_history",
    "fetch_trades",
    "list_political_markets",
    "load_or_fetch_price_history",
]
