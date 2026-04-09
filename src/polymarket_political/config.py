"""Central configuration — signal, cost model, and backtest parameters.

All thresholds live here so sensitivity sweeps are a one-line change.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class SignalConfig(BaseSettings):
    """Thresholds & windows for the insider-flow detector.

    Overridable via env vars prefixed ``POLYSIG_``
    (e.g. ``POLYSIG_VOLUME_MULTIPLIER=5``).
    """

    model_config = SettingsConfigDict(env_prefix="POLYSIG_", extra="ignore")

    # Rolling windows
    volume_lookback: str = "7d"
    taker_imbalance_window: str = "1h"
    price_move_window: str = "1h"

    # Thresholds
    volume_multiplier: float = 3.0
    price_move_threshold: float = 0.05
    taker_imbalance_threshold: float = 0.6

    # Filters
    news_filter_enabled: bool = True


class CostConfig(BaseSettings):
    """Cost model parameters."""

    model_config = SettingsConfigDict(env_prefix="PMI_", extra="ignore")

    fee_rate: float = 0.02
    gas_cost: float = 1.0
    slippage_coef: float = 0.5
    slippage_cap: float = 0.05
    entry_spread_frac: float = 1.0
    exit_spread_frac: float = 0.5


class BacktestConfig(BaseSettings):
    """Backtest runtime parameters."""

    model_config = SettingsConfigDict(env_prefix="PMI_BT_", extra="ignore")

    default_size_usd: float = 100.0
    control_group_size: int = 5
    volume_quartile_bins: int = 4
    random_seed: int = 42


class IngestConfig(BaseSettings):
    """Data-ingestion parameters — API hosts, liquidity floor, rate limits, paths.

    Overridable via env vars prefixed ``PMI_INGEST_``
    (e.g. ``PMI_INGEST_MIN_VOLUME=250000``).
    """

    model_config = SettingsConfigDict(env_prefix="PMI_INGEST_", extra="ignore")

    # API hosts (PRD §8)
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Tag filter (PRD §4.1) — Polymarket "Politics" tag
    political_tag_slug: str = "politics"
    political_tag_label: str = "Politics"

    # Liquidity floor (PRD §4.1 — cumulative lifetime vol ≥ $500k, OI ≥ $50k)
    min_volume: float = 500_000.0
    min_open_interest: float = 50_000.0

    # Rate-limit knobs (Polymarket does not publish a formal limit — be conservative)
    requests_per_second: float = 5.0
    request_timeout_sec: float = 30.0
    max_retries: int = 3
    page_size: int = 500  # /gamma/markets limit per page

    # Trades chunking — Polymarket caps /trades offset paging at 3000, so we
    # slice the requested window into fixed time chunks and paginate each.
    # On cap-hit within a chunk we halve that chunk only (bounded recovery).
    trades_chunk_seconds: int = 1800  # 30 min baseline
    trades_min_chunk_seconds: int = 30  # don't subdivide below this

    # GDELT caps at 250 records per DOC-API call
    gdelt_max_records: int = 250

    # Data paths
    data_root: str = "data"
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"


# Backwards-compat alias used by the signal detector.
Config = SignalConfig
