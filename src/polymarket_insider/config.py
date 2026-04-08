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


# Backwards-compat alias used by the signal detector.
Config = SignalConfig
