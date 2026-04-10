"""Pure event detector for insider-flow signals.

``detect_events`` takes an already-joined per-(market, hour) feature
DataFrame plus a ``SignalConfig`` and returns a list of ``Event`` records.
No I/O, no globals, deterministic for a fixed input.

Supports two signal modes (controlled by ``config.signal_mode``):

- **prices_only** (default): 2-factor — ``volatility_ratio × price_move``.
  Uses minute-resolution price data to compute realized volatility as a
  proxy for activity. This is the only viable mode for most Polymarket
  markets because the Data API caps trade history at 3000 records.

- **trades**: 3-factor — ``volume_ratio × price_move × taker_imbalance``.
  Requires the full trade tape. Only usable for low-volume markets where
  3000 trades covers the analysis window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ..config import SignalConfig


@dataclass(frozen=True)
class Event:
    """A single detected insider-flow signal."""

    market_id: str
    timestamp: pd.Timestamp
    volume_ratio: float
    abs_price_move: float
    imbalance: float

    def as_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "timestamp": self.timestamp,
            "volume_ratio": self.volume_ratio,
            "abs_price_move": self.abs_price_move,
            "imbalance": self.imbalance,
        }


# Required columns differ by signal mode. We validate dynamically.
_TRADES_COLUMNS = ("market_id", "timestamp", "volume_ratio", "abs_price_move", "imbalance")
_PRICES_ONLY_COLUMNS = ("market_id", "timestamp", "volatility_ratio", "abs_price_move")


def _validate(df: pd.DataFrame, config: SignalConfig) -> None:
    required = (
        _PRICES_ONLY_COLUMNS if config.signal_mode == "prices_only"
        else _TRADES_COLUMNS
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"market_features missing required columns for "
            f"signal_mode={config.signal_mode!r}: {missing}"
        )


def detect_events(
    market_features: pd.DataFrame, config: SignalConfig
) -> list[Event]:
    """Detect insider-flow events from a joined feature DataFrame.

    Parameters
    ----------
    market_features:
        One row per (market_id, hour). Required columns depend on
        ``config.signal_mode``:

        **prices_only** mode:
            - ``market_id``, ``timestamp``, ``volatility_ratio``,
              ``abs_price_move``

        **trades** mode:
            - ``market_id``, ``timestamp``, ``volume_ratio``,
              ``abs_price_move``, ``imbalance``

        Optional (both modes):
            - ``news_window`` (bool). When ``config.news_filter_enabled``
              is True, rows with ``news_window=True`` are excluded.
    config:
        Injected ``SignalConfig`` holding all thresholds.

    Returns
    -------
    list[Event]
        Sorted by ``(market_id, timestamp)`` for determinism.
    """
    if market_features.empty:
        return []

    _validate(market_features, config)

    df = market_features.copy()

    if config.news_filter_enabled and "news_window" in df.columns:
        df = df[~df["news_window"].fillna(False).astype(bool)]

    if config.signal_mode == "prices_only":
        # 2-factor: volatility spike + price move
        df = df.dropna(subset=["volatility_ratio", "abs_price_move"])
        mask = (
            (df["volatility_ratio"] >= config.volatility_multiplier)
            & (df["abs_price_move"] >= config.price_move_threshold)
        )
    else:
        # 3-factor: volume spike + price move + taker imbalance
        df = df.dropna(subset=["volume_ratio", "abs_price_move", "imbalance"])
        mask = (
            (df["volume_ratio"] >= config.volume_multiplier)
            & (df["abs_price_move"] >= config.price_move_threshold)
            & (df["imbalance"].abs() >= config.taker_imbalance_threshold)
        )

    hits = df[mask].sort_values(["market_id", "timestamp"])

    events: list[Event] = [
        Event(
            market_id=str(row.market_id),
            timestamp=pd.Timestamp(row.timestamp),
            volume_ratio=float(
                getattr(row, "volatility_ratio", 0.0)
                if config.signal_mode == "prices_only"
                else getattr(row, "volume_ratio", 0.0)
            ),
            abs_price_move=float(row.abs_price_move),
            imbalance=float(
                getattr(row, "imbalance", 0.0)
                if hasattr(row, "imbalance")
                else 0.0
            ),
        )
        for row in hits.itertuples(index=False)
    ]
    return events


def events_to_frame(events: Iterable[Event]) -> pd.DataFrame:
    """Helper: materialize a list of Events into a DataFrame."""
    rows = [e.as_dict() for e in events]
    if not rows:
        return pd.DataFrame(
            columns=[
                "market_id",
                "timestamp",
                "volume_ratio",
                "abs_price_move",
                "imbalance",
            ]
        )
    return pd.DataFrame(rows)
