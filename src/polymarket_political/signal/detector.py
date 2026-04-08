"""Pure event detector for insider-flow signals.

``detect_events`` takes an already-joined per-(market, hour) feature
DataFrame plus a ``SignalConfig`` and returns a list of ``Event`` records.
No I/O, no globals, deterministic for a fixed input.
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


REQUIRED_COLUMNS: tuple[str, ...] = (
    "market_id",
    "timestamp",
    "volume_ratio",
    "abs_price_move",
    "imbalance",
)


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"market_features missing required columns: {missing}"
        )


def detect_events(
    market_features: pd.DataFrame, config: SignalConfig
) -> list[Event]:
    """Detect insider-flow events from a joined feature DataFrame.

    Parameters
    ----------
    market_features:
        One row per (market_id, hour) with at least these columns:
            - ``market_id`` (str)
            - ``timestamp`` (datetime64[ns, UTC])
            - ``volume_ratio`` (float)
            - ``abs_price_move`` (float, probability units)
            - ``imbalance`` (float in [-1, 1])
        Optional:
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

    _validate(market_features)

    df = market_features.copy()

    if config.news_filter_enabled and "news_window" in df.columns:
        df = df[~df["news_window"].fillna(False).astype(bool)]

    # Drop rows where any feature is missing (cannot evaluate).
    df = df.dropna(
        subset=["volume_ratio", "abs_price_move", "imbalance"]
    )

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
            volume_ratio=float(row.volume_ratio),
            abs_price_move=float(row.abs_price_move),
            imbalance=float(row.imbalance),
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
