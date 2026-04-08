"""Signal feature engineering & event detection."""
from .detector import Event, detect_events
from .features import (
    compute_1h_price_move,
    compute_taker_imbalance,
    rolling_hourly_volume,
)

__all__ = [
    "Event",
    "detect_events",
    "rolling_hourly_volume",
    "compute_1h_price_move",
    "compute_taker_imbalance",
]
