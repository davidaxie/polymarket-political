"""Realistic synthetic Polymarket market data for integration tests.

The two factory functions return DataFrames whose schema matches the output of
``ingest.trades.fetch_trades`` (``TRADES_SCHEMA``) and
``ingest.prices.fetch_price_history`` (``PRICES_SCHEMA``) respectively. The
goal is fixtures realistic enough to exercise the full
ingest -> features -> detector -> ledger pipeline without ever touching the
network.

If ``volume_spike_at`` is supplied, the synthetic market is engineered so the
insider-flow signal SHOULD fire on that hour:

* 10x more trades in the spike hour vs baseline,
* mid-price drifts by ~+0.10 across the spike hour, and
* ~70% of spike trades are buys (taker-side imbalance).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd


__all__ = ["make_synthetic_market"]


def _utc(ts: datetime) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def make_synthetic_market(
    market_id: str,
    start: datetime,
    end: datetime,
    seed: int = 42,
    volume_spike_at: datetime | None = None,
    n_trades: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate matched trades + prices DataFrames for one synthetic market.

    Parameters
    ----------
    market_id:
        Identifier copied verbatim into ``market_id`` on every row of both
        DataFrames. Strings recommended (matches Polymarket condition IDs).
    start, end:
        Half-open window ``[start, end)`` for the simulation. Either tz-aware
        or naive — naive timestamps are localized to UTC.
    seed:
        Numpy seed for deterministic output.
    volume_spike_at:
        If provided, the hour CONTAINING this timestamp gets a 10x trade-count
        spike, a +0.10 price drift, and ~70% buy-side aggressors. The hour is
        rounded down (``floor("1h")``) before injection.
    n_trades:
        Approximate number of baseline trades evenly distributed across the
        full window.

    Returns
    -------
    (trades, prices):
        ``trades`` columns: ``trade_id, market_id, token_id, timestamp, price,
        size, taker_side, outcome``. Matches ``TRADES_SCHEMA``.
        ``prices`` columns: ``market_id, timestamp, mid_price``. Matches
        ``PRICES_SCHEMA``. Mid prices are sampled hourly across the window.
    """
    rng = np.random.default_rng(seed)
    start_utc = _utc(start)
    end_utc = _utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be strictly after start")

    span_seconds = (end_utc - start_utc).total_seconds()
    if span_seconds <= 0:
        raise ValueError("non-positive window")

    # ----------------- baseline price walk on an hourly grid -----------------
    hourly_index = pd.date_range(
        start_utc.floor("1h"), end_utc.ceil("1h"), freq="1h", tz="UTC"
    )
    n_hours = len(hourly_index)
    # Random walk in [0.30, 0.70] anchored at 0.50.
    drift = rng.normal(0.0, 0.005, size=n_hours).cumsum()
    mids = np.clip(0.50 + drift, 0.30, 0.70)

    spike_hour: pd.Timestamp | None = None
    if volume_spike_at is not None:
        spike_hour = _utc(volume_spike_at).floor("1h")
        if spike_hour in hourly_index:
            idx = hourly_index.get_loc(spike_hour)
            # Apply +0.10 drift across the spike hour and persist it forward.
            mids[idx:] = np.clip(mids[idx:] + 0.10, 0.01, 0.99)

    prices = pd.DataFrame(
        {
            "market_id": market_id,
            "timestamp": hourly_index,
            "mid_price": mids.astype(np.float64),
        }
    )

    # ----------------- baseline trades --------------------------------------
    n_trades = max(int(n_trades), 1)
    base_offsets = rng.uniform(0.0, span_seconds, size=n_trades)
    base_offsets.sort()
    base_ts = pd.to_datetime(start_utc) + pd.to_timedelta(base_offsets, unit="s")
    base_ts = pd.DatetimeIndex(base_ts).tz_convert("UTC")

    # Look up the hourly mid for each trade timestamp.
    floored = base_ts.floor("1h")
    mid_lookup = dict(zip(hourly_index, mids))
    base_prices = np.array(
        [mid_lookup.get(ts, mids[-1]) for ts in floored], dtype=np.float64
    )
    # Tiny per-trade noise around the mid.
    base_prices = np.clip(
        base_prices + rng.normal(0.0, 0.002, size=len(base_prices)),
        0.01,
        0.99,
    )
    base_sides = rng.choice(["buy", "sell"], size=len(base_ts), p=[0.5, 0.5])
    base_sizes = rng.uniform(50.0, 200.0, size=len(base_ts))

    trades_df = pd.DataFrame(
        {
            "trade_id": [f"{market_id}-base-{i}" for i in range(len(base_ts))],
            "market_id": market_id,
            "token_id": f"tok-{market_id}-yes",
            "timestamp": base_ts,
            "price": base_prices.astype(np.float64),
            "size": base_sizes.astype(np.float64),
            "taker_side": base_sides,
            "outcome": "YES",
        }
    )

    # ----------------- spike injection --------------------------------------
    if spike_hour is not None and spike_hour in hourly_index:
        # 10x the average per-hour trade count.
        avg_per_hour = max(int(round(n_trades / max(n_hours, 1))), 1)
        n_spike = avg_per_hour * 10
        spike_offsets = rng.uniform(0.0, 3600.0, size=n_spike)
        spike_offsets.sort()
        spike_ts = pd.DatetimeIndex(
            spike_hour + pd.to_timedelta(spike_offsets, unit="s")
        ).tz_convert("UTC")
        # Linear price drift across the spike hour from the pre-spike mid up
        # by 0.10 — independent samples per trade so the rolling 1h move is
        # large.
        pre_idx = max(hourly_index.get_loc(spike_hour) - 1, 0)
        pre_mid = float(mids[pre_idx])
        ramp = np.linspace(pre_mid, pre_mid + 0.10, num=n_spike)
        ramp = np.clip(ramp, 0.01, 0.99)
        # 70% buy-side aggressors.
        spike_sides = rng.choice(["buy", "sell"], size=n_spike, p=[0.7, 0.3])
        spike_sizes = rng.uniform(150.0, 400.0, size=n_spike)
        spike_df = pd.DataFrame(
            {
                "trade_id": [f"{market_id}-spike-{i}" for i in range(n_spike)],
                "market_id": market_id,
                "token_id": f"tok-{market_id}-yes",
                "timestamp": spike_ts,
                "price": ramp.astype(np.float64),
                "size": spike_sizes.astype(np.float64),
                "taker_side": spike_sides,
                "outcome": "YES",
            }
        )
        trades_df = pd.concat([trades_df, spike_df], ignore_index=True)

    trades_df["timestamp"] = pd.to_datetime(
        trades_df["timestamp"], utc=True
    )
    trades_df = trades_df.sort_values("timestamp").reset_index(drop=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    return trades_df, prices
