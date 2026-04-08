"""Tests for signal feature engineering & detector.

Uses hand-crafted fixture DataFrames — no real data is required.
"""
from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polymarket_political.config import SignalConfig
from polymarket_political.signal.detector import Event, detect_events
from polymarket_political.signal.features import (
    compute_1h_price_move,
    compute_taker_imbalance,
    rolling_hourly_volume,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(market_id, ts, volume_ratio, abs_price_move, imbalance, news=False):
    return {
        "market_id": market_id,
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "volume_ratio": volume_ratio,
        "abs_price_move": abs_price_move,
        "imbalance": imbalance,
        "news_window": news,
    }


@pytest.fixture
def config() -> SignalConfig:
    return SignalConfig(
        volume_multiplier=3.0,
        price_move_threshold=0.05,
        taker_imbalance_threshold=0.6,
        news_filter_enabled=True,
    )


# ---------------------------------------------------------------------------
# Feature function smoke tests
# ---------------------------------------------------------------------------

def test_rolling_hourly_volume_basic():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-04-01", periods=10, freq="30min", tz="UTC"
            ),
            "market_id": ["m1"] * 10,
            "size": [100.0] * 10,
        }
    )
    out = rolling_hourly_volume(trades, window="2h")
    assert {
        "timestamp",
        "market_id",
        "hourly_volume",
        "baseline_volume",
        "volume_ratio",
    } <= set(out.columns)
    # Each hour gets 200 (two 30-min buckets of 100).
    assert (out["hourly_volume"] == 200.0).all()


def test_compute_1h_price_move():
    ts = pd.date_range("2026-04-01", periods=5, freq="1h", tz="UTC")
    ob = pd.DataFrame(
        {
            "timestamp": ts,
            "market_id": ["m1"] * 5,
            "mid_price": [0.50, 0.52, 0.60, 0.58, 0.70],
        }
    )
    out = compute_1h_price_move(ob)
    # Second row: 0.52 vs 0.50 = 0.02
    assert out.iloc[1]["abs_price_move"] == pytest.approx(0.02)
    # Third row: 0.60 vs 0.52 = 0.08
    assert out.iloc[2]["abs_price_move"] == pytest.approx(0.08)


def test_compute_taker_imbalance():
    ts = pd.date_range("2026-04-01", periods=4, freq="15min", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": ts,
            "market_id": ["m1"] * 4,
            "size": [100.0, 100.0, 100.0, 100.0],
            "side": ["buy", "buy", "buy", "sell"],
        }
    )
    out = compute_taker_imbalance(trades, window="1h")
    # 300 buy vs 100 sell -> (300-100)/400 = 0.5
    assert out.iloc[0]["imbalance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Detector scenario tests
# ---------------------------------------------------------------------------

def test_volume_spike_without_price_move_does_not_fire(config):
    df = pd.DataFrame(
        [
            _row("m1", "2026-04-01T10:00", volume_ratio=10.0,
                 abs_price_move=0.01, imbalance=0.8),
        ]
    )
    assert detect_events(df, config) == []


def test_price_move_in_news_window_is_filtered(config):
    df = pd.DataFrame(
        [
            _row("m1", "2026-04-01T10:00", volume_ratio=10.0,
                 abs_price_move=0.10, imbalance=0.8, news=True),
        ]
    )
    assert detect_events(df, config) == []


def test_clean_signal_fires(config):
    df = pd.DataFrame(
        [
            _row("m1", "2026-04-01T10:00", volume_ratio=5.0,
                 abs_price_move=0.08, imbalance=0.7, news=False),
        ]
    )
    events = detect_events(df, config)
    assert len(events) == 1
    evt = events[0]
    assert isinstance(evt, Event)
    assert evt.market_id == "m1"
    assert evt.volume_ratio == 5.0


def test_multiple_markets_sorted(config):
    df = pd.DataFrame(
        [
            _row("m2", "2026-04-01T10:00", 5.0, 0.08, 0.7),
            _row("m1", "2026-04-01T10:00", 5.0, 0.08, 0.7),
            _row("m1", "2026-04-01T11:00", 5.0, 0.08, -0.8),
        ]
    )
    events = detect_events(df, config)
    assert [e.market_id for e in events] == ["m1", "m1", "m2"]


def test_negative_imbalance_also_fires(config):
    df = pd.DataFrame(
        [
            _row("m1", "2026-04-01T10:00", 5.0, 0.08, imbalance=-0.9),
        ]
    )
    assert len(detect_events(df, config)) == 1


def test_threshold_sweep_via_config():
    df = pd.DataFrame(
        [
            _row("m1", "2026-04-01T10:00", 2.0, 0.08, 0.7),
        ]
    )
    # Default 3.0 multiplier blocks it.
    assert detect_events(df, SignalConfig()) == []
    # Loosen via config injection — one-line sweep.
    loose = SignalConfig(volume_multiplier=1.5)
    assert len(detect_events(df, loose)) == 1


def test_missing_column_raises(config):
    df = pd.DataFrame(
        [{"market_id": "m1", "timestamp": pd.Timestamp("2026-04-01", tz="UTC")}]
    )
    with pytest.raises(ValueError, match="missing required columns"):
        detect_events(df, config)


def test_empty_input_returns_empty(config):
    assert detect_events(pd.DataFrame(), config) == []


# ---------------------------------------------------------------------------
# Property-based determinism test
# ---------------------------------------------------------------------------

@st.composite
def _feature_rows(draw):
    n = draw(st.integers(min_value=0, max_value=20))
    rows = []
    for i in range(n):
        rows.append(
            {
                "market_id": draw(st.sampled_from(["m1", "m2", "m3"])),
                "timestamp": pd.Timestamp("2026-04-01", tz="UTC")
                + pd.Timedelta(hours=i),
                "volume_ratio": draw(
                    st.floats(min_value=0.0, max_value=20.0,
                              allow_nan=False, allow_infinity=False)
                ),
                "abs_price_move": draw(
                    st.floats(min_value=0.0, max_value=1.0,
                              allow_nan=False, allow_infinity=False)
                ),
                "imbalance": draw(
                    st.floats(min_value=-1.0, max_value=1.0,
                              allow_nan=False, allow_infinity=False)
                ),
                "news_window": draw(st.booleans()),
            }
        )
    return pd.DataFrame(rows)


@settings(max_examples=50, deadline=None)
@given(df=_feature_rows())
def test_detector_is_deterministic(df):
    cfg = SignalConfig()
    a = detect_events(df.copy(), cfg)
    b = detect_events(df.copy(), cfg)
    assert a == b
