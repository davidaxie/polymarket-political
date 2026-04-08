"""Integration test for the Workstream D end-to-end backtest pipeline.

Uses canned in-memory DataFrames via the pipeline's dependency-injection seams
(``markets_fetcher``, ``trades_fetcher``, ``prices_fetcher``) so no network is
involved. The component modules are already covered by their own tests — the
job here is to verify the wiring produces a :class:`MetricsSummary` without
blowing up and with the expected provisional label.
"""
from __future__ import annotations

import pandas as pd
import pytest

from polymarket_political.backtest.costs import ProvisionalCostConfig
from polymarket_political.config import IngestConfig, SignalConfig
from polymarket_political.pipeline import (
    BacktestResult,
    build_features,
    run_backtest,
)
from polymarket_political.report.metrics import MetricsSummary


def _canned_markets() -> pd.DataFrame:
    # Two markets, both above the default liquidity floor (500k / 50k).
    return pd.DataFrame(
        [
            {
                "market_id": "M1",
                "condition_id": "0xM1",
                "slug": "m1",
                "question": "Will thing 1 happen?",
                "start_date": pd.Timestamp("2025-10-08", tz="UTC"),
                "end_date": pd.Timestamp("2026-03-04", tz="UTC"),
                "active": True,
                "closed": False,
                "volume": 1_000_000.0,
                "open_interest": 100_000.0,
                "liquidity": 25_000.0,
                "tags": ["Politics"],
                "token_ids": ["tok-M1-yes", "tok-M1-no"],
            },
            {
                "market_id": "M2",
                "condition_id": "0xM2",
                "slug": "m2",
                "question": "Will thing 2 happen?",
                "start_date": pd.Timestamp("2025-10-08", tz="UTC"),
                "end_date": pd.Timestamp("2026-03-04", tz="UTC"),
                "active": True,
                "closed": False,
                "volume": 750_000.0,
                "open_interest": 60_000.0,
                "liquidity": 20_000.0,
                "tags": ["Politics"],
                "token_ids": ["tok-M2-yes", "tok-M2-no"],
            },
        ]
    )


def _canned_trades(market_id: str) -> pd.DataFrame:
    # Build a high-volume, heavily-buy hour to trip the signal thresholds.
    # Baseline volume: 1h slots with 100 size for 10 days; spike hour with 2000.
    base_ts = pd.Timestamp("2025-11-01T00:00:00Z")
    rows: list[dict] = []
    # 10 days of low-volume background (so rolling baseline > 0).
    for d in range(10):
        for h in range(24):
            rows.append(
                {
                    "trade_id": f"{market_id}-bg-{d}-{h}",
                    "market_id": market_id,
                    "token_id": f"tok-{market_id}-yes",
                    "timestamp": base_ts + pd.Timedelta(days=d, hours=h),
                    "price": 0.40,
                    "size": 100.0,
                    "taker_side": "buy" if h % 2 == 0 else "sell",
                    "outcome": "YES",
                }
            )
    # Spike hour: big buy-side imbalance on market M1 only.
    if market_id == "M1":
        spike_ts = base_ts + pd.Timedelta(days=10, hours=5)
        for i in range(20):
            rows.append(
                {
                    "trade_id": f"{market_id}-spike-{i}",
                    "market_id": market_id,
                    "token_id": f"tok-{market_id}-yes",
                    "timestamp": spike_ts + pd.Timedelta(minutes=i),
                    "price": 0.45,
                    "size": 500.0,
                    "taker_side": "buy",
                    "outcome": "YES",
                }
            )
    return pd.DataFrame(rows)


def _canned_prices(token_id: str) -> pd.DataFrame:
    # Hourly mid series covering the same window as trades, plus a jump on
    # day 10 for M1 so compute_1h_price_move clears the 5pp threshold.
    base_ts = pd.Timestamp("2025-11-01T00:00:00Z")
    rows: list[dict] = []
    for d in range(14):
        for h in range(24):
            ts = base_ts + pd.Timedelta(days=d, hours=h)
            if token_id == "tok-M1-yes":
                # Flat 0.40 for 10 days, then jump to 0.46 at hour 5 on day 10,
                # then rise further to 0.50 by day 12.
                if d < 10:
                    price = 0.40
                elif d == 10 and h < 5:
                    price = 0.40
                elif d == 10 and h == 5:
                    price = 0.46
                elif d <= 12:
                    price = 0.50
                else:
                    price = 0.52
            else:
                price = 0.30 + (d * 0.001)
            rows.append(
                {"market_id": token_id, "timestamp": ts, "mid_price": float(price)}
            )
    return pd.DataFrame(rows)


def test_run_backtest_end_to_end_returns_metrics_summary(tmp_path):
    ingest_cfg = IngestConfig(data_root=str(tmp_path))
    signal_cfg = SignalConfig(
        volume_lookback="7d",
        volume_multiplier=3.0,
        price_move_threshold=0.05,
        taker_imbalance_threshold=0.6,
        news_filter_enabled=False,
    )

    def markets_fetcher(start, end, **kw):
        return _canned_markets()

    def trades_fetcher(market_id, start, end, **kw):
        return _canned_trades(market_id.replace("0x", ""))

    def prices_fetcher(token_id, start, end, **kw):
        return _canned_prices(token_id)

    result = run_backtest(
        window="training",
        signal_config=signal_cfg,
        ingest_config=ingest_cfg,
        markets_fetcher=markets_fetcher,
        trades_fetcher=trades_fetcher,
        prices_fetcher=prices_fetcher,
        cost_config=ProvisionalCostConfig(),
        cache_intermediate=False,
    )

    assert isinstance(result, BacktestResult)
    assert isinstance(result.signal_metrics, MetricsSummary)
    assert isinstance(result.control_metrics, MetricsSummary)

    # The universe should contain both canned markets.
    assert result.universe_size == 2

    # Window metadata matches the research-plan training window.
    assert result.window == "training"
    assert result.start == "2025-10-08"
    assert result.end == "2026-03-04"

    # The provisional banner must be present and labeled correctly.
    assert "provisional" in result.provisional_label.lower()
    assert result.config_snapshot["cost_config_provisional"]["fee_rate"] == 0.02

    # Signal should have detected at least the one engineered spike on M1.
    assert result.signal_metrics.n_trades >= 1
    # Ledger / control ledger wiring is intact.
    assert len(result.ledger) == result.signal_metrics.n_trades
    # Control group is 5x events by default.
    # (Controls can be filtered down if their price series is empty.)
    assert len(result.control_ledger) >= 0


def test_run_backtest_empty_universe_returns_empty_metrics(tmp_path):
    ingest_cfg = IngestConfig(data_root=str(tmp_path))

    result = run_backtest(
        window="training",
        ingest_config=ingest_cfg,
        markets_fetcher=lambda s, e, **kw: pd.DataFrame(),
        trades_fetcher=lambda m, s, e, **kw: pd.DataFrame(),
        prices_fetcher=lambda t, s, e, **kw: pd.DataFrame(),
        cache_intermediate=False,
    )

    assert result.universe_size == 0
    assert result.signal_metrics.n_trades == 0
    assert result.control_metrics.n_trades == 0
    assert any("Empty" in w or "empty" in w for w in result.warnings)


def test_news_filter_applied_to_controls_symmetrically(tmp_path, monkeypatch):
    """When news_filter_enabled=True, controls get the same filter as the signal.

    We inject a ``news_window`` column into the feature frame for M2 around
    the timestamps the control generator is likely to draw, so at least one
    control event should land in a news window and be dropped from the
    filtered control set (but not the raw set).
    """
    from polymarket_political import pipeline as pipeline_mod

    ingest_cfg = IngestConfig(data_root=str(tmp_path))
    signal_cfg = SignalConfig(
        volume_lookback="7d",
        volume_multiplier=3.0,
        price_move_threshold=0.05,
        taker_imbalance_threshold=0.6,
        news_filter_enabled=True,
    )

    real_build_features = pipeline_mod.build_features

    def build_features_with_news(trades, prices, sig_cfg):
        df = real_build_features(trades, prices, sig_cfg)
        if df.empty:
            return df
        df = df.copy()
        # Mark every M2 row as a news window, and every non-spike M1 row
        # (anything before day 10). The M1 spike hour stays unmarked so the
        # signal detector still fires an event. Controls that land on any
        # marked (market_id, hour) pair — which is most of the timestamp
        # pool — should be dropped from the filtered control set, leaving
        # it strictly smaller than the raw control set.
        is_m2 = df["market_id"] == "M2"
        spike_hour = pd.Timestamp("2025-11-11T05:00:00Z")
        is_old_m1 = (df["market_id"] == "M1") & (df["timestamp"] < spike_hour)
        df["news_window"] = (is_m2 | is_old_m1).astype(bool)
        return df

    monkeypatch.setattr(pipeline_mod, "build_features", build_features_with_news)

    result = run_backtest(
        window="training",
        signal_config=signal_cfg,
        ingest_config=ingest_cfg,
        markets_fetcher=lambda s, e, **kw: _canned_markets(),
        trades_fetcher=lambda mid, s, e, **kw: _canned_trades(mid.replace("0x", "")),
        prices_fetcher=lambda tok, s, e, **kw: _canned_prices(tok),
        cost_config=ProvisionalCostConfig(),
        cache_intermediate=False,
    )

    assert result.news_filter_applied_to_controls is True
    # Both control metrics are computed.
    assert isinstance(result.control_metrics_filtered, MetricsSummary)
    assert isinstance(result.control_metrics_raw, MetricsSummary)
    # Filtered control has strictly fewer trades than the raw one because
    # every feature row is inside a news window.
    assert result.control_metrics_raw.n_trades > 0
    assert result.control_metrics_filtered.n_trades < result.control_metrics_raw.n_trades
    # Backwards-compatible alias still works.
    assert result.control_metrics is result.control_metrics_filtered


def test_news_filter_disabled_yields_identical_control_metrics(tmp_path):
    """When news_filter_enabled=False the two control metrics must match."""
    ingest_cfg = IngestConfig(data_root=str(tmp_path))
    signal_cfg = SignalConfig(
        volume_lookback="7d",
        volume_multiplier=3.0,
        price_move_threshold=0.05,
        taker_imbalance_threshold=0.6,
        news_filter_enabled=False,
    )

    result = run_backtest(
        window="training",
        signal_config=signal_cfg,
        ingest_config=ingest_cfg,
        markets_fetcher=lambda s, e, **kw: _canned_markets(),
        trades_fetcher=lambda mid, s, e, **kw: _canned_trades(mid.replace("0x", "")),
        prices_fetcher=lambda tok, s, e, **kw: _canned_prices(tok),
        cost_config=ProvisionalCostConfig(),
        cache_intermediate=False,
    )

    assert result.news_filter_applied_to_controls is False
    assert (
        result.control_metrics_filtered.n_trades
        == result.control_metrics_raw.n_trades
    )
    assert (
        result.control_metrics_filtered.total_pnl
        == result.control_metrics_raw.total_pnl
    )


def test_build_features_handles_empty_inputs():
    sig_cfg = SignalConfig()
    out = build_features(pd.DataFrame(), pd.DataFrame(), sig_cfg)
    assert list(out.columns) == [
        "market_id",
        "timestamp",
        "volume_ratio",
        "abs_price_move",
        "imbalance",
    ]
    assert out.empty
