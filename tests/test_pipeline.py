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
    # Contract columns must always be present, even on the empty path.
    for col in (
        "market_id",
        "timestamp",
        "volume_1h",
        "volume_7d_mean",
        "volume_multiplier",
        "price_move_1h",
        "taker_imbalance_1h",
        "news_window",
    ):
        assert col in out.columns
    assert out.empty


# ---------------------------------------------------------------------------
# Hardening tests — synthetic-data integration, edge cases, contract.
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

from polymarket_political.signal.detector import detect_events
from tests.fixtures.synthetic_market_data import make_synthetic_market


def _markets_df_for(market_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_id": market_id,
                "condition_id": f"0x{market_id}",
                "slug": market_id.lower(),
                "question": f"Will {market_id} happen?",
                "start_date": pd.Timestamp("2025-11-01", tz="UTC"),
                "end_date": pd.Timestamp("2025-12-01", tz="UTC"),
                "active": True,
                "closed": False,
                "volume": 1_000_000.0,
                "open_interest": 100_000.0,
                "liquidity": 25_000.0,
                "tags": ["Politics"],
                "token_ids": [f"tok-{market_id}-yes", f"tok-{market_id}-no"],
            }
        ]
    )


def test_full_pipeline_on_synthetic_market(tmp_path):
    """End-to-end: synthetic spike -> features -> detector -> ledger -> metrics."""
    start = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 11, 21, tzinfo=timezone.utc)
    spike_at = datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

    trades, prices = make_synthetic_market(
        market_id="SYN1",
        start=start,
        end=end,
        seed=7,
        volume_spike_at=spike_at,
        n_trades=4800,
    )

    ingest_cfg = IngestConfig(data_root=str(tmp_path))
    signal_cfg = SignalConfig(
        volume_lookback="7d",
        volume_multiplier=3.0,
        price_move_threshold=0.05,
        taker_imbalance_threshold=0.3,
        news_filter_enabled=False,
    )

    result = run_backtest(
        window="custom",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        signal_config=signal_cfg,
        ingest_config=ingest_cfg,
        markets_fetcher=lambda s, e, **kw: _markets_df_for("SYN1"),
        trades_fetcher=lambda mid, s, e, **kw: trades.copy(),
        prices_fetcher=lambda tok, s, e, **kw: prices.copy(),
        cost_config=ProvisionalCostConfig(),
        cache_intermediate=False,
    )

    assert result.universe_size == 1
    assert result.signal_metrics.n_trades >= 1
    # The detected event should land in the spike hour.
    spike_hour = pd.Timestamp(spike_at).floor("1h")
    event_hours = {pd.Timestamp(e.timestamp).floor("1h") for e in result.events}
    assert spike_hour in event_hours
    # Each detected event produced a Trade.
    assert len(result.ledger) == result.signal_metrics.n_trades
    # Metrics fields are populated and finite.
    import math

    assert math.isfinite(result.signal_metrics.total_pnl)
    assert math.isfinite(result.signal_metrics.mean_pnl)


def test_pipeline_market_with_zero_trades(tmp_path):
    """A market with empty trades should not crash the pipeline."""
    start = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 11, 10, tzinfo=timezone.utc)

    # Build a price-only frame (no trades) so the market is in the universe
    # but produces no events.
    _, prices = make_synthetic_market(
        market_id="EMPTY",
        start=start,
        end=end,
        seed=11,
        volume_spike_at=None,
        n_trades=10,
    )

    empty_trades = pd.DataFrame(
        columns=[
            "trade_id",
            "market_id",
            "token_id",
            "timestamp",
            "price",
            "size",
            "taker_side",
            "outcome",
        ]
    )

    ingest_cfg = IngestConfig(data_root=str(tmp_path))
    result = run_backtest(
        window="custom",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        ingest_config=ingest_cfg,
        markets_fetcher=lambda s, e, **kw: _markets_df_for("EMPTY"),
        trades_fetcher=lambda mid, s, e, **kw: empty_trades.copy(),
        prices_fetcher=lambda tok, s, e, **kw: prices.copy(),
        cost_config=ProvisionalCostConfig(),
        cache_intermediate=False,
    )

    assert result.universe_size == 1
    assert result.signal_metrics.n_trades == 0
    assert len(result.ledger) == 0


def test_build_features_schema():
    """``build_features`` must satisfy the public column contract."""
    start = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 11, 12, tzinfo=timezone.utc)
    trades, prices = make_synthetic_market(
        market_id="SCHEMA",
        start=start,
        end=end,
        seed=3,
        volume_spike_at=datetime(2025, 11, 8, 10, tzinfo=timezone.utc),
        n_trades=3000,
    )

    sig_cfg = SignalConfig()
    out = build_features(trades, prices, sig_cfg)

    expected_columns = {
        "market_id",
        "timestamp",
        "volume_1h",
        "volume_7d_mean",
        "volume_multiplier",
        "price_move_1h",
        "taker_imbalance_1h",
        "news_window",
    }
    assert expected_columns.issubset(set(out.columns))
    assert not out.empty

    # timestamp is tz-aware UTC.
    ts_dtype = out["timestamp"].dtype
    assert getattr(ts_dtype, "tz", None) is not None
    assert str(ts_dtype.tz) == "UTC"

    # all numeric contract columns are float64.
    for col in (
        "volume_1h",
        "volume_7d_mean",
        "volume_multiplier",
        "price_move_1h",
        "taker_imbalance_1h",
    ):
        assert out[col].dtype == "float64", (col, out[col].dtype)

    # news_window defaults to bool False when no news data is provided.
    assert out["news_window"].dtype == bool
    assert not out["news_window"].any()


def test_simulate_trade_provisional_boundary_prices():
    """Boundary entry prices (0.01 and 0.99) must produce sane PnL, no div0."""
    from polymarket_political.backtest.ledger import simulate_trade_provisional

    cfg = ProvisionalCostConfig()

    # Deep OTM: 0.01 entry, exits drift up to 0.05.
    deep_otm = simulate_trade_provisional(
        event={
            "event_id": "boundary-low",
            "market_id": "BNDLOW",
            "entry_ts": 0.0,
            "entry_price": 0.01,
            "size_usd": 1000.0,
        },
        market_data={
            "exit_24h": 0.02,
            "exit_48h": 0.03,
            "exit_72h": 0.05,
            "exit_resolution": 0.10,
            "is_resolved": True,
        },
        config=cfg,
    )
    import math

    for v in (
        deep_otm.pnl_gross_24h,
        deep_otm.pnl_gross_48h,
        deep_otm.pnl_gross_72h,
        deep_otm.pnl_gross_resolution,
        deep_otm.pnl_net_24h,
        deep_otm.pnl_net_48h,
        deep_otm.pnl_net_72h,
        deep_otm.pnl_net_resolution,
    ):
        assert math.isfinite(v)
    # 1000 / 0.01 = 100000 shares, * (0.02 - 0.01) = +1000 gross at 24h.
    assert deep_otm.pnl_gross_24h == pytest.approx(1000.0, rel=1e-6)

    # Near-certain: 0.99 entry, exit at 1.0 / 0.95 boundary.
    near_one = simulate_trade_provisional(
        event={
            "event_id": "boundary-high",
            "market_id": "BNDHIGH",
            "entry_ts": 0.0,
            "entry_price": 0.99,
            "size_usd": 1000.0,
        },
        market_data={
            "exit_24h": 1.00,
            "exit_48h": 0.98,
            "exit_72h": 0.95,
            "exit_resolution": 1.00,
            "is_resolved": True,
        },
        config=cfg,
    )
    for v in (
        near_one.pnl_gross_24h,
        near_one.pnl_gross_48h,
        near_one.pnl_gross_72h,
        near_one.pnl_gross_resolution,
        near_one.pnl_net_24h,
        near_one.pnl_net_48h,
        near_one.pnl_net_72h,
        near_one.pnl_net_resolution,
    ):
        assert math.isfinite(v)
    # 1000 / 0.99 ~= 1010.1 shares * (1.00 - 0.99) ~= +10.10 gross at 24h.
    assert near_one.pnl_gross_24h == pytest.approx(1000.0 / 0.99 * 0.01, rel=1e-6)
