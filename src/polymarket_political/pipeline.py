"""End-to-end backtest pipeline (Workstream D).

Wires Workstreams A (ingest) → B (signal) → C (backtest infra) into a single
function, :func:`run_backtest`, which Returns a :class:`BacktestResult`.

This is the Week-3 "first real run" path described in the research plan. It
deliberately uses **Option 1 — Polymarket /prices-history mid-price only**
for the price series and the :class:`ProvisionalCostConfig` fixed-spread cost
overrides. Any metric produced here MUST be labeled
``"provisional — mid-price only, fixed-spread cost model"``.

Data flow::

    list_political_markets(start, end)
        └─ apply_liquidity_floor
             └─ for each market:
                   fetch_trades   ── compute_taker_imbalance
                   fetch_prices  ── compute_1h_price_move
                                └─ rolling_hourly_volume
                                └─ join features → detect_events
                                └─ simulate_trade_provisional (for each event)
            └─ generate_control_group → simulate_trade_provisional
            └─ compute_metrics for both ledgers
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .backtest.control import generate_control_group
from .backtest.costs import ProvisionalCostConfig
from .backtest.ledger import Trade, simulate_trade_provisional
from .config import BacktestConfig, IngestConfig, SignalConfig
from .ingest import (
    apply_liquidity_floor,
    fetch_trades,
    list_political_markets,
    load_or_fetch_price_history,
)
from .report.metrics import MetricsSummary, compute_metrics
from .signal.detector import Event, detect_events, events_to_frame
from .signal.features import (
    compute_1h_price_move,
    compute_taker_imbalance,
    rolling_hourly_volume,
)

log = logging.getLogger(__name__)


# --- windows ------------------------------------------------------------------

# Research Plan training / holdout windows. Pinned here so the pipeline and CLI
# never disagree about what "training" means.
TRAINING_START = "2025-10-08"
TRAINING_END = "2026-03-04"
HOLDOUT_START = "2026-03-05"
HOLDOUT_END = "2026-04-08"


def resolve_window(
    window: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> tuple[str, str]:
    """Return (start, end) for ``"training"``, ``"holdout"``, or explicit dates."""
    if window == "training":
        return TRAINING_START, TRAINING_END
    if window == "holdout":
        return HOLDOUT_START, HOLDOUT_END
    if window == "custom":
        if start is None or end is None:
            raise ValueError("window='custom' requires start and end")
        return str(start), str(end)
    raise ValueError(
        f"unknown window {window!r}; expected one of training/holdout/custom"
    )


# --- result container ---------------------------------------------------------


@dataclass
class BacktestResult:
    """Typed return value for :func:`run_backtest`.

    ``signal_metrics`` is always the news-filtered signal group.
    ``control_metrics_filtered`` applies the same news filter to the control
    group so signal-vs-control is an apples-to-apples comparison.
    ``control_metrics_raw`` is the legacy unfiltered control group kept for
    transparency. All metric summaries are 48h net PnL (the PRD §6 default
    horizon). ``config_snapshot`` is a plain dict so the caller can dump it
    straight to parquet/json for provenance.
    """

    window: str
    start: str
    end: str
    universe_size: int
    events: list[Event]
    ledger: list[Trade]
    control_ledger: list[Trade]
    control_ledger_raw: list[Trade]
    signal_metrics: MetricsSummary
    control_metrics_filtered: MetricsSummary
    control_metrics_raw: MetricsSummary
    config_snapshot: dict[str, Any]
    news_filter_applied_to_controls: bool = False
    provisional_label: str = (
        "provisional — mid-price only, fixed-spread cost model"
    )
    warnings: list[str] = field(default_factory=list)

    @property
    def control_metrics(self) -> MetricsSummary:
        """Deprecated alias — returns :attr:`control_metrics_filtered`.

        Kept so existing notebooks and CLI code keep working. New code should
        read ``control_metrics_filtered`` (apples-to-apples vs the signal) or
        ``control_metrics_raw`` (legacy, unfiltered) explicitly.
        """
        return self.control_metrics_filtered

    def to_ledger_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.ledger])

    def to_control_ledger_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.control_ledger])

    def to_control_ledger_raw_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.control_ledger_raw])

    def to_events_frame(self) -> pd.DataFrame:
        return events_to_frame(self.events)


# --- feature join -------------------------------------------------------------


def build_features(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    signal_config: SignalConfig,
) -> pd.DataFrame:
    """Join the three per-(market, hour) feature frames for the detector.

    ``trades`` must have ``[timestamp, market_id, size, taker_side]`` (as
    produced by :func:`ingest.trades.fetch_trades`). ``prices`` must have
    ``[timestamp, market_id, mid_price]`` (from
    :func:`ingest.prices.fetch_price_history`).
    """
    if trades.empty and prices.empty:
        return pd.DataFrame(
            columns=[
                "market_id",
                "timestamp",
                "volume_ratio",
                "abs_price_move",
                "imbalance",
            ]
        )

    vol = rolling_hourly_volume(trades, window=signal_config.volume_lookback)

    # compute_taker_imbalance expects a `side` column; trades have `taker_side`.
    if not trades.empty:
        imb_input = trades.rename(columns={"taker_side": "side"})
    else:
        imb_input = trades
    imb = compute_taker_imbalance(
        imb_input, window=signal_config.taker_imbalance_window
    )

    mv = compute_1h_price_move(prices)

    # Align all three to the hourly grid keyed on (market_id, timestamp).
    # Floor timestamps so the price series (which can be minutely) joins
    # correctly against the hourly volume/imbalance frames.
    if not mv.empty:
        mv = mv.copy()
        mv["timestamp"] = pd.to_datetime(mv["timestamp"], utc=True).dt.floor("1h")
        mv = (
            mv.groupby(["market_id", "timestamp"], as_index=False)
            .agg(
                price_1h_ago=("price_1h_ago", "last"),
                abs_price_move=("abs_price_move", "max"),
            )
        )

    joined = vol.merge(mv, on=["market_id", "timestamp"], how="outer")
    joined = joined.merge(imb, on=["market_id", "timestamp"], how="outer")

    # Fill columns the detector requires but that may be missing on sparse rows.
    for col in ("volume_ratio", "abs_price_move", "imbalance"):
        if col not in joined.columns:
            joined[col] = float("nan")
    joined = joined.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    return joined


# --- event → market_data shim for simulate_trade_provisional ------------------


def _market_data_for_event(
    event: Event,
    prices: pd.DataFrame,
    *,
    resolution_price: float | None = None,
    is_resolved: bool = False,
) -> dict[str, Any]:
    """Pull exit mid-prices at +24h / +48h / +72h from the price series.

    No orderbook. No depth. This is the provisional path.
    """
    if prices.empty:
        return {
            "exit_24h": None,
            "exit_48h": None,
            "exit_72h": None,
            "exit_resolution": resolution_price,
            "is_resolved": is_resolved,
        }

    s = (
        prices[prices["market_id"] == event.market_id]
        .sort_values("timestamp")
        .set_index("timestamp")["mid_price"]
    )
    if s.empty:
        return {
            "exit_24h": None,
            "exit_48h": None,
            "exit_72h": None,
            "exit_resolution": resolution_price,
            "is_resolved": is_resolved,
        }

    def at(offset: str) -> float | None:
        target = event.timestamp + pd.Timedelta(offset)
        sub = s[s.index <= target]
        if sub.empty:
            return None
        return float(sub.iloc[-1])

    return {
        "exit_24h": at("24h"),
        "exit_48h": at("48h"),
        "exit_72h": at("72h"),
        "exit_resolution": resolution_price,
        "is_resolved": is_resolved,
    }


def _entry_price_for_event(event: Event, prices: pd.DataFrame) -> float | None:
    if prices.empty:
        return None
    s = (
        prices[prices["market_id"] == event.market_id]
        .sort_values("timestamp")
        .set_index("timestamp")["mid_price"]
    )
    if s.empty:
        return None
    sub = s[s.index <= event.timestamp]
    if sub.empty:
        return float(s.iloc[0])
    return float(sub.iloc[-1])


# --- pipeline main entry ------------------------------------------------------


def run_backtest(
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    window: str = "training",
    *,
    signal_config: SignalConfig | None = None,
    ingest_config: IngestConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    cost_config: ProvisionalCostConfig | None = None,
    # Dependency-injection seams for testing. Defaults call the real ingest
    # modules which will hit the network unless the caller supplies an
    # httpx.Client with a MockTransport attached.
    markets_fetcher=None,
    trades_fetcher=None,
    prices_fetcher=None,
    client: httpx.Client | None = None,
    max_markets: int | None = None,
    cache_intermediate: bool = True,
) -> BacktestResult:
    """Run the full backtest pipeline over the requested window.

    Parameters
    ----------
    window:
        ``"training"`` (default), ``"holdout"``, or ``"custom"``. Custom
        requires ``start``/``end`` as ISO strings.
    markets_fetcher, trades_fetcher, prices_fetcher:
        Dependency-injection seams. Each is called with the same
        ``(..., config=..., client=...)`` signature as the underlying ingest
        function. Unit tests pass canned lambdas here; production code leaves
        them as ``None`` to use the real ingest modules.
    cache_intermediate:
        When True, writes the final ledger and metrics to
        ``data/processed/`` as parquets so re-runs can skip a full pass.
    """
    sig_cfg = signal_config or SignalConfig()
    ing_cfg = ingest_config or IngestConfig()
    bt_cfg = backtest_config or BacktestConfig()
    prov_cost_cfg = cost_config or ProvisionalCostConfig()

    resolved_start, resolved_end = resolve_window(window, start, end)
    log.info("run_backtest window=%s start=%s end=%s", window, resolved_start, resolved_end)

    fetch_markets = markets_fetcher or (
        lambda s, e, **kw: list_political_markets(s, e, config=ing_cfg, **kw)
    )
    fetch_trades_fn = trades_fetcher or (
        lambda mid, s, e, **kw: fetch_trades(mid, s, e, config=ing_cfg, **kw)
    )
    fetch_prices_fn = prices_fetcher or (
        lambda tok, s, e, **kw: load_or_fetch_price_history(
            tok, s, e, config=ing_cfg, **kw
        )
    )

    # 1. Universe
    markets_df = fetch_markets(resolved_start, resolved_end, client=client)
    markets_df = apply_liquidity_floor(markets_df, config=ing_cfg)
    if max_markets is not None:
        markets_df = markets_df.head(max_markets).reset_index(drop=True)
    universe_size = int(len(markets_df))
    log.info("universe after liquidity floor: %d markets", universe_size)

    warnings: list[str] = []
    if universe_size == 0:
        warnings.append("Empty market universe after liquidity floor.")

    # 2. Per-market trade/price pulls and feature build.
    all_features: list[pd.DataFrame] = []
    prices_by_market: dict[str, pd.DataFrame] = {}
    trades_by_market: dict[str, pd.DataFrame] = {}
    per_market_volume: dict[str, float] = {}
    per_market_timestamps: dict[str, list[pd.Timestamp]] = {}

    for row in markets_df.itertuples(index=False):
        market_id = getattr(row, "market_id", "")
        condition_id = getattr(row, "condition_id", "") or market_id
        token_ids = list(getattr(row, "token_ids", []) or [])
        # YES-token is conventionally the first token id.
        yes_token = token_ids[0] if token_ids else market_id

        try:
            trades = fetch_trades_fn(condition_id, resolved_start, resolved_end, client=client)
        except Exception as exc:  # pragma: no cover — network-path guard
            warnings.append(f"trades fetch failed for {market_id}: {exc}")
            continue
        try:
            prices = fetch_prices_fn(yes_token, resolved_start, resolved_end, client=client)
        except Exception as exc:  # pragma: no cover — network-path guard
            warnings.append(f"prices fetch failed for {market_id}: {exc}")
            continue

        # Re-tag the price frame so join keys match the trade/market id, not
        # the token id it was fetched with.
        if not prices.empty:
            prices = prices.copy()
            prices["market_id"] = market_id
        if not trades.empty:
            trades = trades.copy()
            trades["market_id"] = market_id

        trades_by_market[market_id] = trades
        prices_by_market[market_id] = prices
        per_market_volume[market_id] = float(getattr(row, "volume", 0.0) or 0.0)
        if not prices.empty:
            per_market_timestamps[market_id] = list(prices["timestamp"])

        features = build_features(trades, prices, sig_cfg)
        if not features.empty:
            all_features.append(features)

    # 3. Event detection.
    if all_features:
        features_df = pd.concat(all_features, ignore_index=True)
    else:
        features_df = pd.DataFrame(
            columns=[
                "market_id",
                "timestamp",
                "volume_ratio",
                "abs_price_move",
                "imbalance",
            ]
        )
    events = detect_events(features_df, sig_cfg)
    log.info("detected %d events", len(events))

    # 4. Simulate trades (provisional cost path).
    ledger: list[Trade] = []
    for idx, event in enumerate(events):
        prices = prices_by_market.get(event.market_id, pd.DataFrame())
        entry_price = _entry_price_for_event(event, prices)
        if entry_price is None or entry_price <= 0:
            warnings.append(
                f"skipped event {event.market_id}@{event.timestamp}: no entry price"
            )
            continue
        market_data = _market_data_for_event(event, prices)
        trade = simulate_trade_provisional(
            event={
                "event_id": f"sig::{idx}::{event.market_id}",
                "market_id": event.market_id,
                "entry_ts": float(event.timestamp.timestamp()),
                "entry_price": entry_price,
                "size_usd": bt_cfg.default_size_usd,
            },
            market_data=market_data,
            config=prov_cost_cfg,
            bt_config=bt_cfg,
        )
        ledger.append(trade)

    # 5. Control group — volume-quartile matched random markets + timestamps.
    event_dicts: list[dict] = []
    for idx, event in enumerate(events):
        event_dicts.append(
            {
                "event_id": f"sig::{idx}::{event.market_id}",
                "market_id": event.market_id,
                "entry_ts": float(event.timestamp.timestamp()),
                "volume": per_market_volume.get(event.market_id, 0.0),
            }
        )
    universe_for_controls: list[dict] = []
    for mid, vol in per_market_volume.items():
        universe_for_controls.append(
            {
                "market_id": mid,
                "volume": vol,
                "timestamps": [
                    float(ts.timestamp())
                    for ts in per_market_timestamps.get(mid, [])
                ]
                or [pd.Timestamp(resolved_start, tz="UTC").timestamp()],
            }
        )
    control_events = generate_control_group(
        event_dicts,
        universe_for_controls,
        n=bt_cfg.control_group_size,
        config=bt_cfg,
    )

    # Build a set of (market_id, hour-floored timestamp) pairs marked as
    # news windows. When ``news_filter_enabled=True`` this mirrors exactly
    # what ``detect_events`` drops from the signal group, so we can apply
    # the same filter to the control group.
    news_window_pairs: set[tuple[str, pd.Timestamp]] = set()
    if (
        sig_cfg.news_filter_enabled
        and not features_df.empty
        and "news_window" in features_df.columns
    ):
        mask = features_df["news_window"].fillna(False).astype(bool)
        for row in features_df.loc[mask, ["market_id", "timestamp"]].itertuples(
            index=False
        ):
            ts = pd.Timestamp(row.timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            news_window_pairs.add((str(row.market_id), ts.floor("1h")))

    def _is_news_window(market_id: str, ts: pd.Timestamp) -> bool:
        if not news_window_pairs:
            return False
        floor = ts.floor("1h")
        return (str(market_id), floor) in news_window_pairs

    def _simulate_control(c: dict) -> Trade | None:
        prices = prices_by_market.get(c["market_id"], pd.DataFrame())
        ts = pd.Timestamp(c["entry_ts"], unit="s", tz="UTC")
        pseudo_event = Event(
            market_id=c["market_id"],
            timestamp=ts,
            volume_ratio=0.0,
            abs_price_move=0.0,
            imbalance=0.0,
        )
        entry_price = _entry_price_for_event(pseudo_event, prices)
        if entry_price is None or entry_price <= 0:
            return None
        market_data = _market_data_for_event(pseudo_event, prices)
        return simulate_trade_provisional(
            event={
                "event_id": c["event_id"],
                "market_id": c["market_id"],
                "entry_ts": float(c["entry_ts"]),
                "entry_price": entry_price,
                "size_usd": bt_cfg.default_size_usd,
            },
            market_data=market_data,
            config=prov_cost_cfg,
            bt_config=bt_cfg,
        )

    control_ledger_raw: list[Trade] = []
    control_ledger_filtered: list[Trade] = []
    for c in control_events:
        trade = _simulate_control(c)
        if trade is None:
            continue
        control_ledger_raw.append(trade)
        ts = pd.Timestamp(c["entry_ts"], unit="s", tz="UTC")
        if not _is_news_window(c["market_id"], ts):
            control_ledger_filtered.append(trade)

    # 6. Metrics — default horizon per PRD §6 is 48h net PnL.
    def _pnls(trades: Iterable[Trade]) -> list[float]:
        return [t.pnl_net_48h for t in trades]

    signal_metrics = compute_metrics(_pnls(ledger))
    control_metrics_filtered = compute_metrics(_pnls(control_ledger_filtered))
    control_metrics_raw = compute_metrics(_pnls(control_ledger_raw))

    config_snapshot = {
        "window": window,
        "start": resolved_start,
        "end": resolved_end,
        "signal_config": sig_cfg.model_dump(),
        "cost_config_provisional": asdict(prov_cost_cfg),
        "backtest_config": bt_cfg.model_dump(),
        "universe_size": universe_size,
    }

    result = BacktestResult(
        window=window,
        start=resolved_start,
        end=resolved_end,
        universe_size=universe_size,
        events=events,
        ledger=ledger,
        control_ledger=control_ledger_filtered,
        control_ledger_raw=control_ledger_raw,
        signal_metrics=signal_metrics,
        control_metrics_filtered=control_metrics_filtered,
        control_metrics_raw=control_metrics_raw,
        news_filter_applied_to_controls=bool(sig_cfg.news_filter_enabled),
        config_snapshot=config_snapshot,
        warnings=warnings,
    )

    if cache_intermediate:
        try:
            _cache_result(result, ing_cfg)
        except Exception as exc:  # pragma: no cover — best-effort cache
            log.warning("failed to cache result: %s", exc)

    return result


def _cache_result(result: BacktestResult, ing_cfg: IngestConfig) -> None:
    out_dir = Path(ing_cfg.data_root) / ing_cfg.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger_df = result.to_ledger_frame()
    if not ledger_df.empty:
        ledger_df.to_parquet(out_dir / f"ledger_{result.window}.parquet")
    ctrl_df = result.to_control_ledger_frame()
    if not ctrl_df.empty:
        ctrl_df.to_parquet(out_dir / f"control_ledger_{result.window}.parquet")

    metrics_rows = [
        {"group": "signal", **asdict(result.signal_metrics)},
        {"group": "control_filtered", **asdict(result.control_metrics_filtered)},
        {"group": "control_raw", **asdict(result.control_metrics_raw)},
    ]
    pd.DataFrame(metrics_rows).to_parquet(
        out_dir / f"metrics_{result.window}.parquet"
    )
