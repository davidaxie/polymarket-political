"""Parameter sensitivity sweep for the insider-flow signal.

Runs :func:`pipeline.run_backtest` across a grid of threshold values and
collects metrics for each combination. The result is a DataFrame that
shows how signal quality degrades/improves as thresholds change —
critical for choosing production-ready thresholds before the holdout test.

Usage::

    from polymarket_political.sweep import run_sweep, DEFAULT_GRID
    results = run_sweep(window="training", grid=DEFAULT_GRID)
    results.to_parquet("data/processed/sweep_training.parquet")
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from .backtest.costs import ProvisionalCostConfig
from .config import BacktestConfig, IngestConfig, SignalConfig
from .pipeline import run_backtest

log = logging.getLogger(__name__)


# Default sweep grid — focused on the prices-only 2-factor signal.
DEFAULT_GRID: dict[str, list] = {
    "volatility_multiplier": [2.0, 3.0, 5.0, 8.0],
    "price_move_threshold": [0.02, 0.03, 0.05, 0.08, 0.10],
}

# Extended grid including entry timing experiments.
TIMING_GRID: dict[str, list] = {
    "volatility_multiplier": [3.0, 5.0],
    "price_move_threshold": [0.03, 0.05],
    "entry_delay_hours": [0, 1, 2, 4],
    "entry_direction": ["momentum", "contrarian"],
}


@dataclass
class SweepRow:
    """One row of sweep results."""

    volatility_multiplier: float
    price_move_threshold: float
    entry_delay_hours: int
    entry_direction: str
    n_events: int
    n_trades: int
    hit_rate: float
    mean_pnl: float
    median_pnl: float
    total_pnl: float
    sharpe: float
    max_drawdown: float
    ctrl_n_trades: int
    ctrl_hit_rate: float
    ctrl_mean_pnl: float
    ctrl_sharpe: float
    edge_vs_ctrl: float  # signal mean_pnl - control mean_pnl


def run_sweep(
    window: str = "training",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    *,
    grid: dict[str, list[float]] | None = None,
    ingest_config: IngestConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    cost_config: ProvisionalCostConfig | None = None,
    max_markets: int | None = None,
    # DI seams — pass through to run_backtest
    markets_fetcher=None,
    trades_fetcher=None,
    prices_fetcher=None,
) -> pd.DataFrame:
    """Run backtest across all parameter combinations in ``grid``.

    Returns a DataFrame with one row per combination, suitable for sorting
    by Sharpe, hit rate, or edge_vs_ctrl to find the best thresholds.
    """
    grid = grid or DEFAULT_GRID
    ing_cfg = ingest_config or IngestConfig()
    bt_cfg = backtest_config or BacktestConfig()
    prov_cfg = cost_config or ProvisionalCostConfig()

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    log.info("sweep: %d parameter combinations", len(combos))

    rows: list[dict[str, Any]] = []

    for combo in combos:
        params = dict(zip(keys, combo))
        log.info("sweep: running %s", params)

        sig_cfg = SignalConfig(
            signal_mode="prices_only",
            volatility_multiplier=params.get("volatility_multiplier", 3.0),
            price_move_threshold=params.get("price_move_threshold", 0.05),
            entry_delay_hours=int(params.get("entry_delay_hours", 0)),
            entry_direction=str(params.get("entry_direction", "momentum")),
            news_filter_enabled=False,
        )

        try:
            result = run_backtest(
                window=window,
                start=start,
                end=end,
                signal_config=sig_cfg,
                ingest_config=ing_cfg,
                backtest_config=bt_cfg,
                cost_config=prov_cfg,
                max_markets=max_markets,
                markets_fetcher=markets_fetcher,
                trades_fetcher=trades_fetcher,
                prices_fetcher=prices_fetcher,
                cache_intermediate=False,
            )
        except Exception as exc:
            log.warning("sweep: failed on %s: %s", params, exc)
            continue

        sm = result.signal_metrics
        cm = result.control_metrics_filtered
        edge = sm.mean_pnl - cm.mean_pnl if cm.n_trades > 0 else sm.mean_pnl

        rows.append(
            asdict(
                SweepRow(
                    volatility_multiplier=params.get("volatility_multiplier", 3.0),
                    price_move_threshold=params.get("price_move_threshold", 0.05),
                    entry_delay_hours=int(params.get("entry_delay_hours", 0)),
                    entry_direction=str(params.get("entry_direction", "momentum")),
                    n_events=len(result.events),
                    n_trades=sm.n_trades,
                    hit_rate=sm.hit_rate,
                    mean_pnl=sm.mean_pnl,
                    median_pnl=sm.median_pnl,
                    total_pnl=sm.total_pnl,
                    sharpe=sm.sharpe,
                    max_drawdown=sm.max_drawdown,
                    ctrl_n_trades=cm.n_trades,
                    ctrl_hit_rate=cm.hit_rate,
                    ctrl_mean_pnl=cm.mean_pnl,
                    ctrl_sharpe=cm.sharpe,
                    edge_vs_ctrl=edge,
                )
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df
