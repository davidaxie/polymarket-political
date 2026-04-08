"""Metrics: hit rate, mean/median PnL, Sharpe, max drawdown, tail concentration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math


@dataclass
class MetricsSummary:
    n_trades: int
    hit_rate: float
    mean_pnl: float
    median_pnl: float
    total_pnl: float
    sharpe: float
    max_drawdown: float
    top_decile_pnl_share: float  # fraction of positive PnL from top 10% of trades


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _max_drawdown(pnls: Sequence[float]) -> float:
    """Max drawdown on the equity curve (cumulative PnL).

    Returned as a positive number (the largest peak-to-trough decline).
    """
    peak = 0.0
    cum = 0.0
    dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = max(dd, peak - cum)
    return dd


def _top_decile_share(pnls: Sequence[float]) -> float:
    wins = [p for p in pnls if p > 0]
    if not wins:
        return 0.0
    wins_sorted = sorted(wins, reverse=True)
    k = max(1, len(wins_sorted) // 10)
    top_sum = sum(wins_sorted[:k])
    total = sum(wins_sorted)
    return top_sum / total if total > 0 else 0.0


def compute_metrics(pnls: Sequence[float]) -> MetricsSummary:
    """Build a `MetricsSummary` from a list of per-trade net PnL values."""
    n = len(pnls)
    if n == 0:
        return MetricsSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hits = sum(1 for p in pnls if p > 0)
    mean = _mean(pnls)
    sd = _stdev(pnls)
    sharpe = (mean / sd) * math.sqrt(n) if sd > 0 else 0.0
    return MetricsSummary(
        n_trades=n,
        hit_rate=hits / n,
        mean_pnl=mean,
        median_pnl=_median(pnls),
        total_pnl=sum(pnls),
        sharpe=sharpe,
        max_drawdown=_max_drawdown(pnls),
        top_decile_pnl_share=_top_decile_share(pnls),
    )
