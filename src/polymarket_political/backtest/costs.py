"""Cost model: fees, spread, slippage, gas.

Two modes:

1. **Real-depth mode** (default) — takes a live orderbook dict with ``bid``/``ask``
   and a ``depth`` number, and computes spread + linear slippage from them.
   This is what will be used once Workstream E lands the Polygon subgraph.

2. **Provisional mode** — :class:`ProvisionalCostConfig` pins spread and
   slippage to FIXED per-trade fractions so the Week-3 backtest can run with
   only the ``/prices-history`` mid-price series as input (see
   ``ingest/prices.py``). Metrics produced via this path MUST be labeled
   "provisional — mid-price only, fixed-spread cost model" in every report.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import CostConfig

_DEFAULT = CostConfig()


# --- Provisional (mid-price-only) cost config ---------------------------------


@dataclass(frozen=True)
class ProvisionalCostConfig:
    """Fixed-cost overrides used when we only have a mid-price series.

    PROVISIONAL: these numbers are deliberately conservative placeholders. They
    should be replaced with a depth-aware model (Workstream E) before any
    result is treated as a live edge estimate.

    Defaults (per the Workstream D spec):
      - 2¢ round-trip spread (1¢ each side)
      - 0.5¢ slippage per side
      - 2% fee + $1 gas from the baseline :class:`CostConfig`
    """

    fee_rate: float = 0.02
    gas_cost: float = 1.0

    # spread, expressed in probability units (same scale as mid prices)
    entry_spread: float = 0.01  # 1¢
    exit_spread: float = 0.01  # 1¢

    # slippage, as a FRACTION of notional per side (so 0.005 = 0.5%)
    entry_slippage_frac: float = 0.005
    exit_slippage_frac: float = 0.005

    label: str = (
        "provisional — mid-price only, fixed-spread cost model"
    )


def provisional_total_costs(
    *,
    size_usd: float,
    gross_pnl: float,
    is_resolved: bool,
    config: ProvisionalCostConfig | None = None,
) -> dict:
    """Cost breakdown for the Week-3 mid-price-only backtest path.

    Does NOT take an orderbook dict — we don't have one. Spread is charged per
    side as a fraction of notional, slippage is fixed per side as a fraction
    of notional. Same fee/gas as the real-depth path.
    """
    cfg = config or ProvisionalCostConfig()
    spread_cost = (cfg.entry_spread + cfg.exit_spread) * size_usd
    slippage_cost = (cfg.entry_slippage_frac + cfg.exit_slippage_frac) * size_usd
    gas = cfg.gas_cost
    fee = (gross_pnl * cfg.fee_rate) if (is_resolved and gross_pnl > 0) else 0.0
    return {
        "spread": spread_cost,
        "slippage": slippage_cost,
        "gas": gas,
        "fee": fee,
        "total": spread_cost + slippage_cost + gas + fee,
        "provisional": True,
        "label": cfg.label,
    }


def apply_fees(pnl: float, is_resolved: bool, config: CostConfig = _DEFAULT) -> float:
    """Polymarket 2% fee on winnings, only on resolved markets.

    Returns the fee amount (>= 0). Fee only applies to positive PnL on resolved markets.
    """
    if not is_resolved:
        return 0.0
    if pnl <= 0:
        return 0.0
    return pnl * config.fee_rate


def apply_spread(
    entry_price: float,
    orderbook: dict,
    config: CostConfig = _DEFAULT,
) -> tuple[float, float]:
    """Pay full spread on entry, half on exit.

    `orderbook` is a dict with keys `bid` and `ask`.
    Returns `(entry_cost, exit_cost)` in USD-per-unit-size (fraction of notional).
    """
    bid = float(orderbook["bid"])
    ask = float(orderbook["ask"])
    spread = max(ask - bid, 0.0)
    entry_cost = spread * config.entry_spread_frac
    exit_cost = spread * config.exit_spread_frac
    return entry_cost, exit_cost


def apply_slippage(
    size: float,
    top_of_book_depth: float,
    config: CostConfig = _DEFAULT,
) -> float:
    """Linear slippage in (size / depth), capped.

    Returns slippage as a fraction of notional (0..cap).
    """
    if top_of_book_depth <= 0:
        return config.slippage_cap
    raw = config.slippage_coef * (size / top_of_book_depth)
    return min(raw, config.slippage_cap)


def apply_gas(config: CostConfig = _DEFAULT) -> float:
    """Fixed round-trip gas cost in USD."""
    return config.gas_cost


def total_costs(
    *,
    size_usd: float,
    entry_price: float,
    orderbook: dict,
    depth: float,
    gross_pnl: float,
    is_resolved: bool,
    config: CostConfig = _DEFAULT,
) -> dict:
    """Compute full cost breakdown for a round-trip trade.

    All components in USD. Spread/slippage scaled by notional (size_usd).
    """
    entry_spread, exit_spread = apply_spread(entry_price, orderbook, config)
    spread_cost = (entry_spread + exit_spread) * size_usd
    slip_frac = apply_slippage(size_usd, depth, config)
    slippage_cost = slip_frac * size_usd * 2  # entry + exit
    gas = apply_gas(config)
    fee = apply_fees(gross_pnl, is_resolved, config)
    return {
        "spread": spread_cost,
        "slippage": slippage_cost,
        "gas": gas,
        "fee": fee,
        "total": spread_cost + slippage_cost + gas + fee,
    }
