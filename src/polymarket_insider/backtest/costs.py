"""Cost model: fees, spread, slippage, gas."""
from __future__ import annotations

from ..config import CostConfig

_DEFAULT = CostConfig()


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
