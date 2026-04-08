"""Trade ledger: dataclass + simulation helper."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..config import BacktestConfig, CostConfig
from .costs import ProvisionalCostConfig, provisional_total_costs, total_costs

_DEFAULT_COST = CostConfig()
_DEFAULT_BT = BacktestConfig()


@dataclass
class Trade:
    """Single simulated trade with multi-horizon exits and gross/net PnL."""

    event_id: str
    market_id: str
    entry_ts: float
    entry_price: float
    size_usd: float

    # exit prices by horizon
    exit_24h: float | None = None
    exit_48h: float | None = None
    exit_72h: float | None = None
    exit_resolution: float | None = None

    # gross PnL per horizon (USD)
    pnl_gross_24h: float = 0.0
    pnl_gross_48h: float = 0.0
    pnl_gross_72h: float = 0.0
    pnl_gross_resolution: float = 0.0

    # net PnL per horizon (USD, after costs)
    pnl_net_24h: float = 0.0
    pnl_net_48h: float = 0.0
    pnl_net_72h: float = 0.0
    pnl_net_resolution: float = 0.0

    is_resolved: bool = False
    cost_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gross_pnl(entry_price: float, exit_price: float, size_usd: float) -> float:
    """PnL on a YES-position bought at entry_price, sold at exit_price.

    Units: a prediction-market share has payoff in [0, 1]. Number of shares
    = size_usd / entry_price. PnL = shares * (exit - entry).
    """
    if entry_price <= 0:
        return 0.0
    shares = size_usd / entry_price
    return shares * (exit_price - entry_price)


def simulate_trade(
    event: dict,
    market_data: dict,
    config: CostConfig = _DEFAULT_COST,
    bt_config: BacktestConfig = _DEFAULT_BT,
) -> Trade:
    """Build a Trade from a synthetic event + market data snapshot.

    `event` keys: event_id, market_id, entry_ts, entry_price, size_usd (optional).
    `market_data` keys: orderbook {bid, ask}, depth, exit_24h, exit_48h, exit_72h,
                        exit_resolution, is_resolved.
    """
    size_usd = event.get("size_usd", bt_config.default_size_usd)
    entry_price = float(event["entry_price"])

    e24 = market_data.get("exit_24h")
    e48 = market_data.get("exit_48h")
    e72 = market_data.get("exit_72h")
    eres = market_data.get("exit_resolution")

    gross_24 = _gross_pnl(entry_price, e24, size_usd) if e24 is not None else 0.0
    gross_48 = _gross_pnl(entry_price, e48, size_usd) if e48 is not None else 0.0
    gross_72 = _gross_pnl(entry_price, e72, size_usd) if e72 is not None else 0.0
    gross_res = _gross_pnl(entry_price, eres, size_usd) if eres is not None else 0.0

    is_resolved = bool(market_data.get("is_resolved", False))

    # Costs (spread/slippage/gas apply to every horizon; fee only at resolution)
    base = total_costs(
        size_usd=size_usd,
        entry_price=entry_price,
        orderbook=market_data["orderbook"],
        depth=market_data.get("depth", size_usd),
        gross_pnl=0.0,
        is_resolved=False,
        config=config,
    )
    non_fee = base["total"]  # spread + slippage + gas, no fee

    # For the resolution horizon, add the fee
    res_costs = total_costs(
        size_usd=size_usd,
        entry_price=entry_price,
        orderbook=market_data["orderbook"],
        depth=market_data.get("depth", size_usd),
        gross_pnl=gross_res,
        is_resolved=is_resolved,
        config=config,
    )

    return Trade(
        event_id=event["event_id"],
        market_id=event["market_id"],
        entry_ts=float(event["entry_ts"]),
        entry_price=entry_price,
        size_usd=size_usd,
        exit_24h=e24,
        exit_48h=e48,
        exit_72h=e72,
        exit_resolution=eres,
        pnl_gross_24h=gross_24,
        pnl_gross_48h=gross_48,
        pnl_gross_72h=gross_72,
        pnl_gross_resolution=gross_res,
        pnl_net_24h=gross_24 - non_fee,
        pnl_net_48h=gross_48 - non_fee,
        pnl_net_72h=gross_72 - non_fee,
        pnl_net_resolution=gross_res - res_costs["total"],
        is_resolved=is_resolved,
        cost_breakdown=res_costs,
    )


def simulate_trade_provisional(
    event: dict,
    market_data: dict,
    config: ProvisionalCostConfig | None = None,
    bt_config: BacktestConfig = _DEFAULT_BT,
) -> Trade:
    """Mid-price-only variant of :func:`simulate_trade`.

    Uses :class:`ProvisionalCostConfig` — NO orderbook is consulted. Required
    keys in ``market_data``: ``exit_24h``, ``exit_48h``, ``exit_72h``,
    ``exit_resolution`` (any may be ``None``), ``is_resolved``.

    PROVISIONAL: any PnL this function produces must be labeled accordingly.
    """
    cfg = config or ProvisionalCostConfig()
    size_usd = event.get("size_usd", bt_config.default_size_usd)
    entry_price = float(event["entry_price"])

    e24 = market_data.get("exit_24h")
    e48 = market_data.get("exit_48h")
    e72 = market_data.get("exit_72h")
    eres = market_data.get("exit_resolution")

    gross_24 = _gross_pnl(entry_price, e24, size_usd) if e24 is not None else 0.0
    gross_48 = _gross_pnl(entry_price, e48, size_usd) if e48 is not None else 0.0
    gross_72 = _gross_pnl(entry_price, e72, size_usd) if e72 is not None else 0.0
    gross_res = _gross_pnl(entry_price, eres, size_usd) if eres is not None else 0.0

    is_resolved = bool(market_data.get("is_resolved", False))

    base = provisional_total_costs(
        size_usd=size_usd,
        gross_pnl=0.0,
        is_resolved=False,
        config=cfg,
    )
    non_fee = base["total"]
    res_costs = provisional_total_costs(
        size_usd=size_usd,
        gross_pnl=gross_res,
        is_resolved=is_resolved,
        config=cfg,
    )

    return Trade(
        event_id=event["event_id"],
        market_id=event["market_id"],
        entry_ts=float(event["entry_ts"]),
        entry_price=entry_price,
        size_usd=size_usd,
        exit_24h=e24,
        exit_48h=e48,
        exit_72h=e72,
        exit_resolution=eres,
        pnl_gross_24h=gross_24,
        pnl_gross_48h=gross_48,
        pnl_gross_72h=gross_72,
        pnl_gross_resolution=gross_res,
        pnl_net_24h=gross_24 - non_fee,
        pnl_net_48h=gross_48 - non_fee,
        pnl_net_72h=gross_72 - non_fee,
        pnl_net_resolution=gross_res - res_costs["total"],
        is_resolved=is_resolved,
        cost_breakdown=res_costs,
    )
