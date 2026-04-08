from polymarket_political.backtest.costs import (
    apply_fees,
    apply_spread,
    apply_slippage,
    apply_gas,
    total_costs,
)
from polymarket_political.config import CostConfig


def test_fee_2pct_on_250_winnings():
    # 2% on $250 = $5
    assert apply_fees(250.0, is_resolved=True) == 5.0


def test_fee_not_applied_when_unresolved():
    assert apply_fees(250.0, is_resolved=False) == 0.0


def test_fee_not_applied_on_losses():
    assert apply_fees(-100.0, is_resolved=True) == 0.0


def test_spread_full_on_entry_half_on_exit():
    entry, exit_ = apply_spread(0.5, {"bid": 0.49, "ask": 0.51})
    assert abs(entry - 0.02) < 1e-9
    assert abs(exit_ - 0.01) < 1e-9


def test_slippage_linear_and_capped():
    cfg = CostConfig(slippage_coef=0.5, slippage_cap=0.05)
    # size 10, depth 100 -> 0.5 * 0.1 = 0.05 -> at cap
    assert apply_slippage(10, 100, cfg) == 0.05
    # size 1, depth 100 -> 0.5 * 0.01 = 0.005
    assert abs(apply_slippage(1, 100, cfg) - 0.005) < 1e-9
    # zero depth -> cap
    assert apply_slippage(1, 0, cfg) == 0.05


def test_gas_fixed_one_dollar():
    assert apply_gas() == 1.0


def test_total_costs_breakdown_resolved_winner():
    cfg = CostConfig()
    out = total_costs(
        size_usd=100,
        entry_price=0.5,
        orderbook={"bid": 0.49, "ask": 0.51},
        depth=1000,
        gross_pnl=50,
        is_resolved=True,
        config=cfg,
    )
    # spread: (0.02 + 0.01) * 100 = 3.0
    assert abs(out["spread"] - 3.0) < 1e-9
    # slippage: 0.5 * (100/1000) = 0.05 -> cap 0.05 -> * 100 * 2 = 10
    assert abs(out["slippage"] - 10.0) < 1e-9
    assert out["gas"] == 1.0
    # fee: 2% of 50 = 1
    assert abs(out["fee"] - 1.0) < 1e-9
    assert abs(out["total"] - 15.0) < 1e-9
