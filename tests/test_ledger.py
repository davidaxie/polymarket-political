from polymarket_political.backtest.ledger import Trade, simulate_trade, _gross_pnl


def test_gross_pnl_basic():
    # buy at 0.50, exit at 0.60, size $100 -> 200 shares * 0.10 = $20
    assert abs(_gross_pnl(0.5, 0.6, 100) - 20.0) < 1e-9


def test_simulate_trade_fields_populated():
    event = {
        "event_id": "e1",
        "market_id": "m1",
        "entry_ts": 1000.0,
        "entry_price": 0.5,
        "size_usd": 100,
    }
    market_data = {
        "orderbook": {"bid": 0.49, "ask": 0.51},
        "depth": 10000,
        "exit_24h": 0.55,
        "exit_48h": 0.58,
        "exit_72h": 0.60,
        "exit_resolution": 1.0,
        "is_resolved": True,
    }
    t = simulate_trade(event, market_data)
    assert isinstance(t, Trade)
    assert t.event_id == "e1"
    assert t.market_id == "m1"
    # gross PnL at 24h: 200 shares * (0.55-0.50) = 10
    assert abs(t.pnl_gross_24h - 10.0) < 1e-9
    # gross at resolution: 200 * (1.0 - 0.5) = 100
    assert abs(t.pnl_gross_resolution - 100.0) < 1e-9
    # net < gross (costs applied)
    assert t.pnl_net_resolution < t.pnl_gross_resolution
    # cost breakdown has fee since resolved winner
    assert t.cost_breakdown["fee"] > 0
    assert t.is_resolved is True


def test_simulate_trade_unresolved_no_fee():
    event = {"event_id": "e2", "market_id": "m2", "entry_ts": 0.0, "entry_price": 0.5}
    md = {
        "orderbook": {"bid": 0.5, "ask": 0.5},
        "depth": 10000,
        "exit_24h": 0.5,
        "exit_resolution": 0.6,
        "is_resolved": False,
    }
    t = simulate_trade(event, md)
    assert t.cost_breakdown["fee"] == 0.0
