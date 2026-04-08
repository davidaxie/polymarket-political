from polymarket_insider.backtest.control import (
    generate_control_group,
    quartile_distribution,
)


def _make_universe(n=40):
    # volumes evenly spaced 100..4000
    return [
        {
            "market_id": f"u{i}",
            "volume": 100 * (i + 1),
            "timestamps": [1000.0 + i, 2000.0 + i],
        }
        for i in range(n)
    ]


def test_control_group_size_matches_n():
    events = [
        {"event_id": "e1", "volume": 150, "entry_ts": 1000.0},
        {"event_id": "e2", "volume": 3500, "entry_ts": 1000.0},
    ]
    ctrls = generate_control_group(events, _make_universe(), n=5, seed=1)
    assert len(ctrls) == 10  # 2 events * 5 controls


def test_control_group_quartile_matching():
    universe = _make_universe(40)
    # low-volume event -> quartile 0
    low_event = {"event_id": "lo", "volume": 150, "entry_ts": 1000.0}
    # high-volume event -> quartile 3
    hi_event = {"event_id": "hi", "volume": 3900, "entry_ts": 1000.0}
    ctrls_lo = generate_control_group([low_event], universe, n=20, seed=42)
    ctrls_hi = generate_control_group([hi_event], universe, n=20, seed=42)
    # all lo controls should have small volumes, all hi should have large
    assert max(c["volume"] for c in ctrls_lo) <= 1000
    assert min(c["volume"] for c in ctrls_hi) >= 3000
    dist_lo = quartile_distribution(ctrls_lo)
    dist_hi = quartile_distribution(ctrls_hi)
    assert 0 in dist_lo and dist_lo[0] == 20
    assert 3 in dist_hi and dist_hi[3] == 20


def test_control_group_empty_universe():
    assert generate_control_group([{"event_id": "x", "volume": 1}], [], n=3) == []
