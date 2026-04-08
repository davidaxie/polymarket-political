import math

from polymarket_political.report.metrics import compute_metrics, MetricsSummary


def test_metrics_empty():
    m = compute_metrics([])
    assert m.n_trades == 0
    assert m.hit_rate == 0.0
    assert m.total_pnl == 0.0


def test_metrics_golden_numbers():
    pnls = [10, -5, 20, -10, 30]  # 3 wins / 5
    m = compute_metrics(pnls)
    assert m.n_trades == 5
    assert abs(m.hit_rate - 0.6) < 1e-9
    assert m.total_pnl == 45
    assert abs(m.mean_pnl - 9.0) < 1e-9
    assert m.median_pnl == 10
    # max drawdown: cum = [10, 5, 25, 15, 45] -> peak 25, trough 15 -> dd 10
    assert m.max_drawdown == 10


def test_metrics_sharpe_positive_for_consistent_wins():
    m = compute_metrics([10, 10, 10, 10, 10])
    # zero stdev -> sharpe 0 by our convention
    assert m.sharpe == 0.0
    m2 = compute_metrics([10, 12, 8, 11, 9])
    assert m2.sharpe > 0


def test_top_decile_concentration():
    # 10 winners, one huge one
    pnls = [1] * 9 + [100]
    m = compute_metrics(pnls)
    # top 10% of 10 wins = 1 trade (the 100) -> 100 / 109
    assert abs(m.top_decile_pnl_share - (100 / 109)) < 1e-9


def test_metrics_summary_is_dataclass():
    m = compute_metrics([1, 2, 3])
    assert isinstance(m, MetricsSummary)
