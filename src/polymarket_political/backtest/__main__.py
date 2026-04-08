"""CLI entry point for the Workstream D backtest pipeline.

Usage::

    python -m polymarket_political.backtest run \
        --window training --out reports/week3_first_run.md

The ``run`` subcommand invokes :func:`polymarket_political.pipeline.run_backtest`
and writes three artifacts:

1. A markdown report at ``--out`` (starts with the provisional-data banner).
2. ``data/processed/ledger_<window>.parquet`` — every simulated trade.
3. ``data/processed/metrics_<window>.parquet`` — metric summaries (signal + control).

All metrics are labeled "provisional — mid-price only, fixed-spread cost model".
Do not trust them until Workstream E upgrades the cost model.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..pipeline import BacktestResult, run_backtest
from ..report.metrics import MetricsSummary

log = logging.getLogger(__name__)


# ------- report rendering -----------------------------------------------------


def _format_metrics(m: MetricsSummary) -> str:
    return (
        f"- n_trades: **{m.n_trades}**\n"
        f"- hit_rate: **{m.hit_rate:.1%}**\n"
        f"- mean PnL: **${m.mean_pnl:.2f}**\n"
        f"- median PnL: **${m.median_pnl:.2f}**\n"
        f"- total PnL: **${m.total_pnl:.2f}**\n"
        f"- Sharpe (per-trade, √N): **{m.sharpe:.2f}**\n"
        f"- max drawdown: **${m.max_drawdown:.2f}**\n"
        f"- top-decile PnL share: **{m.top_decile_pnl_share:.1%}**\n"
    )


def _format_signal_vs_control_table(result: BacktestResult) -> str:
    sig = result.signal_metrics
    cf = result.control_metrics_filtered
    cr = result.control_metrics_raw
    rows = [
        "| Group | n_trades | hit_rate | mean PnL | total PnL | Sharpe |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Signal (news-filtered) | {sig.n_trades} | "
            f"{sig.hit_rate:.1%} | ${sig.mean_pnl:.2f} | "
            f"${sig.total_pnl:.2f} | {sig.sharpe:.2f} |"
        ),
        (
            f"| Control (news-filtered) | {cf.n_trades} | "
            f"{cf.hit_rate:.1%} | ${cf.mean_pnl:.2f} | "
            f"${cf.total_pnl:.2f} | {cf.sharpe:.2f} |"
        ),
        (
            f"| Control (raw, legacy) | {cr.n_trades} | "
            f"{cr.hit_rate:.1%} | ${cr.mean_pnl:.2f} | "
            f"${cr.total_pnl:.2f} | {cr.sharpe:.2f} |"
        ),
    ]
    table = "\n".join(rows)

    # Material-spread warning: >20% PnL spread between the two controls.
    warning = ""
    if result.news_filter_applied_to_controls:
        denom = max(abs(cr.total_pnl), 1e-9)
        spread = abs(cf.total_pnl - cr.total_pnl) / denom
        if spread > 0.20:
            warning = (
                "\n\n> ⚠️ **News-filter asymmetry warning.** The filtered and "
                f"raw control totals differ by {spread:.0%} "
                f"(filtered=${cf.total_pnl:.2f} vs raw=${cr.total_pnl:.2f}). "
                "News-filter asymmetry was masking or unmasking apparent edge "
                "in the signal-vs-control comparison — trust the filtered row."
            )
    return table + warning


def _gate_recommendation(result: BacktestResult) -> tuple[str, str]:
    """Week 3 gate logic.

    Per RESEARCH_PLAN.md Week 3: "Signal must at minimum be directionally
    better than control on the 48h metric, pre-cost. If it's not, the
    thresholds are wrong and Week 4 is a parameter sweep instead of iteration."

    We evaluate on the same-horizon net PnL (48h) since we only have the
    post-cost numbers in the summary — and add an explicit caveat.
    """
    sig_n = result.signal_metrics.n_trades
    if sig_n == 0:
        return "red", "No events detected. Thresholds are too strict or ingestion broke."
    sig_mean = result.signal_metrics.mean_pnl
    ctrl_mean = result.control_metrics_filtered.mean_pnl
    sig_hit = result.signal_metrics.hit_rate
    ctrl_hit = result.control_metrics_filtered.hit_rate
    if sig_mean > ctrl_mean and sig_hit >= ctrl_hit:
        return (
            "green",
            "Signal beats control on both mean PnL and hit rate at 48h. "
            "Proceed to Week 4 parameter iteration (one round).",
        )
    if sig_mean > ctrl_mean or sig_hit >= ctrl_hit:
        return (
            "amber",
            "Signal beats control on one of {mean, hit-rate} but not both. "
            "Do a parameter sweep in Week 4 and retest.",
        )
    return (
        "red",
        "Signal does not beat control on either mean or hit rate at 48h. "
        "Thresholds are wrong — Week 4 is a full sweep, not iteration.",
    )


def render_report(result: BacktestResult) -> str:
    """Render the markdown report — the CLI's default output template."""
    gate_color, gate_msg = _gate_recommendation(result)
    gate_emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}[gate_color]

    # Top-5 events by abs_price_move
    top_events_md = "_no events_"
    if result.events:
        ev_df = result.to_events_frame()
        ev_df = ev_df.sort_values("abs_price_move", ascending=False).head(5)
        rows = [
            "| market_id | timestamp | volume_ratio | abs_price_move | imbalance |",
            "|---|---|---:|---:|---:|",
        ]
        for _, e in ev_df.iterrows():
            rows.append(
                f"| `{e['market_id']}` | {e['timestamp']} | "
                f"{e['volume_ratio']:.2f} | {e['abs_price_move']:.3f} | "
                f"{e['imbalance']:.2f} |"
            )
        top_events_md = "\n".join(rows)

    warnings_md = (
        "\n".join(f"- {w}" for w in result.warnings)
        if result.warnings
        else "_none_"
    )

    config_json = json.dumps(result.config_snapshot, indent=2, default=str)

    return f"""# Week 3 — First Backtest Run (PROVISIONAL)

> ⚠️ **PROVISIONAL DATA — DO NOT CIRCULATE AS A RESULT.**
>
> This backtest uses **Polymarket /prices-history mid-price only** (no bid/ask
> depth) and a **fixed-spread cost model** (`ProvisionalCostConfig`: 2¢
> round-trip spread, 0.5¢ per-side slippage, 2% fee, $1 gas). All metrics
> below are labeled:
>
> > *{result.provisional_label}*
>
> Replace with a depth-aware cost model (Workstream E) before treating any
> number here as a real edge estimate.

## Config snapshot

**Window:** `{result.window}` — `{result.start}` → `{result.end}`

```json
{config_json}
```

## Universe

- Markets after liquidity floor: **{result.universe_size}**
- Events detected: **{len(result.events)}**
- Simulated trades (signal): **{len(result.ledger)}**
- Simulated trades (control, filtered): **{len(result.control_ledger)}**
- Simulated trades (control, raw): **{len(result.control_ledger_raw)}**

## Signal metrics (48h net PnL, provisional)

{_format_metrics(result.signal_metrics)}

## Signal vs Control (two views)

Two comparisons are reported so the reader can see whether the news-filter
asymmetry was masking or unmasking edge.

1. **Filtered control** — the same news filter applied to the signal group is
   also applied to the control group. This is the apples-to-apples comparison.
2. **Raw control** — the legacy, unfiltered control group. Kept for
   transparency only.

{_format_signal_vs_control_table(result)}

## Control metrics — filtered (48h net PnL, provisional)

{_format_metrics(result.control_metrics_filtered)}

## Control metrics — raw, unfiltered (48h net PnL, provisional, legacy)

{_format_metrics(result.control_metrics_raw)}

## Top-5 events (by |Δprice|)

{top_events_md}

## Warnings

{warnings_md}

## Caveats

- Mid-price-only: entry/exit prices are modeled mid-points with no bid/ask depth.
- Fixed-spread cost model: numbers are indicative, not predictive.
- Control group is volume-quartile matched random markets + timestamps. When
  ``SignalConfig.news_filter_enabled`` is True, the same news filter applied
  to the signal group is also applied to the control group (see "Signal vs
  Control — two views" above). The raw/unfiltered control is retained for
  transparency.
- Per the Research Plan, any parameter tuning based on these results must
  happen on the **training** window only. The holdout stays sealed.

## Week 3 Gate Recommendation

{gate_emoji} **{gate_color.upper()}** — {gate_msg}
"""


# ------- CLI plumbing ---------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m polymarket_political.backtest",
        description="Run the Workstream D end-to-end backtest pipeline.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the backtest pipeline and emit a report.")
    run.add_argument(
        "--window",
        default="training",
        choices=["training", "holdout", "custom"],
        help="Which window to run. Holdout is sealed — use with care.",
    )
    run.add_argument("--start", default=None, help="ISO date (required for --window custom)")
    run.add_argument("--end", default=None, help="ISO date (required for --window custom)")
    run.add_argument(
        "--out",
        default="reports/week3_first_run.md",
        help="Output markdown report path.",
    )
    run.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Cap the universe for a fast smoke run.",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of ledger/metrics parquets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if args.command != "run":
        _build_arg_parser().print_help()
        return 1

    result = run_backtest(
        start=args.start,
        end=args.end,
        window=args.window,
        max_markets=args.max_markets,
        cache_intermediate=not args.no_cache,
    )
    md = render_report(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    log.info("wrote report to %s", out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
