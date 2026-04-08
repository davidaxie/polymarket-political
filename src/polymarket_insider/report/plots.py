"""Matplotlib plots saved to reports/."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _ensure(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_pnl_curve(pnls: Sequence[float], out_dir: str | Path = "reports") -> Path:
    out = _ensure(out_dir) / "pnl_curve.png"
    cum = []
    running = 0.0
    for p in pnls:
        running += p
        cum.append(running)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Cumulative PnL")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (USD)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_hit_rate_bar(
    hit_rates: dict[str, float], out_dir: str | Path = "reports"
) -> Path:
    out = _ensure(out_dir) / "hit_rate.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(hit_rates.keys())
    vals = [hit_rates[k] for k in labels]
    ax.bar(labels, vals, color="#4c72b0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hit rate")
    ax.set_title("Hit rate by bucket")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_pnl_distribution(
    pnls: Sequence[float], out_dir: str | Path = "reports", bins: int = 30
) -> Path:
    out = _ensure(out_dir) / "pnl_distribution.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(list(pnls), bins=bins, color="#55a868", edgecolor="black")
    ax.axvline(0, color="red", lw=1)
    ax.set_xlabel("Per-trade PnL (USD)")
    ax.set_ylabel("Count")
    ax.set_title("PnL distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
