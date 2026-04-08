"""Control-group generator: volume-quartile matched random markets."""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from ..config import BacktestConfig

_DEFAULT_BT = BacktestConfig()


def _quartile_of(value: float, edges: list[float]) -> int:
    """Return quartile index 0..n-1 for `value` given sorted bin edges."""
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def generate_control_group(
    events: list[dict],
    universe: list[dict],
    n: int = 5,
    config: BacktestConfig = _DEFAULT_BT,
    seed: int | None = None,
) -> list[dict]:
    """Generate `n` volume-quartile-matched control events for each input event.

    Each element of `universe` must have keys: market_id, volume, and a list of
    candidate timestamps (`timestamps`). Each input `event` must have a
    `volume` field (matching volume of the treated market at entry).

    Output: list of synthetic control events shaped like input events with
            fields event_id, market_id, entry_ts, volume, is_control=True.
    """
    rng = random.Random(seed if seed is not None else config.random_seed)

    if not universe:
        return []

    volumes = sorted(m["volume"] for m in universe)
    bins = config.volume_quartile_bins
    if len(volumes) < bins:
        # fall back to whatever we've got
        edges = volumes[:-1] if len(volumes) > 1 else volumes
    else:
        # compute bin edges (exclude max -> last bucket is everything above)
        qs = np.quantile(volumes, np.linspace(1 / bins, 1 - 1 / bins, bins - 1))
        edges = qs.tolist()

    # bucket the universe
    buckets: dict[int, list[dict]] = {i: [] for i in range(bins)}
    for m in universe:
        q = _quartile_of(m["volume"], edges)
        buckets.setdefault(q, []).append(m)

    controls: list[dict] = []
    for ev in events:
        ev_q = _quartile_of(ev["volume"], edges)
        pool = buckets.get(ev_q, []) or [m for b in buckets.values() for m in b]
        if not pool:
            continue
        for _ in range(n):
            m = rng.choice(pool)
            ts_choices = m.get("timestamps") or [ev.get("entry_ts", 0.0)]
            ts = rng.choice(list(ts_choices))
            controls.append(
                {
                    "event_id": f"ctrl::{ev['event_id']}::{len(controls)}",
                    "market_id": m["market_id"],
                    "entry_ts": float(ts),
                    "volume": m["volume"],
                    "quartile": ev_q,
                    "is_control": True,
                }
            )
    return controls


def quartile_distribution(events: Iterable[dict]) -> dict[int, int]:
    """Utility: count events per quartile (for tests)."""
    out: dict[int, int] = {}
    for e in events:
        q = e.get("quartile", -1)
        out[q] = out.get(q, 0) + 1
    return out
