"""Feature engineering for the insider-flow signal.

All functions are pure: they take a DataFrame in, return a DataFrame out,
and never touch global state or I/O. Inputs are expected to have at minimum
a ``timestamp`` column (datetime64[ns, UTC]) and a ``market_id`` column.
"""
from __future__ import annotations

import pandas as pd


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)


def rolling_hourly_volume(
    trades: pd.DataFrame, window: str = "7d"
) -> pd.DataFrame:
    """Compute per-market 1h volume and its rolling baseline.

    Parameters
    ----------
    trades:
        Columns: ``timestamp``, ``market_id``, ``size`` (notional traded).
    window:
        Pandas offset string controlling the rolling baseline length.

    Returns
    -------
    DataFrame indexed 0..N with columns:
        ``timestamp``, ``market_id``, ``hourly_volume``, ``baseline_volume``,
        ``volume_ratio`` (hourly / baseline; ``inf`` when baseline==0).
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "market_id",
                "hourly_volume",
                "baseline_volume",
                "volume_ratio",
            ]
        )

    df = _ensure_sorted(trades.copy())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Resample hourly per market.
    hourly = (
        df.set_index("timestamp")
        .groupby("market_id")["size"]
        .resample("1h")
        .sum()
        .rename("hourly_volume")
        .reset_index()
    )

    # Rolling mean baseline of the hourly volume, per market.
    baselines: list[pd.DataFrame] = []
    for mkt, grp in hourly.groupby("market_id"):
        grp = grp.sort_values("timestamp").copy()
        grp["baseline_volume"] = (
            grp.set_index("timestamp")["hourly_volume"]
            .rolling(window, closed="left")
            .mean()
            .values
        )
        baselines.append(grp)

    out = pd.concat(baselines, ignore_index=True)
    out["volume_ratio"] = out["hourly_volume"] / out["baseline_volume"]
    return out


def compute_1h_price_move(orderbook: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute 1h price move from an orderbook/mid snapshot series.

    Parameters
    ----------
    orderbook:
        Columns: ``timestamp``, ``market_id``, ``mid_price`` (in 0-1 units).

    Returns
    -------
    DataFrame with columns ``timestamp``, ``market_id``, ``price_1h_ago``,
    ``price_move`` (signed), ``abs_price_move``.
    """
    if orderbook.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "market_id",
                "price_1h_ago",
                "price_move",
                "abs_price_move",
            ]
        )

    df = _ensure_sorted(orderbook.copy())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    pieces: list[pd.DataFrame] = []
    for mkt, grp in df.groupby("market_id"):
        grp = grp.sort_values("timestamp").copy()
        s = grp.set_index("timestamp")["mid_price"]
        lagged = s.reindex(
            s.index, method=None
        )  # placeholder to get same index
        # use asof-style lookup: for each timestamp t, find price at t - 1h
        target = s.index - pd.Timedelta("1h")
        prev = s.reindex(s.index.union(target)).ffill().reindex(target).values
        grp["price_1h_ago"] = prev
        grp["price_move"] = grp["mid_price"] - grp["price_1h_ago"]
        grp["abs_price_move"] = grp["price_move"].abs()
        pieces.append(
            grp[
                [
                    "timestamp",
                    "market_id",
                    "price_1h_ago",
                    "price_move",
                    "abs_price_move",
                ]
            ]
        )

    return pd.concat(pieces, ignore_index=True)


def compute_taker_imbalance(
    trades: pd.DataFrame, window: str = "1h"
) -> pd.DataFrame:
    """Compute taker buy/sell imbalance over a rolling window.

    Parameters
    ----------
    trades:
        Columns: ``timestamp``, ``market_id``, ``size``, ``side``
        where ``side`` is ``"buy"`` or ``"sell"``.
    window:
        Rolling window length as a pandas offset string.

    Returns
    -------
    DataFrame indexed per hourly bucket with columns ``timestamp``,
    ``market_id``, ``buy_volume``, ``sell_volume``, ``imbalance``
    (signed = (buy - sell) / total, in [-1, 1]).
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "market_id",
                "buy_volume",
                "sell_volume",
                "imbalance",
            ]
        )

    df = _ensure_sorted(trades.copy())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["buy_size"] = df["size"].where(df["side"] == "buy", 0.0)
    df["sell_size"] = df["size"].where(df["side"] == "sell", 0.0)

    agg = (
        df.set_index("timestamp")
        .groupby("market_id")[["buy_size", "sell_size"]]
        .resample(window)
        .sum()
        .rename(columns={"buy_size": "buy_volume", "sell_size": "sell_volume"})
        .reset_index()
    )
    total = agg["buy_volume"] + agg["sell_volume"]
    agg["imbalance"] = (agg["buy_volume"] - agg["sell_volume"]) / total.where(
        total > 0, other=pd.NA
    )
    agg["imbalance"] = agg["imbalance"].astype(float).fillna(0.0)
    return agg


def rolling_hourly_volatility(
    prices: pd.DataFrame, window: str = "7d"
) -> pd.DataFrame:
    """Compute per-market hourly realized volatility and its rolling baseline.

    This is the prices-only replacement for both ``rolling_hourly_volume``
    (derived from trades) and ``compute_taker_imbalance``. Elevated realized
    volatility is a well-established proxy for informed trading activity —
    large directional flows create price impact that shows up as higher
    return variance.

    Parameters
    ----------
    prices:
        Columns: ``timestamp`` (datetime64[ns, UTC]), ``market_id``,
        ``mid_price`` (float, 0-1 probability units). Should be at
        fine fidelity (1-minute) for good intra-hour statistics.
    window:
        Rolling baseline length for the volatility ratio.

    Returns
    -------
    DataFrame with columns: ``timestamp``, ``market_id``,
    ``hourly_volatility`` (std of log returns within the hour),
    ``baseline_volatility`` (rolling mean of hourly vol),
    ``volatility_ratio`` (hourly / baseline; ``inf`` if baseline==0).
    """
    empty_cols = [
        "timestamp", "market_id", "hourly_volatility",
        "baseline_volatility", "volatility_ratio",
    ]
    if prices.empty:
        return pd.DataFrame(columns=empty_cols)

    df = _ensure_sorted(prices.copy())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Log returns per market — handles 0-prices gracefully.
    pieces: list[pd.DataFrame] = []
    for mkt, grp in df.groupby("market_id"):
        grp = grp.sort_values("timestamp").copy()
        grp["log_ret"] = grp["mid_price"].apply(
            lambda p: float("nan") if p <= 0 else p
        ).pipe(lambda s: s.pct_change().abs())  # absolute return magnitude
        pieces.append(grp)

    if not pieces:
        return pd.DataFrame(columns=empty_cols)
    df = pd.concat(pieces, ignore_index=True)

    # Hourly realized volatility = std of returns within each hour.
    hourly = (
        df.set_index("timestamp")
        .groupby("market_id")["log_ret"]
        .resample("1h")
        .std()
        .rename("hourly_volatility")
        .reset_index()
    )
    hourly["hourly_volatility"] = hourly["hourly_volatility"].fillna(0.0)

    # Rolling baseline of the hourly volatility.
    baselines: list[pd.DataFrame] = []
    for mkt, grp in hourly.groupby("market_id"):
        grp = grp.sort_values("timestamp").copy()
        grp["baseline_volatility"] = (
            grp.set_index("timestamp")["hourly_volatility"]
            .rolling(window, closed="left")
            .mean()
            .values
        )
        baselines.append(grp)

    out = pd.concat(baselines, ignore_index=True)
    out["volatility_ratio"] = out["hourly_volatility"] / out["baseline_volatility"]
    return out
