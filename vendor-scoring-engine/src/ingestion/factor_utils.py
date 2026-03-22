"""Utilities for computing stock-level factor exposure proxies.

Provides compute_factor_proxies() which derives cross-sectional factor
scores from universe metadata and optional price data. The resulting
DataFrame serves as the right-hand-side variables for the orthogonality
module's neutralization regression.

All continuous factor columns are cross-sectionally z-scored per date
so that regression coefficients are in comparable units.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_factor_proxies(
    universe: pd.DataFrame,
    dates: list,
    price_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build stock-level factor exposure proxies for orthogonality analysis.

    Computes the following factors for each (ticker, date) pair:
    - size: Cross-sectional z-score of log(market_cap)
    - sector_XXX: Binary sector dummy variables (one per sector present)
    - momentum (optional): Cross-sectionally z-scored trailing 1-month
      return if price_returns contains a ``ret_1m`` column

    Size is computed from the static universe snapshot (market cap does
    not change across dates here). For production use, callers should
    provide a point-in-time universe per date.

    Args:
        universe: DataFrame with required columns [ticker, market_cap]
            and optional [sector].
        dates: List of dates for which to produce factor rows.
        price_returns: Optional DataFrame with [ticker, date, ret_1m]
            for a momentum proxy. If None, momentum is excluded.

    Returns:
        DataFrame with columns [ticker, date, size, sector_XXX, ...]
        where all continuous factors are cross-sectionally z-scored per
        date. Sector dummies are binary (0/1).

    Raises:
        ValueError: If universe is missing required columns.
    """
    required = {"ticker", "market_cap"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"universe missing required columns: {missing}")

    tickers = universe["ticker"].values
    log_mcap = np.log1p(universe["market_cap"].values)

    # Z-score size cross-sectionally (static snapshot — same across all dates)
    size_std = float(np.std(log_mcap))
    size_z = (log_mcap - float(np.mean(log_mcap))) / (size_std if size_std > 0 else 1.0)

    has_sector = "sector" in universe.columns
    sectors = universe["sector"].values if has_sector else None
    unique_sectors = sorted(set(sectors)) if has_sector else []

    # Build sector dummy arrays ahead of time
    sector_arrays: dict[str, np.ndarray] = {}
    for sec in unique_sectors:
        col = f"sector_{sec.replace(' ', '_')}"
        sector_arrays[col] = (sectors == sec).astype(float)

    rows = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            row: dict = {
                "ticker": ticker,
                "date": date,
                "size": round(float(size_z[i]), 6),
            }
            for col, arr in sector_arrays.items():
                row[col] = arr[i]
            rows.append(row)

    factor_df = pd.DataFrame(rows)

    # Optionally add momentum from price returns
    if price_returns is not None:
        if "ret_1m" not in price_returns.columns:
            logger.warning("price_returns provided but missing 'ret_1m' column — momentum excluded")
        else:
            mom = price_returns[["ticker", "date", "ret_1m"]].copy()
            if not pd.api.types.is_datetime64_any_dtype(mom["date"]):
                mom["date"] = pd.to_datetime(mom["date"])
            factor_df = factor_df.merge(
                mom.rename(columns={"ret_1m": "momentum"}),
                on=["ticker", "date"],
                how="left",
            )
            # Z-score momentum cross-sectionally per date
            factor_df["momentum"] = factor_df.groupby("date")["momentum"].transform(
                lambda x: (x - x.mean()) / (x.std() if x.std() > 1e-8 else 1.0)
            )

    logger.debug(
        "Computed factor proxies: %d rows, %d dates, factors=%s",
        len(factor_df),
        len(dates),
        [c for c in factor_df.columns if c not in ("ticker", "date")],
    )

    return factor_df
