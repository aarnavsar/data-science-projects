"""Generate synthetic vendor datasets and universes for testing.

Run directly: python -m src.ingestion.sample_generator
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SECTORS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Industrials",
    "Communication Services",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Materials",
    "Real Estate",
]


def generate_universe(
    n_tickers: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic trading universe (Russell 3000-like).

    Returns DataFrame with [ticker, sector, market_cap, index_membership].
    """
    rng = np.random.default_rng(seed)

    # Sector weights roughly matching S&P 500
    sector_weights = [0.28, 0.13, 0.13, 0.10, 0.09, 0.09, 0.06, 0.04, 0.03, 0.03, 0.02]
    sector_assignments = rng.choice(SECTORS, size=n_tickers, p=sector_weights)

    # Log-normal market caps (billions)
    market_caps = rng.lognormal(mean=2.0, sigma=1.5, size=n_tickers)
    market_caps = np.clip(market_caps, 0.1, 3000)  # $100M to $3T

    tickers = [f"TICK_{i:04d}" for i in range(n_tickers)]

    return pd.DataFrame(
        {
            "ticker": tickers,
            "sector": sector_assignments,
            "market_cap": market_caps.round(2),
        }
    )


def generate_vendor_dataset(
    universe: pd.DataFrame,
    coverage_rate: float = 0.6,
    n_months: int = 36,
    sector_bias: str | None = None,
    cap_bias: str | None = None,
    coverage_trend: float = 0.0,
    missing_rate: float = 0.05,
    backfill_date: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic vendor signal dataset.

    Args:
        universe: The target universe DataFrame
        coverage_rate: Base fraction of universe covered
        n_months: Months of history to generate
        sector_bias: If set, over-weight this sector (e.g., "Technology")
        cap_bias: "large" or "small" to skew coverage
        coverage_trend: Monthly change in coverage rate (negative = declining)
        missing_rate: Fraction of observations randomly missing
        backfill_date: If set (YYYY-MM), data quality improves suspiciously after this date
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    all_tickers = universe["ticker"].values
    n_universe = len(all_tickers)

    # Apply coverage biases to selection probabilities
    probs = np.ones(n_universe) / n_universe

    if sector_bias:
        sector_mask = (universe["sector"] == sector_bias).values
        probs[sector_mask] *= 8.0
        probs /= probs.sum()

    if cap_bias == "large":
        cap_weights = np.log1p(universe["market_cap"].values)
        probs *= cap_weights
        probs /= probs.sum()
    elif cap_bias == "small":
        cap_weights = 1.0 / np.log1p(universe["market_cap"].values + 1)
        probs *= cap_weights
        probs /= probs.sum()

    # Select a FIXED base set of tickers for this vendor
    # This is realistic: vendors cover specific companies, not random ones each month
    base_n_covered = int(coverage_rate * n_universe)
    base_n_covered = max(1, min(base_n_covered, n_universe))
    base_tickers = rng.choice(all_tickers, size=base_n_covered, replace=False, p=probs)

    # Generate monthly data
    start_date = datetime(2023, 1, 1)
    rows = []

    for month_idx in range(n_months):
        month_date = start_date + timedelta(days=30 * month_idx)

        # Adjust coverage over time by dropping/adding from the base set
        current_coverage = coverage_rate + coverage_trend * month_idx
        current_coverage = np.clip(current_coverage, 0.05, 0.95)

        n_this_month = int(current_coverage * n_universe)
        n_this_month = max(1, min(n_this_month, len(base_tickers)))

        # Sample from the base set (not the full universe)
        covered_tickers = rng.choice(base_tickers, size=n_this_month, replace=False)

        for ticker in covered_tickers:
            # Random signal value (could be anything — sentiment score, web traffic, etc.)
            signal = rng.normal(0, 1)

            # Simulate backfill: pre-backfill data is noisier and sparser
            if backfill_date and month_date < pd.Timestamp(backfill_date):
                if rng.random() < 0.3:  # 30% extra missing before backfill
                    continue
                signal += rng.normal(0, 0.5)  # extra noise

            # Random missingness
            if rng.random() < missing_rate:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "date": month_date,
                    "signal_value": round(signal, 4),
                }
            )

    return pd.DataFrame(rows)


def generate_signal_with_decay(
    n_tickers: int = 200,
    n_dates: int = 60,
    target_ic: float = 0.06,
    half_life_days: float = 15.0,
    lags_days: list[int] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic signal and correlated forward returns with known IC decay.

    Produces a signal and forward return dataset where the cross-sectional IC at
    lag L follows an exponential decay: IC(L) ≈ target_ic * exp(-L * ln(2) / half_life_days).

    Args:
        n_tickers: Number of tickers in the synthetic universe.
        n_dates: Number of distinct dates (treated as calendar days apart).
        target_ic: Desired IC at lag 1 (before decay).
        half_life_days: Lag at which IC decays to 50% of target_ic.
        lags_days: Forward return horizons to generate. Defaults to [1, 5, 10, 21, 63].
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (signal_df, returns_df):
        - signal_df: DataFrame with [ticker, date, signal_value]
        - returns_df: DataFrame with [ticker, date, fwd_1d, fwd_5d, ...]
    """
    if lags_days is None:
        lags_days = [1, 5, 10, 21, 63]

    rng = np.random.default_rng(seed)
    tickers = [f"TICK_{i:04d}" for i in range(n_tickers)]
    start_date = datetime(2022, 1, 3)
    dates = [start_date + timedelta(days=i) for i in range(n_dates)]

    signal_rows = []
    return_rows = []

    for date in dates:
        # Cross-sectional signal scores
        signal = rng.standard_normal(n_tickers)

        for ticker, s in zip(tickers, signal):
            signal_rows.append({"ticker": ticker, "date": date, "signal_value": round(s, 6)})

        # Forward returns for each lag: correlated with signal according to IC decay
        fwd_by_ticker: dict[str, dict[str, float]] = {t: {} for t in tickers}

        for lag in lags_days:
            lambda_ = np.log(2) / half_life_days
            expected_ic = target_ic * np.exp(-lambda_ * lag)

            # Generate cross-sectional returns correlated with signal at this lag
            # If signal ~ N(0,1) and return = beta*signal + noise, then IC ≈ beta / sqrt(1+beta^2)
            # Solve for beta given expected_ic (note: Pearson IC here; Spearman will be close)
            beta = expected_ic / max(np.sqrt(1 - expected_ic**2), 1e-6)
            noise_scale = np.sqrt(max(1.0 - beta**2, 0.01))
            noise = rng.standard_normal(n_tickers) * noise_scale
            fwd_return = beta * signal + noise

            for ticker, ret in zip(tickers, fwd_return):
                fwd_by_ticker[ticker][f"fwd_{lag}d"] = round(float(ret), 6)

        for ticker in tickers:
            row = {"ticker": ticker, "date": date}
            row.update(fwd_by_ticker[ticker])
            return_rows.append(row)

    signal_df = pd.DataFrame(signal_rows)
    returns_df = pd.DataFrame(return_rows)
    return signal_df, returns_df


def generate_signal_with_factor_loading(
    n_tickers: int = 200,
    n_dates: int = 60,
    factor_names: list[str] | None = None,
    factor_loadings: dict[str, float] | None = None,
    residual_ic: float = 0.05,
    factor_return_ic: float = 0.04,
    target_horizon_days: int = 21,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic signal, forward returns, and factor exposures.

    Constructs a signal with a known mixture of factor-driven and orthogonal
    components, enabling controlled tests of the orthogonality module.

    The signal is constructed as:
        signal = sum(factor_loadings[f] * factor_f) + residual_weight * alpha
    where alpha is cross-sectionally independent of all factors.

    Forward returns (at target_horizon_days) are correlated with both the
    factor components (via factor_return_ic) and the orthogonal alpha (via
    residual_ic), reflecting a realistic market where factors and
    vendor-specific signals each carry some predictive power.

    Args:
        n_tickers: Number of tickers in the synthetic universe.
        n_dates: Number of distinct dates.
        factor_names: Names of the factor columns to generate. Defaults to
            ["size", "value", "momentum"].
        factor_loadings: Dict mapping factor name to its loading in the signal.
            Loadings are normalized so their absolute sum + residual weight = 1.
            Defaults to {"size": 0.3, "value": 0.2, "momentum": 0.1}.
        residual_ic: IC contribution from the orthogonal alpha component.
        factor_return_ic: IC of each factor with the forward returns (the
            factors themselves are mildly predictive, separate from the signal).
        target_horizon_days: Stored as column fwd_{target_horizon_days}d in
            returns_df.
        seed: Random seed.

    Returns:
        Tuple of (signal_df, returns_df, factor_exposures_df):
        - signal_df: [ticker, date, signal_value]
        - returns_df: [ticker, date, fwd_{target_horizon_days}d]
        - factor_exposures_df: [ticker, date, *factor_names]
    """
    if factor_names is None:
        factor_names = ["size", "value", "momentum"]
    if factor_loadings is None:
        factor_loadings = {"size": 0.3, "value": 0.2, "momentum": 0.1}

    rng = np.random.default_rng(seed)
    tickers = [f"TICK_{i:04d}" for i in range(n_tickers)]
    start_date = datetime(2022, 1, 3)
    dates = [start_date + timedelta(days=i) for i in range(n_dates)]

    # Normalize loadings: residual gets whatever is left of the "budget"
    total_loading = sum(abs(factor_loadings.get(f, 0.0)) for f in factor_names)
    residual_weight = max(0.0, 1.0 - total_loading)

    # Pre-compute return betas from IC targets (Pearson ≈ Spearman for small IC)
    factor_beta = factor_return_ic / max(np.sqrt(1.0 - factor_return_ic**2), 1e-6)
    alpha_beta = residual_ic / max(np.sqrt(1.0 - residual_ic**2), 1e-6)

    fwd_col = f"fwd_{target_horizon_days}d"

    signal_rows = []
    return_rows = []
    factor_rows = []

    for date in dates:
        # Cross-sectional factor draws (independent per factor, per date)
        factor_vals: dict[str, np.ndarray] = {
            f: rng.standard_normal(n_tickers) for f in factor_names
        }

        # Orthogonal alpha component (independent of all factors)
        alpha = rng.standard_normal(n_tickers)

        # Signal = weighted sum of factors + residual * alpha
        signal = residual_weight * alpha
        for f in factor_names:
            signal += factor_loadings.get(f, 0.0) * factor_vals[f]

        # Forward returns: factors + alpha + noise
        fwd_return = alpha_beta * alpha
        for f in factor_names:
            fwd_return += factor_beta * factor_vals[f]
        noise_scale = np.sqrt(max(1.0 - alpha_beta**2 - len(factor_names) * factor_beta**2, 0.05))
        fwd_return += rng.standard_normal(n_tickers) * noise_scale

        for i, ticker in enumerate(tickers):
            signal_rows.append(
                {"ticker": ticker, "date": date, "signal_value": round(float(signal[i]), 6)}
            )
            return_rows.append(
                {"ticker": ticker, "date": date, fwd_col: round(float(fwd_return[i]), 6)}
            )
            row: dict = {"ticker": ticker, "date": date}
            for f in factor_names:
                row[f] = round(float(factor_vals[f][i]), 6)
            factor_rows.append(row)

    return (
        pd.DataFrame(signal_rows),
        pd.DataFrame(return_rows),
        pd.DataFrame(factor_rows),
    )


if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Generate universe
    universe = generate_universe(n_tickers=500)
    universe_path = os.path.join(base, "data", "reference", "universe.csv")
    universe.to_csv(universe_path, index=False)
    print(f"Universe saved: {universe_path} ({len(universe)} tickers)")

    # Generate a "good" vendor dataset
    good_data = generate_vendor_dataset(universe, coverage_rate=0.7, n_months=36)
    good_path = os.path.join(base, "data", "sample", "vendor_good.csv")
    good_data.to_csv(good_path, index=False)
    print(f"Good vendor saved: {good_path} ({len(good_data)} rows)")

    # Generate a "bad" vendor dataset (biased, declining, backfilled)
    bad_data = generate_vendor_dataset(
        universe,
        coverage_rate=0.4,
        n_months=36,
        sector_bias="Technology",
        cap_bias="large",
        coverage_trend=-0.005,
        backfill_date="2024-06",
    )
    bad_path = os.path.join(base, "data", "sample", "vendor_bad.csv")
    bad_data.to_csv(bad_path, index=False)
    print(f"Bad vendor saved: {bad_path} ({len(bad_data)} rows)")
