"""Tests for the signal decay evaluation module.

Tests use synthetic data with known IC properties to verify that
scoring behaves correctly across different signal quality scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.sample_generator import generate_signal_with_decay
from src.ingestion.schema import VendorDataset, VendorMetadata
from src.modules.signal_decay import (
    SignalDecayConfig,
    _compute_ic_by_lag,
    _compute_rolling_ic,
    _compute_signal_autocorrelation,
    _fit_ic_halflife,
    _merge_signal_returns,
    evaluate_signal_decay,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dataset(signal_df: pd.DataFrame, vendor_name: str = "TestVendor") -> VendorDataset:
    """Wrap a signal DataFrame into a VendorDataset."""
    return VendorDataset(
        data=signal_df,
        metadata=VendorMetadata(
            vendor_name=vendor_name,
            dataset_name="test_signal",
        ),
    )


@pytest.fixture
def strong_signal():
    """Strong, stable signal: IC~0.08 at lag 1, half-life~15 days."""
    signal_df, returns_df = generate_signal_with_decay(
        n_tickers=200, n_dates=60, target_ic=0.08, half_life_days=15.0, seed=42
    )
    return _make_dataset(signal_df), returns_df


@pytest.fixture
def weak_signal():
    """Near-zero signal: IC~0.005 at all lags."""
    signal_df, returns_df = generate_signal_with_decay(
        n_tickers=200, n_dates=60, target_ic=0.005, half_life_days=15.0, seed=42
    )
    return _make_dataset(signal_df), returns_df


@pytest.fixture
def short_history_signal():
    """Only 8 dates — well below the min_dates_for_ic=24 threshold."""
    signal_df, returns_df = generate_signal_with_decay(
        n_tickers=200, n_dates=8, target_ic=0.07, half_life_days=10.0, seed=42
    )
    return _make_dataset(signal_df), returns_df


@pytest.fixture
def long_history_signal():
    """Long history: 120 dates for stable IC estimates."""
    signal_df, returns_df = generate_signal_with_decay(
        n_tickers=200, n_dates=120, target_ic=0.07, half_life_days=15.0, seed=42
    )
    return _make_dataset(signal_df), returns_df


@pytest.fixture
def negative_signal():
    """Negatively predictive signal: negate the returns so IC is negative."""
    signal_df, returns_df = generate_signal_with_decay(
        n_tickers=200, n_dates=60, target_ic=0.08, half_life_days=15.0, seed=42
    )
    # Invert all forward returns to flip the IC sign
    fwd_cols = [c for c in returns_df.columns if c.startswith("fwd_")]
    returns_df[fwd_cols] = -returns_df[fwd_cols]
    return _make_dataset(signal_df), returns_df


# ---------------------------------------------------------------------------
# Strong signal
# ---------------------------------------------------------------------------


class TestStrongSignal:
    """A strong, persistent signal should score well."""

    def test_strong_signal_scores_above_60(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert result.score >= 60, f"Strong signal scored only {result.score}"

    def test_strong_signal_module_name(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert result.module_name == "signal_decay"

    def test_strong_signal_positive_ic_at_lag1(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        ic_lag1 = result.diagnostics["ic_by_lag"][1]["mean_ic"]
        assert ic_lag1 > 0.02, f"Expected IC > 0.02 at lag 1, got {ic_lag1}"

    def test_strong_signal_icir_positive(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        icir = result.diagnostics["icir_by_lag"][1]
        assert icir > 0, f"Expected positive ICIR, got {icir}"

    def test_strong_signal_halflife_estimated(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        hl = result.diagnostics["ic_halflife_days"]
        assert hl is not None, "Expected half-life to be estimated for strong signal"
        assert 1 < hl < 200, f"Half-life {hl} outside plausible range"

    def test_strong_signal_p_value_at_lag1(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        p_val = result.diagnostics["ic_by_lag"][1]["p_value"]
        assert p_val < 0.10, f"Expected significant p-value at lag 1, got {p_val}"


# ---------------------------------------------------------------------------
# Weak signal
# ---------------------------------------------------------------------------


class TestWeakSignal:
    """A near-zero signal should score poorly."""

    def test_weak_signal_scores_below_40(self, weak_signal):
        dataset, returns_df = weak_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert result.score < 40, f"Weak signal scored {result.score} — expected < 40"

    def test_weak_signal_scores_below_strong(self, strong_signal, weak_signal):
        strong_ds, strong_ret = strong_signal
        weak_ds, weak_ret = weak_signal
        strong_result = evaluate_signal_decay(strong_ds, strong_ret)
        weak_result = evaluate_signal_decay(weak_ds, weak_ret)
        assert weak_result.score < strong_result.score


# ---------------------------------------------------------------------------
# Negative signal
# ---------------------------------------------------------------------------


class TestNegativeSignal:
    """A negatively predictive signal should score ≤ 20 and emit a warning."""

    def test_negative_ic_caps_score_at_20(self, negative_signal):
        dataset, returns_df = negative_signal
        config = SignalDecayConfig(target_horizon_days=1)  # use lag 1 as target
        result = evaluate_signal_decay(dataset, returns_df, config=config)
        assert result.score <= 20, f"Negative IC signal scored {result.score}"

    def test_negative_ic_generates_warning(self, negative_signal):
        dataset, returns_df = negative_signal
        config = SignalDecayConfig(target_horizon_days=1)
        result = evaluate_signal_decay(dataset, returns_df, config=config)
        neg_warnings = [w for w in result.warnings if "negatively predictive" in w.lower()]
        assert len(neg_warnings) > 0, "Expected 'negatively predictive' warning"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    """Confidence should reflect data sufficiency."""

    def test_short_history_low_confidence(self, short_history_signal):
        # With 8 dates, time_conf is penalized (~0.47) even if cross-section is large.
        # Full confidence (1.0) is only achievable with ≥ min_dates_for_ic=24 dates.
        dataset, returns_df = short_history_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert result.confidence < 0.85, (
            f"Short history should have confidence < 0.85, got {result.confidence}"
        )

    def test_long_history_higher_confidence(self, long_history_signal):
        dataset, returns_df = long_history_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert result.confidence >= 0.6, (
            f"Long history should have confidence >= 0.6, got {result.confidence}"
        )

    def test_short_history_lower_confidence_than_long(
        self, short_history_signal, long_history_signal
    ):
        short_ds, short_ret = short_history_signal
        long_ds, long_ret = long_history_signal
        short_result = evaluate_signal_decay(short_ds, short_ret)
        long_result = evaluate_signal_decay(long_ds, long_ret)
        assert short_result.confidence < long_result.confidence

    def test_missing_lags_reduces_confidence(self, strong_signal):
        """Providing only one of the configured lags should reduce lag_conf."""
        dataset, returns_df = strong_signal
        # Only pass fwd_1d — drop all other lag columns
        returns_subset = returns_df[["ticker", "date", "fwd_1d"]]
        result_full = evaluate_signal_decay(dataset, returns_df)
        result_partial = evaluate_signal_decay(dataset, returns_subset)
        assert result_partial.confidence <= result_full.confidence


# ---------------------------------------------------------------------------
# ModuleResult contract
# ---------------------------------------------------------------------------


class TestModuleResultContract:
    """The module must return a valid, well-formed ModuleResult."""

    def test_score_in_range(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert 0 <= result.score <= 100

    def test_confidence_in_range(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert 0 <= result.confidence <= 1

    def test_module_name(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert result.module_name == "signal_decay"

    def test_diagnostics_required_keys(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        required_keys = {
            "ic_by_lag",
            "icir_by_lag",
            "ic_halflife_days",
            "ic_halflife_r_squared",
            "rolling_ic",
            "signal_autocorrelation",
            "target_lag_ic",
            "target_lag_icir",
            "target_lag_p_value",
            "n_dates_evaluated",
            "lags_evaluated",
        }
        missing = required_keys - set(result.diagnostics.keys())
        assert not missing, f"Missing diagnostics keys: {missing}"

    def test_narrative_is_substantial(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert len(result.narrative) > 80, "Narrative should be at least a few sentences"

    def test_narrative_contains_ic(self, strong_signal):
        dataset, returns_df = strong_signal
        result = evaluate_signal_decay(dataset, returns_df)
        assert "IC=" in result.narrative or "IC" in result.narrative


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Graceful handling of pathological inputs."""

    def test_missing_fwd_columns_raises(self, strong_signal):
        """returns_data with no fwd_ columns at all should raise ValueError."""
        dataset, returns_df = strong_signal
        bad_returns = returns_df[["ticker", "date"]]  # strip all fwd_ columns
        with pytest.raises(ValueError, match="none of the expected forward return"):
            evaluate_signal_decay(dataset, bad_returns)

    def test_missing_ticker_column_raises(self, strong_signal):
        dataset, returns_df = strong_signal
        bad_returns = returns_df.drop(columns=["ticker"])
        with pytest.raises(ValueError, match="missing required columns"):
            evaluate_signal_decay(dataset, bad_returns)

    def test_no_overlap_returns_empty_ic(self):
        """No date overlap between signal and returns → all IC n_dates=0."""
        signal_df, returns_df = generate_signal_with_decay(n_tickers=50, n_dates=20, seed=1)
        # Shift returns 1000 days forward so there's no overlap
        returns_df["date"] = returns_df["date"] + pd.Timedelta(days=1000)
        dataset = _make_dataset(signal_df)
        result = evaluate_signal_decay(dataset, returns_df)
        # All ICs should have n_dates=0
        for lag, lag_stats in result.diagnostics["ic_by_lag"].items():
            assert lag_stats["n_dates"] == 0, (
                f"Expected 0 dates for lag {lag}, got {lag_stats['n_dates']}"
            )

    def test_constant_signal_handles_zero_variance(self):
        """A constant signal has zero cross-sectional variance — should not crash."""
        signal_df, returns_df = generate_signal_with_decay(n_tickers=50, n_dates=20, seed=1)
        signal_df["signal_value"] = 1.0  # constant
        dataset = _make_dataset(signal_df)
        result = evaluate_signal_decay(dataset, returns_df)
        assert 0 <= result.score <= 100
        # All IC n_dates should be 0 (zero variance dates excluded)
        for lag, lag_stats in result.diagnostics["ic_by_lag"].items():
            assert lag_stats["n_dates"] == 0

    def test_partial_lag_coverage(self, strong_signal):
        """Only providing 2 out of 5 lags should still work."""
        dataset, returns_df = strong_signal
        returns_subset = returns_df[["ticker", "date", "fwd_1d", "fwd_21d"]]
        result = evaluate_signal_decay(dataset, returns_subset)
        assert result.score >= 0
        assert set(result.diagnostics["lags_evaluated"]) == {1, 21}

    def test_short_history_warning_emitted(self, short_history_signal):
        dataset, returns_df = short_history_signal
        result = evaluate_signal_decay(dataset, returns_df)
        history_warnings = [w for w in result.warnings if "dates available" in w]
        assert len(history_warnings) > 0, "Expected short-history warning"


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Unit tests for private helper functions."""

    def test_compute_ic_by_lag_positive_for_correlated_data(self):
        """Manually verify IC computation on known correlated data."""
        rng = np.random.default_rng(0)
        n = 100
        signal = rng.standard_normal(n)
        fwd_return = 0.3 * signal + rng.standard_normal(n) * 0.9  # Pearson r ≈ 0.3

        df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "date": pd.Timestamp("2024-01-01"),
                "signal_value": signal,
                "fwd_21d": fwd_return,
            }
        )
        ic_stats = _compute_ic_by_lag(df, [21], min_obs_per_date=10)
        assert ic_stats[21]["mean_ic"] > 0, "Expected positive IC for positively correlated data"
        assert ic_stats[21]["n_dates"] == 1

    def test_fit_halflife_recovers_known_value(self):
        """IC half-life fitter should approximately recover a known decay rate."""
        half_life_true = 20.0
        lags = [1, 5, 10, 21, 63]
        ic0 = 0.08
        ic_stats = {
            lag: {
                "mean_ic": ic0 * np.exp(-lag * np.log(2) / half_life_true),
                "std_ic": 0.01,
                "icir": 0.5,
                "t_stat": 3.0,
                "p_value": 0.01,
                "n_dates": 50,
            }
            for lag in lags
        }
        result = _fit_ic_halflife(ic_stats, lags)
        assert result["halflife_days"] is not None
        # Allow ±50% tolerance — fitting with 5 points can be noisy
        assert 10.0 <= result["halflife_days"] <= 40.0, (
            f"Half-life estimate {result['halflife_days']} far from true {half_life_true}"
        )

    def test_fit_halflife_returns_none_for_negative_ic(self):
        """Half-life cannot be fit when IC is negative at all lags."""
        lags = [1, 5, 10, 21, 63]
        ic_stats = {
            lag: {
                "mean_ic": -0.03,
                "std_ic": 0.01,
                "icir": -0.5,
                "t_stat": -2.0,
                "p_value": 0.05,
                "n_dates": 30,
            }
            for lag in lags
        }
        result = _fit_ic_halflife(ic_stats, lags)
        assert result["halflife_days"] is None

    def test_signal_autocorrelation_high_for_persistent_signal(self):
        """AR(1) should be high for a slowly-changing signal."""
        n_tickers, n_dates = 10, 50
        tickers = [f"T{i}" for i in range(n_tickers)]
        rows = []
        rng = np.random.default_rng(0)
        for ticker in tickers:
            # random walk — highly persistent
            signal = np.cumsum(rng.standard_normal(n_dates)) * 0.1
            for i, s in enumerate(signal):
                rows.append(
                    {
                        "ticker": ticker,
                        "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=i),
                        "signal_value": s,
                    }
                )
        df = pd.DataFrame(rows)
        ac = _compute_signal_autocorrelation(df)
        assert ac > 0.5, f"Expected high autocorrelation for random walk, got {ac}"

    def test_rolling_ic_returns_list_of_dicts(self, strong_signal):
        """Rolling IC should return a list of date/ic dicts."""
        dataset, returns_df = strong_signal
        merged = _merge_signal_returns(dataset.data, returns_df)
        rolling = _compute_rolling_ic(merged, "fwd_21d", window=10, min_obs_per_date=5)
        assert isinstance(rolling, list)
        if rolling:
            assert "date" in rolling[0]
            assert "ic" in rolling[0]

    def test_rolling_ic_window_larger_than_data_returns_empty(self):
        """If window > n_dates, rolling IC returns []."""
        signal_df, returns_df = generate_signal_with_decay(n_tickers=50, n_dates=5, seed=0)
        dataset = _make_dataset(signal_df)
        merged = _merge_signal_returns(dataset.data, returns_df)
        rolling = _compute_rolling_ic(merged, "fwd_21d", window=20, min_obs_per_date=5)
        assert rolling == []
