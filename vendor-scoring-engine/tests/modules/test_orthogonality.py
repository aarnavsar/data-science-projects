"""Tests for the orthogonality evaluation module.

All tests use synthetic data with known factor structure so that
expected qualitative behaviors can be asserted. The key invariants:
- Pure factor signal → high IC degradation → low score
- Orthogonal signal → low IC degradation → higher score
- Partial contamination → intermediate behavior
- Edge cases (missing data, insufficient dates) → graceful failure
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.sample_generator import generate_signal_with_factor_loading
from src.ingestion.schema import VendorDataset, VendorMetadata
from src.modules.orthogonality import (
    OrthogonalityConfig,
    _compute_ic_degradation,
    _compute_vif,
    _find_fwd_col,
    _neutralize_signal,
    evaluate_orthogonality,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FACTOR_NAMES = ["size", "value", "momentum"]


def _make_dataset(signal_df: pd.DataFrame, vendor_name: str = "TestVendor") -> VendorDataset:
    """Wrap a signal DataFrame into a VendorDataset."""
    return VendorDataset(
        data=signal_df,
        metadata=VendorMetadata(vendor_name=vendor_name, dataset_name="test_signal"),
    )


@pytest.fixture
def orthogonal_signal():
    """Signal with no factor loading — all IC comes from orthogonal alpha."""
    signal_df, returns_df, factor_df = generate_signal_with_factor_loading(
        n_tickers=200,
        n_dates=60,
        factor_names=_FACTOR_NAMES,
        factor_loadings={"size": 0.0, "value": 0.0, "momentum": 0.0},
        residual_ic=0.07,
        factor_return_ic=0.03,
        seed=42,
    )
    return _make_dataset(signal_df), returns_df, factor_df


@pytest.fixture
def factor_contaminated_signal():
    """Signal dominated by factor loadings — most IC degrades on neutralization."""
    signal_df, returns_df, factor_df = generate_signal_with_factor_loading(
        n_tickers=200,
        n_dates=60,
        factor_names=_FACTOR_NAMES,
        factor_loadings={"size": 0.45, "value": 0.35, "momentum": 0.10},
        residual_ic=0.01,
        factor_return_ic=0.06,
        seed=42,
    )
    return _make_dataset(signal_df), returns_df, factor_df


@pytest.fixture
def mixed_signal():
    """Signal with partial factor contamination."""
    signal_df, returns_df, factor_df = generate_signal_with_factor_loading(
        n_tickers=200,
        n_dates=60,
        factor_names=_FACTOR_NAMES,
        factor_loadings={"size": 0.25, "value": 0.15, "momentum": 0.10},
        residual_ic=0.05,
        factor_return_ic=0.04,
        seed=42,
    )
    return _make_dataset(signal_df), returns_df, factor_df


@pytest.fixture
def short_history_signal():
    """Only 8 dates — well below min_dates threshold."""
    signal_df, returns_df, factor_df = generate_signal_with_factor_loading(
        n_tickers=200,
        n_dates=8,
        factor_names=_FACTOR_NAMES,
        factor_loadings={"size": 0.0, "value": 0.0, "momentum": 0.0},
        residual_ic=0.07,
        seed=42,
    )
    return _make_dataset(signal_df), returns_df, factor_df


# ---------------------------------------------------------------------------
# ModuleResult contract
# ---------------------------------------------------------------------------


class TestModuleResultContract:
    """Verify that the module always returns a valid ModuleResult."""

    def test_returns_module_result_fields(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        assert result.module_name == "orthogonality"
        assert 0 <= result.score <= 100
        assert 0 <= result.confidence <= 1
        assert isinstance(result.narrative, str) and len(result.narrative) > 20
        assert isinstance(result.diagnostics, dict)
        assert isinstance(result.warnings, list)

    def test_required_diagnostics_keys(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        required_keys = {
            "raw_ic",
            "residual_ic",
            "ic_degradation_pct",
            "mean_r_squared",
            "factor_loadings",
            "dominant_factor",
            "residual_ic_t_stat",
            "residual_ic_p_value",
            "vif_scores",
            "n_dates_evaluated",
            "n_tickers_evaluated",
            "factors_used",
        }
        missing = required_keys - set(result.diagnostics.keys())
        assert not missing, f"Missing diagnostic keys: {missing}"

    def test_score_is_float(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        assert isinstance(result.score, float)

    def test_default_config_used_when_none(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df, config=None)
        assert result.score >= 0


# ---------------------------------------------------------------------------
# Signal quality scoring invariants
# ---------------------------------------------------------------------------


class TestScoringInvariants:
    """Orthogonal signals should score better than factor-contaminated ones."""

    def test_orthogonal_signal_scores_higher_than_contaminated(
        self, orthogonal_signal, factor_contaminated_signal
    ):
        ds_orth, ret_orth, fac_orth = orthogonal_signal
        ds_cont, ret_cont, fac_cont = factor_contaminated_signal

        result_orth = evaluate_orthogonality(ds_orth, fac_orth, ret_orth)
        result_cont = evaluate_orthogonality(ds_cont, fac_cont, ret_cont)

        assert result_orth.score > result_cont.score, (
            f"Expected orthogonal ({result_orth.score}) > contaminated ({result_cont.score})"
        )

    def test_orthogonal_signal_low_ic_degradation(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        deg = result.diagnostics["ic_degradation_pct"]
        # IC degradation should be low for an orthogonal signal
        if deg is not None:
            assert deg < 50, f"Expected low IC degradation, got {deg:.1f}%"

    def test_contaminated_signal_high_ic_degradation(self, factor_contaminated_signal):
        dataset, returns_df, factor_df = factor_contaminated_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        deg = result.diagnostics["ic_degradation_pct"]
        if deg is not None:
            assert deg > 30, f"Expected high IC degradation for contaminated signal, got {deg:.1f}%"

    def test_orthogonal_signal_low_r_squared(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        r2 = result.diagnostics["mean_r_squared"]
        assert r2 < 0.30, f"Expected low factor R² for orthogonal signal, got {r2:.3f}"

    def test_contaminated_signal_higher_r_squared(self, factor_contaminated_signal):
        dataset, returns_df, factor_df = factor_contaminated_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        r2 = result.diagnostics["mean_r_squared"]
        assert r2 > 0.10, f"Expected higher R² for contaminated signal, got {r2:.3f}"


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    """Specific conditions should trigger appropriate warnings."""

    def test_contaminated_signal_warns_ic_degradation(self, factor_contaminated_signal):
        dataset, returns_df, factor_df = factor_contaminated_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        assert any("degradation" in w.lower() or "factor" in w.lower() for w in result.warnings), (
            "Expected a degradation or factor contamination warning"
        )

    def test_short_history_warns_insufficient_dates(self, short_history_signal):
        dataset, returns_df, factor_df = short_history_signal
        config = OrthogonalityConfig(factors=_FACTOR_NAMES, min_dates=24)
        result = evaluate_orthogonality(dataset, factor_df, returns_df, config=config)
        assert any("dates" in w.lower() for w in result.warnings), (
            "Expected a warning about insufficient dates"
        )

    def test_missing_factors_warns(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        # Configure extra factors that don't exist in factor_df
        config = OrthogonalityConfig(factors=["size", "value", "momentum", "quality", "mkt_beta"])
        result = evaluate_orthogonality(dataset, factor_df, returns_df, config=config)
        assert any("unavailable" in w.lower() or "factor" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Graceful handling of degenerate inputs."""

    def test_no_fwd_col_returns_zero_score(self, orthogonal_signal):
        dataset, _, factor_df = orthogonal_signal
        bad_returns = pd.DataFrame({"ticker": ["A"], "date": pd.to_datetime(["2022-01-01"])})
        result = evaluate_orthogonality(dataset, factor_df, bad_returns)
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_no_factor_cols_returns_zero_score(self, orthogonal_signal):
        dataset, returns_df, _ = orthogonal_signal
        bad_factors = pd.DataFrame(
            {"ticker": ["A"], "date": pd.to_datetime(["2022-01-01"]), "irrelevant": [1.0]}
        )
        result = evaluate_orthogonality(dataset, bad_factors, returns_df)
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_no_overlap_returns_gracefully(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        # Shift factor_df dates far into the future so there's no overlap
        factor_df_shifted = factor_df.copy()
        factor_df_shifted["date"] = factor_df_shifted["date"] + pd.DateOffset(years=10)
        result = evaluate_orthogonality(dataset, factor_df_shifted, returns_df)
        assert result.score == 0.0
        assert result.confidence <= 0.2

    def test_short_history_low_confidence(self, short_history_signal):
        dataset, returns_df, factor_df = short_history_signal
        result = evaluate_orthogonality(dataset, factor_df, returns_df)
        assert result.confidence < 0.75, (
            f"Expected low confidence for short history, got {result.confidence}"
        )


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    """Confidence should reflect data quality."""

    def test_full_history_higher_confidence_than_short(
        self, orthogonal_signal, short_history_signal
    ):
        ds_full, ret_full, fac_full = orthogonal_signal
        ds_short, ret_short, fac_short = short_history_signal

        result_full = evaluate_orthogonality(ds_full, fac_full, ret_full)
        result_short = evaluate_orthogonality(ds_short, fac_short, ret_short)

        assert result_full.confidence > result_short.confidence, (
            f"Full history confidence ({result_full.confidence}) should exceed "
            f"short history ({result_short.confidence})"
        )

    def test_missing_factors_reduces_confidence(self, orthogonal_signal):
        dataset, returns_df, factor_df = orthogonal_signal
        full_config = OrthogonalityConfig(factors=["size", "value", "momentum"])
        sparse_config = OrthogonalityConfig(
            factors=["size", "value", "momentum", "quality", "investment", "mkt_beta"]
        )

        result_full = evaluate_orthogonality(dataset, factor_df, returns_df, config=full_config)
        result_sparse = evaluate_orthogonality(dataset, factor_df, returns_df, config=sparse_config)

        assert result_full.confidence >= result_sparse.confidence


# ---------------------------------------------------------------------------
# Unit tests for private helpers
# ---------------------------------------------------------------------------


class TestFindFwdCol:
    def test_exact_match(self):
        df = pd.DataFrame({"ticker": [], "date": [], "fwd_21d": []})
        assert _find_fwd_col(df, 21) == "fwd_21d"

    def test_closest_match(self):
        df = pd.DataFrame({"ticker": [], "date": [], "fwd_5d": [], "fwd_63d": []})
        assert _find_fwd_col(df, 21) == "fwd_5d"

    def test_no_fwd_cols(self):
        df = pd.DataFrame({"ticker": [], "date": [], "other": []})
        assert _find_fwd_col(df, 21) is None


class TestICDegradation:
    def test_zero_raw_ic_returns_none(self):
        result = _compute_ic_degradation(0.0, 0.05)
        assert result["interpretable"] is False
        assert result["degradation_pct"] is None

    def test_no_degradation(self):
        result = _compute_ic_degradation(0.05, 0.05)
        assert result["interpretable"] is True
        assert abs(result["degradation_pct"]) < 1.0

    def test_full_degradation(self):
        result = _compute_ic_degradation(0.05, 0.0)
        assert result["degradation_pct"] == pytest.approx(100.0, abs=1.0)

    def test_negative_raw_ic_handled(self):
        """A negatively predictive signal: raw IC negative, residual smaller in abs value."""
        result = _compute_ic_degradation(-0.05, -0.03)
        assert result["interpretable"] is True
        assert result["degradation_pct"] == pytest.approx(40.0, abs=1.0)

    def test_improvement_gives_negative_degradation(self):
        """Residual IC larger than raw IC in absolute terms → negative degradation."""
        result = _compute_ic_degradation(0.03, 0.06)
        assert result["degradation_pct"] < 0.0


class TestComputeVIF:
    def test_orthogonal_factors_low_vif(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "size": rng.standard_normal(n),
                "value": rng.standard_normal(n),
                "momentum": rng.standard_normal(n),
            }
        )
        vif = _compute_vif(df, ["size", "value", "momentum"])
        assert all(v < 3.0 for v in vif.values()), f"Expected low VIF, got {vif}"

    def test_correlated_factors_high_vif(self):
        rng = np.random.default_rng(42)
        n = 200
        base = rng.standard_normal(n)
        df = pd.DataFrame(
            {
                "size": base + 0.1 * rng.standard_normal(n),
                "size2": base + 0.1 * rng.standard_normal(n),  # near-duplicate
                "momentum": rng.standard_normal(n),
            }
        )
        vif = _compute_vif(df, ["size", "size2", "momentum"])
        assert vif["size"] > 3.0 or vif["size2"] > 3.0, (
            f"Expected high VIF for correlated factors, got {vif}"
        )

    def test_single_factor_returns_one(self):
        df = pd.DataFrame({"size": [1.0, 2.0, 3.0]})
        vif = _compute_vif(df, ["size"])
        assert vif["size"] == pytest.approx(1.0)


class TestNeutralizeSignal:
    def test_residual_signal_column_added(self):
        rng = np.random.default_rng(0)
        n = 100
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        rows = []
        for d in dates:
            for i in range(n):
                rows.append(
                    {
                        "ticker": f"T{i}",
                        "date": d,
                        "signal_value": rng.standard_normal(),
                        "size": rng.standard_normal(),
                        "fwd_21d": rng.standard_normal(),
                    }
                )
        df = pd.DataFrame(rows)
        result, meta = _neutralize_signal(df, ["size"], use_ridge=False, min_obs=30)
        assert "residual_signal" in result.columns
        assert meta["n_dates_regressed"] == 5

    def test_residual_has_lower_factor_correlation(self):
        """After neutralization, residual should be less correlated with the factor."""
        rng = np.random.default_rng(7)
        n = 150
        dates = pd.date_range("2022-01-01", periods=30, freq="D")
        rows = []
        for d in dates:
            size = rng.standard_normal(n)
            signal = 0.8 * size + 0.2 * rng.standard_normal(n)  # heavily factor-loaded
            for i in range(n):
                rows.append(
                    {
                        "ticker": f"T{i}",
                        "date": d,
                        "signal_value": float(signal[i]),
                        "size": float(size[i]),
                        "fwd_21d": rng.standard_normal(),
                    }
                )
        df = pd.DataFrame(rows)
        result, _ = _neutralize_signal(df, ["size"], use_ridge=False, min_obs=30)

        valid = result.dropna(subset=["residual_signal"])
        raw_corr = abs(valid["signal_value"].corr(valid["size"]))
        resid_corr = abs(valid["residual_signal"].corr(valid["size"]))

        assert resid_corr < raw_corr, (
            f"Residual ({resid_corr:.3f}) should be less correlated with factor "
            f"than raw signal ({raw_corr:.3f})"
        )
