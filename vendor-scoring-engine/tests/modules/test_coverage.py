"""Tests for the coverage evaluation module.

Tests use synthetic data to verify scoring behavior across
different coverage scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.schema import VendorDataset, VendorMetadata, UniverseDefinition
from src.ingestion.sample_generator import generate_universe, generate_vendor_dataset
from src.modules.coverage import evaluate_coverage, CoverageConfig


@pytest.fixture
def universe():
    """Standard test universe with 200 tickers."""
    return UniverseDefinition(
        data=generate_universe(n_tickers=200, seed=42),
        name="test_universe",
    )


def _make_dataset(universe_df, **kwargs):
    """Helper to create a VendorDataset from generator kwargs."""
    data = generate_vendor_dataset(universe_df, **kwargs)
    metadata = VendorMetadata(
        vendor_name="TestVendor",
        dataset_name="test_signal",
    )
    return VendorDataset(data=data, metadata=metadata)


class TestHighCoverageVendor:
    """A vendor with good coverage should score well."""

    def test_high_coverage_scores_above_70(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.8, n_months=24, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert result.score >= 70, f"High coverage vendor scored {result.score}"

    def test_high_coverage_no_warnings(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.8, n_months=24, seed=42)
        result = evaluate_coverage(dataset, universe)
        # May have minor warnings but shouldn't have critical ones
        assert result.confidence >= 0.7


class TestLowCoverageVendor:
    """A vendor with poor coverage should score poorly."""

    def test_low_coverage_scores_below_50(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.2, n_months=24, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert result.score < 50, f"Low coverage vendor scored {result.score}"


class TestBiasedVendor:
    """A vendor with sector or cap bias should be penalized."""

    def test_sector_bias_generates_warning(self, universe):
        dataset = _make_dataset(
            universe.data, coverage_rate=0.6, sector_bias="Technology",
            n_months=24, seed=42
        )
        result = evaluate_coverage(dataset, universe)
        bias_warnings = [w for w in result.warnings if "Sector bias" in w]
        assert len(bias_warnings) > 0, "Expected sector bias warning"

    def test_cap_bias_generates_warning(self, universe):
        dataset = _make_dataset(
            universe.data, coverage_rate=0.6, cap_bias="large",
            n_months=24, seed=42
        )
        result = evaluate_coverage(dataset, universe)
        bias_warnings = [w for w in result.warnings if "cap" in w.lower()]
        assert len(bias_warnings) > 0, "Expected market cap bias warning"


class TestDecliningCoverage:
    """A vendor with declining coverage should be flagged."""

    def test_declining_coverage_warning(self, universe):
        dataset = _make_dataset(
            universe.data, coverage_rate=0.7, coverage_trend=-0.01,
            n_months=36, seed=42
        )
        result = evaluate_coverage(dataset, universe)
        trend_warnings = [w for w in result.warnings if "declining" in w.lower()]
        assert len(trend_warnings) > 0, "Expected declining coverage warning"

    def test_declining_scores_lower_than_stable(self, universe):
        stable = _make_dataset(
            universe.data, coverage_rate=0.6, coverage_trend=0.0,
            n_months=36, seed=42
        )
        declining = _make_dataset(
            universe.data, coverage_rate=0.6, coverage_trend=-0.01,
            n_months=36, seed=42
        )
        stable_result = evaluate_coverage(stable, universe)
        declining_result = evaluate_coverage(declining, universe)
        assert declining_result.score < stable_result.score


class TestNarrativeGeneration:
    """Narrative should be informative and non-empty."""

    def test_narrative_contains_coverage_rate(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.5, n_months=12, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert "%" in result.narrative or "coverage" in result.narrative.lower()

    def test_narrative_is_substantial(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.5, n_months=12, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert len(result.narrative) > 50, "Narrative should be at least a few sentences"


class TestConfidence:
    """Confidence should reflect data sufficiency."""

    def test_short_history_low_confidence(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.6, n_months=2, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert result.confidence < 0.6, "Short history should have low confidence"

    def test_long_history_higher_confidence(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.6, n_months=36, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert result.confidence >= 0.7, "Long history should have decent confidence"


class TestModuleResultContract:
    """Verify the module returns a valid ModuleResult."""

    def test_score_in_range(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.5, n_months=12, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert 0 <= result.score <= 100

    def test_confidence_in_range(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.5, n_months=12, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert 0 <= result.confidence <= 1

    def test_module_name_is_coverage(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.5, n_months=12, seed=42)
        result = evaluate_coverage(dataset, universe)
        assert result.module_name == "coverage"

    def test_diagnostics_has_required_keys(self, universe):
        dataset = _make_dataset(universe.data, coverage_rate=0.5, n_months=12, seed=42)
        result = evaluate_coverage(dataset, universe)
        required_keys = {
            "overall_coverage", "sector_coverage", "cap_quintile_coverage",
            "coverage_timeseries", "coverage_trend",
        }
        assert required_keys.issubset(set(result.diagnostics.keys()))
