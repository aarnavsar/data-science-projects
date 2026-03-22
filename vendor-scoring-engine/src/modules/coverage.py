"""Coverage analysis module.

Evaluates how well a vendor dataset covers a target trading universe.
This is the most concrete module — it answers: "Can I actually use this
data for the stocks I trade?"

Metrics computed:
- Overall coverage rate (% of universe tickers present)
- Sector coverage distribution (is it biased toward tech?)
- Market cap coverage distribution (does it only cover mega-caps?)
- Coverage over time (is it growing or shrinking?)
- Coverage consistency (does coverage drop on specific dates/events?)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.ingestion.schema import VendorDataset, UniverseDefinition
from src.scoring.results import ModuleResult

logger = logging.getLogger(__name__)


@dataclass
class CoverageConfig:
    """Configuration for coverage evaluation."""

    min_acceptable_coverage: float = 0.3  # below this, score tanks
    ideal_coverage: float = 0.8  # above this, full marks for coverage rate
    sector_bias_threshold: float = 0.15  # max acceptable deviation from universe weights
    min_history_months: int = 12  # minimum history to evaluate trends
    coverage_drop_zscore: float = 2.0  # flag dates where coverage drops this many SDs


def evaluate_coverage(
    dataset: VendorDataset,
    universe: UniverseDefinition,
    config: CoverageConfig | None = None,
) -> ModuleResult:
    """Run the full coverage evaluation.

    Args:
        dataset: The vendor dataset to evaluate
        universe: The trading universe to measure coverage against
        config: Optional configuration overrides

    Returns:
        ModuleResult with coverage score, diagnostics, and narrative
    """
    if config is None:
        config = CoverageConfig()

    diagnostics = {}
    warnings = []

    # 1. Overall coverage rate
    overall = _compute_overall_coverage(dataset, universe)
    diagnostics["overall_coverage"] = overall

    # 2. Sector breakdown
    sector_analysis = _compute_sector_coverage(dataset, universe)
    diagnostics["sector_coverage"] = sector_analysis["coverage_by_sector"]
    diagnostics["sector_bias"] = sector_analysis["bias_scores"]

    if sector_analysis["max_bias"] > config.sector_bias_threshold:
        warnings.append(
            f"Sector bias detected: {sector_analysis['most_biased_sector']} "
            f"is {sector_analysis['max_bias']:.1%} overweight vs universe"
        )

    # 3. Market cap distribution
    cap_analysis = _compute_cap_coverage(dataset, universe)
    diagnostics["cap_quintile_coverage"] = cap_analysis["coverage_by_quintile"]
    diagnostics["cap_bias_direction"] = cap_analysis["bias_direction"]

    if cap_analysis["bias_direction"] != "balanced":
        warnings.append(
            f"Market cap bias: coverage skews toward {cap_analysis['bias_direction']}-cap names"
        )

    # 4. Coverage over time
    time_analysis = _compute_coverage_over_time(dataset, universe, config)
    diagnostics["coverage_timeseries"] = time_analysis["monthly_coverage"]
    diagnostics["coverage_trend"] = time_analysis["trend_slope"]

    if time_analysis["trend_slope"] < -0.005:
        warnings.append(
            f"Coverage declining: {time_analysis['trend_slope']:.3f} per month"
        )

    # 5. Coverage anomalies (suspicious drops)
    anomalies = _detect_coverage_anomalies(
        time_analysis["monthly_coverage"], config.coverage_drop_zscore
    )
    diagnostics["coverage_anomalies"] = anomalies
    if anomalies:
        warnings.append(
            f"{len(anomalies)} anomalous coverage drops detected"
        )

    # Compute composite score
    score = _compute_coverage_score(overall, sector_analysis, cap_analysis, time_analysis, config)
    confidence = _compute_confidence(dataset, universe, time_analysis)

    narrative = _generate_narrative(
        overall, sector_analysis, cap_analysis, time_analysis, anomalies, warnings
    )

    return ModuleResult(
        module_name="coverage",
        score=score,
        confidence=confidence,
        diagnostics=diagnostics,
        narrative=narrative,
        warnings=warnings,
    )


def _compute_overall_coverage(
    dataset: VendorDataset, universe: UniverseDefinition
) -> dict:
    """What fraction of the universe does this dataset cover?"""
    universe_tickers = set(universe.tickers)
    vendor_tickers = set(dataset.tickers)

    covered = universe_tickers & vendor_tickers
    extra = vendor_tickers - universe_tickers  # tickers not in universe

    return {
        "universe_size": len(universe_tickers),
        "vendor_tickers": len(vendor_tickers),
        "covered_tickers": len(covered),
        "coverage_rate": len(covered) / len(universe_tickers) if universe_tickers else 0,
        "extra_tickers": len(extra),
    }


def _compute_sector_coverage(
    dataset: VendorDataset, universe: UniverseDefinition
) -> dict:
    """Coverage broken down by sector, with bias detection."""
    universe_sectors = universe.data.groupby("sector")["ticker"].apply(set).to_dict()
    vendor_tickers = set(dataset.tickers)

    coverage_by_sector = {}
    universe_weights = {}
    vendor_weights = {}

    total_universe = sum(len(v) for v in universe_sectors.values())

    for sector, tickers in universe_sectors.items():
        covered = tickers & vendor_tickers
        coverage_by_sector[sector] = {
            "universe_count": len(tickers),
            "covered_count": len(covered),
            "coverage_rate": len(covered) / len(tickers) if tickers else 0,
        }
        universe_weights[sector] = len(tickers) / total_universe if total_universe else 0
        vendor_weights[sector] = (
            len(covered) / sum(
                len(universe_sectors[s] & vendor_tickers)
                for s in universe_sectors
            )
            if sum(len(universe_sectors[s] & vendor_tickers) for s in universe_sectors) > 0
            else 0
        )

    # Bias = deviation from universe weights
    bias_scores = {
        sector: vendor_weights.get(sector, 0) - universe_weights.get(sector, 0)
        for sector in universe_sectors
    }
    max_bias_sector = max(bias_scores, key=lambda s: abs(bias_scores[s]))

    return {
        "coverage_by_sector": coverage_by_sector,
        "universe_weights": universe_weights,
        "vendor_weights": vendor_weights,
        "bias_scores": bias_scores,
        "max_bias": abs(bias_scores[max_bias_sector]),
        "most_biased_sector": max_bias_sector,
    }


def _compute_cap_coverage(
    dataset: VendorDataset, universe: UniverseDefinition
) -> dict:
    """Coverage by market cap quintile."""
    vendor_tickers = set(dataset.tickers)
    uni = universe.data.copy()

    uni["cap_quintile"] = pd.qcut(
        uni["market_cap"], q=5, labels=["Q1_small", "Q2", "Q3", "Q4", "Q5_large"]
    )

    coverage_by_quintile = {}
    for q in ["Q1_small", "Q2", "Q3", "Q4", "Q5_large"]:
        q_tickers = set(uni[uni["cap_quintile"] == q]["ticker"])
        covered = q_tickers & vendor_tickers
        coverage_by_quintile[q] = {
            "count": len(q_tickers),
            "covered": len(covered),
            "rate": len(covered) / len(q_tickers) if q_tickers else 0,
        }

    # Determine bias direction
    small_rate = coverage_by_quintile["Q1_small"]["rate"]
    large_rate = coverage_by_quintile["Q5_large"]["rate"]

    if large_rate > small_rate * 1.5:
        bias = "large"
    elif small_rate > large_rate * 1.5:
        bias = "small"
    else:
        bias = "balanced"

    return {
        "coverage_by_quintile": coverage_by_quintile,
        "bias_direction": bias,
    }


def _compute_coverage_over_time(
    dataset: VendorDataset, universe: UniverseDefinition, config: CoverageConfig
) -> dict:
    """Monthly coverage rate over the dataset's history."""
    universe_tickers = set(universe.tickers)
    df = dataset.data.copy()
    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby("month")["ticker"]
        .apply(lambda x: len(set(x) & universe_tickers) / len(universe_tickers))
        .reset_index()
    )
    monthly.columns = ["month", "coverage_rate"]
    monthly["month_str"] = monthly["month"].astype(str)

    # Linear trend
    if len(monthly) >= 3:
        x = np.arange(len(monthly))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, monthly["coverage_rate"]
        )
    else:
        slope, r_value, p_value = 0.0, 0.0, 1.0

    return {
        "monthly_coverage": monthly[["month_str", "coverage_rate"]].to_dict("records"),
        "trend_slope": slope,
        "trend_r_squared": r_value**2,
        "trend_p_value": p_value,
        "n_months": len(monthly),
    }


def _detect_coverage_anomalies(
    monthly_coverage: list[dict], zscore_threshold: float
) -> list[dict]:
    """Find months where coverage drops abnormally."""
    if len(monthly_coverage) < 6:
        return []

    rates = [m["coverage_rate"] for m in monthly_coverage]
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)

    if std_rate == 0:
        return []

    anomalies = []
    for m in monthly_coverage:
        z = (m["coverage_rate"] - mean_rate) / std_rate
        if z < -zscore_threshold:
            anomalies.append({
                "month": m["month_str"],
                "coverage_rate": m["coverage_rate"],
                "zscore": round(z, 2),
            })

    return anomalies


def _compute_coverage_score(
    overall: dict,
    sector_analysis: dict,
    cap_analysis: dict,
    time_analysis: dict,
    config: CoverageConfig,
) -> float:
    """Combine sub-metrics into a 0-100 coverage score.

    Weighting:
    - Overall coverage rate: 40%
    - Sector balance: 20%
    - Cap balance: 20%
    - Trend stability: 20%
    """
    # Overall coverage: sigmoid-like mapping
    rate = overall["coverage_rate"]
    if rate >= config.ideal_coverage:
        rate_score = 100.0
    elif rate <= config.min_acceptable_coverage:
        # Below minimum: harsh penalty, max 20 points
        rate_score = max(0, rate / config.min_acceptable_coverage * 20)
    else:
        # Linear between min and ideal
        rate_score = 20 + 80 * (rate - config.min_acceptable_coverage) / (
            config.ideal_coverage - config.min_acceptable_coverage
        )

    # Sector balance: penalize deviation from universe
    max_bias = sector_analysis["max_bias"]
    sector_score = max(0, 100 - (max_bias / config.sector_bias_threshold) * 50)

    # Cap balance
    cap_score = 100.0 if cap_analysis["bias_direction"] == "balanced" else 50.0

    # Trend stability
    slope = time_analysis["trend_slope"]
    if slope >= 0:
        trend_score = 100.0
    elif slope >= -0.01:
        trend_score = 60.0
    else:
        trend_score = max(0, 30 + slope * 1000)

    composite = 0.4 * rate_score + 0.2 * sector_score + 0.2 * cap_score + 0.2 * trend_score

    # Hard ceiling: if coverage is below minimum, the other dimensions can't save it
    if rate < config.min_acceptable_coverage:
        composite = min(composite, rate_score + 10)

    return round(min(100, max(0, composite)), 1)


def _compute_confidence(
    dataset: VendorDataset,
    universe: UniverseDefinition,
    time_analysis: dict,
) -> float:
    """How confident are we in this module's score?

    Low confidence when:
    - Very little history (< 12 months)
    - Very small universe overlap
    - High variance in coverage over time
    """
    months = time_analysis["n_months"]
    if months < 3:
        return 0.3
    elif months < 12:
        time_conf = 0.5 + 0.5 * (months / 12)
    else:
        time_conf = 1.0

    overlap_rate = len(
        set(dataset.tickers) & set(universe.tickers)
    ) / max(1, len(universe.tickers))
    overlap_conf = min(1.0, overlap_rate * 2)  # full confidence at 50%+ overlap

    return round(min(time_conf, overlap_conf), 2)


def _generate_narrative(
    overall: dict,
    sector_analysis: dict,
    cap_analysis: dict,
    time_analysis: dict,
    anomalies: list[dict],
    warnings: list[str],
) -> str:
    """Generate PM-readable summary of coverage analysis."""
    rate = overall["coverage_rate"]
    parts = []

    # Lead with the headline number
    parts.append(
        f"This dataset covers {overall['covered_tickers']} of "
        f"{overall['universe_size']} tickers in the target universe "
        f"({rate:.1%} coverage)."
    )

    # Sector insight
    if sector_analysis["max_bias"] > 0.1:
        parts.append(
            f"Coverage is unevenly distributed across sectors — "
            f"{sector_analysis['most_biased_sector']} is notably "
            f"{'over' if sector_analysis['bias_scores'][sector_analysis['most_biased_sector']] > 0 else 'under'}represented."
        )
    else:
        parts.append("Sector coverage is roughly proportional to the universe.")

    # Cap insight
    if cap_analysis["bias_direction"] != "balanced":
        parts.append(
            f"The data skews toward {cap_analysis['bias_direction']}-cap names, "
            f"which may limit utility for parts of the book."
        )

    # Trend insight
    slope = time_analysis["trend_slope"]
    if slope > 0.005:
        parts.append("Coverage has been expanding over time, which is a positive sign.")
    elif slope < -0.005:
        parts.append(
            "Coverage has been declining, which raises concerns about "
            "the vendor's ongoing data collection efforts."
        )

    # Anomalies
    if anomalies:
        months = ", ".join(a["month"] for a in anomalies[:3])
        parts.append(f"Unusual coverage drops were detected in: {months}.")

    return " ".join(parts)
