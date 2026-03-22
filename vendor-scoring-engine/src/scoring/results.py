"""Evaluation result containers and score aggregation.

Every evaluation module returns a ModuleResult. The aggregator
combines them into a composite VendorScore.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ModuleResult:
    """Output of a single evaluation module.

    Args:
        module_name: Which module produced this (e.g. "coverage", "orthogonality")
        score: 0-100 composite score for this dimension
        confidence: 0-1 how confident the module is in its score
                    (low confidence = not enough data to evaluate properly)
        diagnostics: Structured dict of intermediate metrics for the dashboard
        narrative: Plain-English summary a PM can read and act on
        warnings: List of issues found (e.g. "suspected backfill after 2020-03")
        plots: List of matplotlib/plotly figure objects for the report
    """

    module_name: str
    score: float
    confidence: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    warnings: list[str] = field(default_factory=list)
    plots: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError(f"Score must be 0-100, got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass
class VendorScore:
    """Composite evaluation of a vendor dataset across all modules.

    This is the top-level object that gets stored and displayed.
    """

    vendor_name: str
    dataset_name: str
    evaluated_at: datetime
    composite_score: float
    module_results: dict[str, ModuleResult]
    recommendation: str  # "strong_pass", "pass", "conditional", "fail"
    executive_summary: str  # 2-3 sentence summary for a PM

    @classmethod
    def from_results(
        cls,
        vendor_name: str,
        dataset_name: str,
        results: list[ModuleResult],
        weights: dict[str, float] | None = None,
    ) -> "VendorScore":
        """Aggregate module results into a composite score.

        Args:
            vendor_name: Name of the data vendor
            dataset_name: Name of the specific dataset
            results: List of ModuleResult from each evaluation module
            weights: Optional weight overrides per module. Defaults to equal weight
                     adjusted by confidence.
        """
        if not results:
            raise ValueError("Cannot score with zero module results")

        module_dict = {r.module_name: r for r in results}

        # Default weights: equal, but scaled by confidence
        if weights is None:
            weights = {r.module_name: 1.0 for r in results}

        # Normalize weights
        total_weight = sum(
            weights.get(r.module_name, 1.0) * r.confidence for r in results
        )
        if total_weight == 0:
            composite = 0.0
        else:
            composite = sum(
                r.score * weights.get(r.module_name, 1.0) * r.confidence
                for r in results
            ) / total_weight

        # Determine recommendation
        recommendation = _compute_recommendation(composite, results)
        summary = _generate_executive_summary(composite, results, recommendation)

        return cls(
            vendor_name=vendor_name,
            dataset_name=dataset_name,
            evaluated_at=datetime.now(),
            composite_score=round(composite, 1),
            module_results=module_dict,
            recommendation=recommendation,
            executive_summary=summary,
        )


def _compute_recommendation(
    composite: float, results: list[ModuleResult]
) -> str:
    """Map composite score + red flags to a recommendation.

    A high composite score can still get downgraded if critical
    modules (orthogonality, backtest_integrity) flag issues.
    """
    # Hard fails: if orthogonality or backtest integrity score below 30
    critical_modules = {"orthogonality", "backtest_integrity"}
    for r in results:
        if r.module_name in critical_modules and r.score < 30:
            return "fail"

    if composite >= 75:
        return "strong_pass"
    elif composite >= 55:
        return "pass"
    elif composite >= 35:
        return "conditional"
    else:
        return "fail"


def _generate_executive_summary(
    composite: float,
    results: list[ModuleResult],
    recommendation: str,
) -> str:
    """Generate a 2-3 sentence summary for PM consumption."""
    sorted_results = sorted(results, key=lambda r: r.score)
    weakest = sorted_results[0] if sorted_results else None
    strongest = sorted_results[-1] if sorted_results else None

    rec_text = {
        "strong_pass": "This dataset shows strong potential for alpha generation.",
        "pass": "This dataset meets baseline criteria and warrants deeper investigation.",
        "conditional": "This dataset has meaningful limitations that need resolution before adoption.",
        "fail": "This dataset does not meet minimum standards for production use.",
    }

    summary = f"Composite score: {composite:.0f}/100. {rec_text[recommendation]}"

    if weakest and strongest and weakest.module_name != strongest.module_name:
        summary += (
            f" Strongest dimension: {strongest.module_name} ({strongest.score:.0f}). "
            f"Weakest: {weakest.module_name} ({weakest.score:.0f})."
        )

    # Append any critical warnings
    all_warnings = [w for r in results for w in r.warnings]
    if all_warnings:
        summary += f" ⚠ {len(all_warnings)} warning(s) flagged."

    return summary
