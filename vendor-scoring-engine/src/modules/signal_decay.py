"""Signal decay analysis module.

Evaluates the predictive power of a vendor signal and how that power
decays over time. Answers: "Does this signal predict returns, at what
horizon, and how consistently?"

Metrics computed:
- Information Coefficient (IC) at multiple lags via cross-sectional Spearman correlation
- IC t-statistic and p-value (statistical significance per lag)
- ICIR = mean(IC) / std(IC) — consistency of the signal's predictive power
- IC half-life — exponential decay fit to IC-vs-lag curve
- Rolling IC — 126-day rolling IC to detect regime breaks
- Signal autocorrelation — how persistent the signal itself is (AR(1))

Note: This module measures raw predictive power only. Risk-adjusted IC
(after stripping Fama-French factors) is handled by the orthogonality module.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from src.ingestion.schema import VendorDataset
from src.scoring.results import ModuleResult

logger = logging.getLogger(__name__)


@dataclass
class SignalDecayConfig:
    """Configuration for signal decay evaluation."""

    lags_days: list[int] = field(default_factory=lambda: [1, 5, 10, 21, 63])
    target_horizon_days: int = 21  # PM's intended holding period
    min_obs_per_date: int = 20  # minimum cross-section size to include a date
    min_dates_for_ic: int = 24  # minimum date points for reliable IC
    min_ic_threshold: float = 0.02  # IC below this is indistinguishable from noise
    good_ic_threshold: float = 0.05  # IC considered "good"
    great_ic_threshold: float = 0.10  # IC considered "excellent"
    good_icir_threshold: float = 0.5  # ICIR above this is "good"
    rolling_ic_window: int = 126  # trading days for rolling IC stability


def evaluate_signal_decay(
    dataset: VendorDataset,
    returns_data: pd.DataFrame,
    config: SignalDecayConfig | None = None,
) -> ModuleResult:
    """Run the full signal decay evaluation.

    Args:
        dataset: The vendor dataset to evaluate. Must have [ticker, date, signal_value].
        returns_data: Forward return data with columns [ticker, date] plus at least
            one of [fwd_1d, fwd_5d, fwd_10d, fwd_21d, fwd_63d]. Additional lag
            columns are ignored; missing lag columns reduce confidence.
        config: Optional configuration overrides.

    Returns:
        ModuleResult with signal decay score, diagnostics, and narrative.

    Raises:
        ValueError: If returns_data is missing both ticker/date columns and all
            recognized lag columns.
    """
    if config is None:
        config = SignalDecayConfig()

    _validate_returns_data(returns_data, config)

    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []

    # Determine which lags we can actually evaluate
    available_lags = _get_available_lags(returns_data, config.lags_days)
    diagnostics["lags_evaluated"] = available_lags

    if not available_lags:
        return ModuleResult(
            module_name="signal_decay",
            score=0.0,
            confidence=0.0,
            diagnostics=diagnostics,
            narrative=(
                "Cannot evaluate signal decay: no forward return columns found in returns_data."
            ),
            warnings=["No recognized forward return columns (fwd_Xd) found in returns_data."],
        )

    # Merge signal with returns
    merged = _merge_signal_returns(dataset.data, returns_data)
    diagnostics["n_dates_evaluated"] = merged["date"].nunique()
    diagnostics["n_tickers_evaluated"] = merged["ticker"].nunique()

    # 1. IC by lag
    ic_stats = _compute_ic_by_lag(merged, available_lags, config.min_obs_per_date)
    diagnostics["ic_by_lag"] = ic_stats

    # Check for insufficient data
    total_dates = diagnostics["n_dates_evaluated"]
    if total_dates < config.min_dates_for_ic:
        warnings.append(
            f"Only {total_dates} dates available; IC estimates may be unreliable "
            f"(recommend >= {config.min_dates_for_ic})"
        )

    # 2. ICIR by lag
    icir_by_lag = {lag: stats_["icir"] for lag, stats_ in ic_stats.items()}
    diagnostics["icir_by_lag"] = icir_by_lag

    # 3. IC half-life (exponential decay fit)
    halflife_result = _fit_ic_halflife(ic_stats, available_lags)
    diagnostics["ic_halflife_days"] = halflife_result["halflife_days"]
    diagnostics["ic_halflife_r_squared"] = halflife_result["r_squared"]
    diagnostics["ic_non_monotonic"] = halflife_result["non_monotonic"]

    if halflife_result["non_monotonic"]:
        warnings.append(
            "IC curve is non-monotonic across lags — signal may be regime-dependent "
            "or have multiple alpha sources at different horizons."
        )

    # 4. Rolling IC at the target-closest lag
    target_lag = _closest_lag(available_lags, config.target_horizon_days)
    fwd_col = f"fwd_{target_lag}d"
    rolling_ic = _compute_rolling_ic(
        merged, fwd_col, config.rolling_ic_window, config.min_obs_per_date
    )
    diagnostics["rolling_ic"] = rolling_ic
    diagnostics["target_lag"] = target_lag

    rolling_ic_values = [pt["ic"] for pt in rolling_ic if pt["ic"] is not None]
    if len(rolling_ic_values) >= 4:
        rolling_std = float(np.std(rolling_ic_values))
        diagnostics["rolling_ic_std"] = rolling_std
        if rolling_std > 0.05:
            warnings.append(
                f"Rolling IC is unstable (std={rolling_std:.3f}) — "
                "signal alpha may be regime-dependent."
            )

    # 5. Signal autocorrelation
    signal_ac = _compute_signal_autocorrelation(merged)
    diagnostics["signal_autocorrelation"] = signal_ac

    # 6. Target-horizon summary stats (for scoring and narrative)
    target_stats = ic_stats.get(target_lag, {})
    target_ic = target_stats.get("mean_ic", 0.0)
    target_icir = target_stats.get("icir", 0.0)
    target_pval = target_stats.get("p_value", 1.0)

    diagnostics["target_lag_ic"] = target_ic
    diagnostics["target_lag_icir"] = target_icir
    diagnostics["target_lag_p_value"] = target_pval

    # Negative IC warning
    if target_ic < -config.min_ic_threshold:
        warnings.append(
            f"Signal is negatively predictive at the target horizon "
            f"(IC={target_ic:.4f}). Consider inverting the signal."
        )

    # 7. Compute score and confidence
    score = _compute_decay_score(target_ic, target_icir, target_pval, halflife_result, config)
    confidence = _compute_confidence(merged, available_lags, config.lags_days, total_dates, config)

    narrative = _generate_narrative(
        target_ic,
        target_icir,
        target_lag,
        halflife_result,
        rolling_ic_values,
        signal_ac,
        warnings,
        config,
    )

    return ModuleResult(
        module_name="signal_decay",
        score=score,
        confidence=confidence,
        diagnostics=diagnostics,
        narrative=narrative,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_returns_data(returns_data: pd.DataFrame, config: SignalDecayConfig) -> None:
    """Validate that returns_data has the minimum required structure."""
    required_base = {"ticker", "date"}
    missing = required_base - set(returns_data.columns)
    if missing:
        raise ValueError(f"returns_data is missing required columns: {missing}")

    recognized_lags = {f"fwd_{lag}d" for lag in config.lags_days}
    present = recognized_lags & set(returns_data.columns)
    if not present:
        raise ValueError(
            f"returns_data has none of the expected forward return columns: "
            f"{sorted(recognized_lags)}"
        )


def _get_available_lags(returns_data: pd.DataFrame, lags_days: list[int]) -> list[int]:
    """Return the subset of configured lags that are present in returns_data."""
    return [lag for lag in lags_days if f"fwd_{lag}d" in returns_data.columns]


def _closest_lag(available_lags: list[int], target: int) -> int:
    """Return the lag closest to target from the available set."""
    return min(available_lags, key=lambda lag: abs(lag - target))


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def _merge_signal_returns(
    signal_df: pd.DataFrame,
    returns_data: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join signal with returns on (ticker, date).

    Args:
        signal_df: DataFrame with [ticker, date, signal_value].
        returns_data: DataFrame with [ticker, date, fwd_Xd, ...].

    Returns:
        Merged DataFrame with signal_value and all available fwd_Xd columns.
    """
    sig = signal_df[["ticker", "date", "signal_value"]].copy()
    ret = returns_data.copy()

    if not pd.api.types.is_datetime64_any_dtype(ret["date"]):
        ret["date"] = pd.to_datetime(ret["date"])

    merged = sig.merge(ret, on=["ticker", "date"], how="inner")
    logger.debug(
        "Merged signal+returns: %d rows, %d dates, %d tickers",
        len(merged),
        merged["date"].nunique(),
        merged["ticker"].nunique(),
    )
    return merged


def _compute_ic_by_lag(
    merged: pd.DataFrame,
    available_lags: list[int],
    min_obs_per_date: int,
) -> dict[int, dict[str, float]]:
    """Compute cross-sectional IC (Spearman) for each lag.

    For each lag L:
    - Group by date
    - Compute cross-sectional Spearman rank correlation between signal_value and fwd_Ld
    - Aggregate: mean IC, std IC, ICIR, t-stat, p-value

    Args:
        merged: Merged signal + returns DataFrame.
        available_lags: List of lag days with corresponding fwd_Xd columns.
        min_obs_per_date: Minimum cross-section size to include a date.

    Returns:
        Dict keyed by lag_days with IC statistics.
    """
    result = {}

    for lag in available_lags:
        fwd_col = f"fwd_{lag}d"
        ic_series = []

        for date, group in merged.groupby("date"):
            # Drop rows missing either signal or return for this lag
            clean = group[["signal_value", fwd_col]].dropna()
            if len(clean) < min_obs_per_date:
                continue

            if clean["signal_value"].std() == 0 or clean[fwd_col].std() == 0:
                continue

            corr, _ = stats.spearmanr(clean["signal_value"], clean[fwd_col])
            if not np.isnan(corr):
                ic_series.append(corr)

        if not ic_series:
            result[lag] = {
                "mean_ic": 0.0,
                "std_ic": 0.0,
                "icir": 0.0,
                "t_stat": 0.0,
                "p_value": 1.0,
                "n_dates": 0,
            }
            continue

        ic_arr = np.array(ic_series)
        n = len(ic_arr)
        mean_ic = float(np.mean(ic_arr))
        std_ic = float(np.std(ic_arr, ddof=1)) if n > 1 else 0.0

        if std_ic > 0 and n > 1:
            icir = mean_ic / std_ic
            t_stat = mean_ic / (std_ic / np.sqrt(n))
            p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        else:
            icir = 0.0
            t_stat = 0.0
            p_value = 1.0

        result[lag] = {
            "mean_ic": round(mean_ic, 6),
            "std_ic": round(std_ic, 6),
            "icir": round(icir, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "n_dates": n,
        }

        logger.debug(
            "IC lag=%dd: mean=%.4f std=%.4f icir=%.3f p=%.4f (n=%d dates)",
            lag,
            mean_ic,
            std_ic,
            icir,
            p_value,
            n,
        )

    return result


def _fit_ic_halflife(
    ic_stats: dict[int, dict[str, float]],
    available_lags: list[int],
) -> dict[str, Any]:
    """Fit an exponential decay model to IC vs lag.

    Model: IC(L) = IC₀ · exp(−λ · L)
    Half-life: ln(2) / λ

    Only fits if IC is positive at the shortest lag and the curve has at
    least 3 positive IC values (otherwise fit is meaningless).

    Args:
        ic_stats: IC statistics keyed by lag_days.
        available_lags: Sorted list of available lag values.

    Returns:
        Dict with halflife_days, r_squared, non_monotonic flag.
    """
    lags_sorted = sorted(available_lags)
    ic_values = [ic_stats[lag]["mean_ic"] for lag in lags_sorted]

    # Monotonicity check
    non_monotonic = any(
        ic_values[i] < ic_values[i + 1]
        for i in range(len(ic_values) - 1)
        if ic_values[i] > 0 and ic_values[i + 1] > 0
    )

    # Only fit if IC is positive at the first lag and we have ≥ 3 positive values
    positive_pairs = [(lag, ic) for lag, ic in zip(lags_sorted, ic_values) if ic > 0]
    if len(positive_pairs) < 3 or ic_values[0] <= 0:
        return {
            "halflife_days": None,
            "r_squared": None,
            "non_monotonic": non_monotonic,
        }

    fit_lags = np.array([p[0] for p in positive_pairs], dtype=float)
    fit_ics = np.array([p[1] for p in positive_pairs], dtype=float)

    try:

        def exp_decay(x: np.ndarray, ic0: float, lam: float) -> np.ndarray:
            return ic0 * np.exp(-lam * x)

        popt, _ = curve_fit(
            exp_decay,
            fit_lags,
            fit_ics,
            p0=[fit_ics[0], 0.05],
            bounds=([0, 1e-6], [1.0, 1.0]),
            maxfev=2000,
        )
        ic0_fit, lam_fit = popt
        halflife = float(np.log(2) / lam_fit)

        # R² of the fit on the positive subset
        predicted = exp_decay(fit_lags, ic0_fit, lam_fit)
        ss_res = float(np.sum((fit_ics - predicted) ** 2))
        ss_tot = float(np.sum((fit_ics - np.mean(fit_ics)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "halflife_days": round(halflife, 1),
            "r_squared": round(r_squared, 4),
            "non_monotonic": non_monotonic,
        }

    except (RuntimeError, ValueError) as exc:
        logger.debug("IC half-life fit failed: %s", exc)
        return {
            "halflife_days": None,
            "r_squared": None,
            "non_monotonic": non_monotonic,
        }


def _compute_rolling_ic(
    merged: pd.DataFrame,
    fwd_col: str,
    window: int,
    min_obs_per_date: int,
) -> list[dict[str, Any]]:
    """Compute rolling cross-sectional IC over a trailing window.

    Args:
        merged: Merged signal + returns DataFrame.
        fwd_col: Forward return column name to use (e.g. "fwd_21d").
        window: Number of dates to include in each rolling window.
        min_obs_per_date: Minimum cross-section size per date.

    Returns:
        List of dicts [{"date": str, "ic": float | None}].
    """
    if fwd_col not in merged.columns:
        return []

    clean = merged[["date", "ticker", "signal_value", fwd_col]].dropna()
    dates = sorted(clean["date"].unique())

    if len(dates) < window:
        return []

    results = []
    for i in range(window - 1, len(dates)):
        window_dates = dates[i - window + 1 : i + 1]
        window_data = clean[clean["date"].isin(window_dates)]

        # Compute IC across all ticker-date rows in the window
        if len(window_data) < min_obs_per_date:
            results.append({"date": str(dates[i].date()), "ic": None})
            continue

        if window_data["signal_value"].std() == 0 or window_data[fwd_col].std() == 0:
            results.append({"date": str(dates[i].date()), "ic": None})
            continue

        corr, _ = stats.spearmanr(window_data["signal_value"], window_data[fwd_col])
        results.append(
            {
                "date": str(dates[i].date()),
                "ic": round(float(corr), 6) if not np.isnan(corr) else None,
            }
        )

    return results


def _compute_signal_autocorrelation(merged: pd.DataFrame) -> float:
    """Compute average AR(1) autocorrelation of the signal across tickers.

    A high autocorrelation (> 0.7) means the signal changes slowly,
    which is needed for strategies with longer holding periods.

    Args:
        merged: Merged DataFrame with [ticker, date, signal_value].

    Returns:
        Mean AR(1) autocorrelation across tickers with sufficient data.
    """
    autocorrs = []
    for ticker, grp in merged.groupby("ticker"):
        ts = grp.sort_values("date")["signal_value"].dropna()
        if len(ts) < 10 or ts.std() == 0:
            continue
        corr = float(ts.autocorr(lag=1))
        if not np.isnan(corr):
            autocorrs.append(corr)

    return round(float(np.mean(autocorrs)), 4) if autocorrs else 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _compute_decay_score(
    target_ic: float,
    target_icir: float,
    target_pval: float,
    halflife_result: dict[str, Any],
    config: SignalDecayConfig,
) -> float:
    """Compute composite 0-100 signal decay score.

    Weighting:
    - IC magnitude at target horizon: 30%
    - ICIR at target horizon: 30%
    - Half-life alignment with target horizon: 20%
    - Statistical significance: 20%

    Args:
        target_ic: Mean IC at the lag closest to target_horizon_days.
        target_icir: ICIR at that lag.
        target_pval: p-value of the IC at that lag.
        halflife_result: Dict from _fit_ic_halflife.
        config: Scoring configuration.

    Returns:
        Score in [0, 100].
    """
    # 1. IC magnitude
    if target_ic <= 0:
        ic_score = 0.0
    elif target_ic >= config.great_ic_threshold:
        ic_score = 100.0
    elif target_ic >= config.good_ic_threshold:
        ic_score = 60.0 + 40.0 * (
            (target_ic - config.good_ic_threshold)
            / (config.great_ic_threshold - config.good_ic_threshold)
        )
    elif target_ic >= config.min_ic_threshold:
        ic_score = 20.0 + 40.0 * (
            (target_ic - config.min_ic_threshold)
            / (config.good_ic_threshold - config.min_ic_threshold)
        )
    else:
        ic_score = max(0.0, target_ic / config.min_ic_threshold * 20.0)

    # 2. ICIR
    if target_icir <= 0:
        icir_score = 0.0
    elif target_icir >= config.good_icir_threshold:
        icir_score = 100.0
    else:
        icir_score = (target_icir / config.good_icir_threshold) * 100.0

    # 3. Half-life alignment
    halflife_days = halflife_result.get("halflife_days")
    if halflife_days is None:
        halflife_score = 40.0  # uncertain, give partial credit
    else:
        ratio = halflife_days / config.target_horizon_days
        # Score peaks at ratio=1.0 (perfect alignment), decays symmetrically
        log_ratio = abs(np.log(max(ratio, 1e-6)))
        halflife_score = max(0.0, 100.0 - log_ratio * 60.0)

    # 4. Statistical significance
    if target_pval <= 0.01:
        sig_score = 100.0
    elif target_pval <= 0.05:
        sig_score = 60.0 + 40.0 * ((0.05 - target_pval) / 0.04)
    elif target_pval <= 0.10:
        sig_score = 30.0 + 30.0 * ((0.10 - target_pval) / 0.05)
    else:
        sig_score = max(0.0, (1.0 - target_pval) * 30.0)

    composite = 0.30 * ic_score + 0.30 * icir_score + 0.20 * halflife_score + 0.20 * sig_score

    # Hard floor: negative IC caps the score
    if target_ic < -config.min_ic_threshold:
        composite = min(composite, 20.0)

    return round(min(100.0, max(0.0, composite)), 1)


def _compute_confidence(
    merged: pd.DataFrame,
    available_lags: list[int],
    configured_lags: list[int],
    n_dates: int,
    config: SignalDecayConfig,
) -> float:
    """Compute confidence in the module's score.

    Low confidence when:
    - Fewer than min_dates_for_ic dates are available
    - Only a subset of configured lags could be evaluated
    - Cross-section size is small on average

    Args:
        merged: Merged signal + returns DataFrame.
        available_lags: Lags that had return data.
        configured_lags: All configured lags.
        n_dates: Number of distinct evaluation dates.
        config: Configuration.

    Returns:
        Confidence in [0, 1].
    """
    # History confidence
    if n_dates < 6:
        time_conf = 0.2
    elif n_dates < config.min_dates_for_ic:
        time_conf = 0.3 + 0.5 * (n_dates / config.min_dates_for_ic)
    else:
        time_conf = 1.0

    # Lag coverage confidence
    lag_conf = len(available_lags) / max(1, len(configured_lags))

    # Cross-section size confidence
    avg_cs = merged.groupby("date")["ticker"].nunique().mean()
    cs_conf = min(1.0, avg_cs / 50.0)  # full confidence at 50+ tickers per date

    confidence = 0.5 * time_conf + 0.3 * lag_conf + 0.2 * cs_conf
    return round(min(1.0, max(0.0, confidence)), 2)


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------


def _generate_narrative(
    target_ic: float,
    target_icir: float,
    target_lag: int,
    halflife_result: dict[str, Any],
    rolling_ic_values: list[float],
    signal_ac: float,
    warnings: list[str],
    config: SignalDecayConfig,
) -> str:
    """Generate a PM-readable summary of signal decay analysis.

    Args:
        target_ic: IC at the target-horizon lag.
        target_icir: ICIR at the target-horizon lag.
        target_lag: The actual lag evaluated (closest to target_horizon_days).
        halflife_result: Dict from _fit_ic_halflife.
        rolling_ic_values: List of rolling IC values for stability comment.
        signal_ac: Average AR(1) autocorrelation.
        warnings: List of warnings generated by evaluate_signal_decay.
        config: Scoring config.

    Returns:
        Multi-sentence plain-English narrative.
    """
    parts = []

    # Lead with IC at target horizon
    if target_ic >= config.great_ic_threshold:
        quality = "strong"
    elif target_ic >= config.good_ic_threshold:
        quality = "moderate"
    elif target_ic >= config.min_ic_threshold:
        quality = "weak but non-trivial"
    elif target_ic > 0:
        quality = "marginal"
    else:
        quality = "negative (or zero)"

    parts.append(
        f"At the {target_lag}-day horizon "
        f"(closest to the {config.target_horizon_days}-day target), "
        f"the signal shows {quality} predictive power "
        f"(IC={target_ic:.4f}, ICIR={target_icir:.2f})."
    )

    # Half-life insight
    halflife_days = halflife_result.get("halflife_days")
    if halflife_days is not None:
        if abs(halflife_days - config.target_horizon_days) / config.target_horizon_days < 0.3:
            hl_comment = (
                f"The IC half-life ({halflife_days:.0f} days) aligns well with the "
                f"target holding period."
            )
        elif halflife_days < config.target_horizon_days * 0.5:
            hl_comment = (
                f"The signal decays quickly (half-life ~{halflife_days:.0f} days), "
                f"faster than the {config.target_horizon_days}-day target — "
                f"this strategy may require more frequent rebalancing."
            )
        else:
            hl_comment = (
                f"The IC half-life (~{halflife_days:.0f} days) exceeds the target horizon, "
                f"suggesting the signal is persistent and may work well at longer frequencies."
            )
        parts.append(hl_comment)
    else:
        parts.append(
            "A reliable IC half-life could not be estimated — "
            "the IC curve does not follow a clean exponential decay."
        )

    # Stability
    if rolling_ic_values:
        ic_min = min(rolling_ic_values)
        ic_max = max(rolling_ic_values)
        if ic_max - ic_min < 0.03:
            parts.append("Rolling IC is stable across the sample period.")
        elif ic_min < 0 < ic_max:
            parts.append(
                f"Rolling IC has been both positive and negative (range: {ic_min:.3f} to "
                f"{ic_max:.3f}), indicating regime-dependent alpha."
            )
        else:
            parts.append(
                f"Rolling IC shows some variability (range: {ic_min:.3f} to {ic_max:.3f})."
            )

    # Signal persistence
    if signal_ac > 0.8:
        parts.append(
            f"The signal is highly autocorrelated (AR(1)={signal_ac:.2f}), "
            "making it suitable for lower-turnover strategies."
        )
    elif signal_ac < 0.3:
        parts.append(
            f"The signal changes rapidly (AR(1)={signal_ac:.2f}), "
            "which may lead to high turnover costs."
        )

    return " ".join(parts)
