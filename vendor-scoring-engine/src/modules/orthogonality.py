"""Orthogonality analysis module.

Evaluates whether a vendor signal adds information beyond well-known
systematic risk factors (Fama-French 5 + momentum). Answers: "Is this
signal just levered tech exposure, or does it carry genuine orthogonal alpha?"

Approach: cross-sectional factor neutralization.
For each date, regress the signal cross-sectionally on the provided factor
exposures via OLS (Ridge when multicollinearity is detected). Compute IC
on both the raw signal and the factor-residualized signal, then compare.

Metrics computed:
- Factor model R²: fraction of signal's cross-sectional variance explained
  by known factors (lower = more orthogonal)
- IC degradation: how much does IC drop after neutralization (lower = better)
- Residual IC: predictive power after stripping factor contamination
- Residual IC significance: t-stat / p-value of residual IC
- Factor loadings: which factors dominate the signal (for PM narrative)
- Pairwise factor correlations: multicollinearity diagnostic
- VIF: per-factor variance inflation factor
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge

from src.ingestion.schema import VendorDataset
from src.scoring.results import ModuleResult

logger = logging.getLogger(__name__)

_DEFAULT_FACTORS = ["size", "value", "momentum", "quality", "investment", "mkt_beta"]


@dataclass
class OrthogonalityConfig:
    """Configuration for orthogonality evaluation.

    Args:
        factors: Factor column names expected in factor_exposures DataFrame.
            The module will use whichever of these are actually present.
        target_horizon_days: Forward return lag to use for IC computation.
            Should match the PM's intended holding period.
        min_obs_for_regression: Minimum cross-section size per date to
            include a date in the regression.
        min_dates: Minimum number of dates required for reliable estimates.
        high_r2_threshold: Factor model R² above this triggers a warning.
        ic_degradation_warn: IC degradation fraction above this triggers a warning.
        ic_degradation_fail: IC degradation fraction above this is hard-fail territory.
        vif_threshold: VIF above this automatically switches to Ridge regression.
    """

    factors: list[str] = field(default_factory=lambda: list(_DEFAULT_FACTORS))
    target_horizon_days: int = 21
    min_obs_for_regression: int = 30
    min_dates: int = 24
    high_r2_threshold: float = 0.30
    ic_degradation_warn: float = 0.40
    ic_degradation_fail: float = 0.70
    vif_threshold: float = 5.0


def evaluate_orthogonality(
    dataset: VendorDataset,
    factor_exposures: pd.DataFrame,
    returns_data: pd.DataFrame,
    config: OrthogonalityConfig | None = None,
) -> ModuleResult:
    """Run the full orthogonality evaluation.

    Args:
        dataset: The vendor dataset to evaluate. Must have [ticker, date,
            signal_value].
        factor_exposures: Stock-level factor exposure scores with columns
            [ticker, date] plus at least one factor column (e.g. size,
            value, momentum). Values should be cross-sectionally z-scored
            per date. Use src.ingestion.factor_utils.compute_factor_proxies
            to build this from universe + price data.
        returns_data: Forward return data with columns [ticker, date] plus
            at least one fwd_Xd column matching target_horizon_days. Columns
            like fwd_21d, fwd_5d, etc. are recognized.
        config: Optional configuration overrides.

    Returns:
        ModuleResult with orthogonality score (0–100), confidence, rich
        diagnostics dict, and PM-readable narrative.
    """
    if config is None:
        config = OrthogonalityConfig()

    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []

    # --- Identify target forward-return column ---
    fwd_col = _find_fwd_col(returns_data, config.target_horizon_days)
    if fwd_col is None:
        return ModuleResult(
            module_name="orthogonality",
            score=0.0,
            confidence=0.0,
            diagnostics=diagnostics,
            narrative=(
                "Cannot evaluate orthogonality: no recognized forward return column "
                f"(fwd_Xd) found in returns_data near the {config.target_horizon_days}-day horizon."
            ),
            warnings=[
                f"No fwd_Xd column found near target_horizon_days={config.target_horizon_days}"
            ],
        )

    # --- Identify available factors ---
    available_factors = _identify_available_factors(factor_exposures, config.factors)
    if not available_factors:
        return ModuleResult(
            module_name="orthogonality",
            score=0.0,
            confidence=0.0,
            diagnostics=diagnostics,
            narrative=(
                "Cannot evaluate orthogonality: none of the configured factor columns "
                f"({config.factors}) are present in factor_exposures."
            ),
            warnings=["No configured factor columns found in factor_exposures."],
        )

    missing_factors = [f for f in config.factors if f not in available_factors]
    if missing_factors:
        warnings.append(
            f"Factor data unavailable for: {missing_factors}. "
            f"Evaluating with {len(available_factors)} of {len(config.factors)} factors."
        )

    diagnostics["factors_used"] = available_factors

    # --- Three-way merge: signal + factors + returns ---
    merged = _merge_all(dataset.data, factor_exposures, returns_data, available_factors, fwd_col)
    n_dates = merged["date"].nunique()
    n_tickers = merged["ticker"].nunique()
    diagnostics["n_dates_evaluated"] = n_dates
    diagnostics["n_tickers_evaluated"] = n_tickers

    if n_dates < 3:
        return ModuleResult(
            module_name="orthogonality",
            score=0.0,
            confidence=0.1,
            diagnostics=diagnostics,
            narrative="Insufficient overlapping data between signal, factors, and returns.",
            warnings=["Fewer than 3 dates with overlapping signal + factor + return data."],
        )

    if n_dates < config.min_dates:
        warnings.append(
            f"Only {n_dates} dates available (recommend >= {config.min_dates}). "
            "Estimates may be unreliable."
        )

    # --- Multicollinearity check (pooled) → decides OLS vs Ridge ---
    vif_scores = _compute_vif(merged, available_factors)
    diagnostics["vif_scores"] = {k: round(v, 2) for k, v in vif_scores.items()}
    use_ridge = max(vif_scores.values(), default=0.0) > config.vif_threshold

    if use_ridge:
        high_vif = {k: v for k, v in vif_scores.items() if v > config.vif_threshold}
        warnings.append(
            f"Multicollinearity detected (VIF > {config.vif_threshold}): "
            + ", ".join(f"{k}={v:.1f}" for k, v in high_vif.items())
            + ". Switching to Ridge regression."
        )

    diagnostics["used_ridge"] = use_ridge

    # --- Pairwise factor correlation matrix ---
    factor_corr = _compute_pairwise_factor_corr(merged, available_factors)
    diagnostics["factor_pairwise_corr"] = factor_corr

    # --- Neutralize signal: per-date regression → residuals + R² ---
    neutralized, neutralize_meta = _neutralize_signal(
        merged, available_factors, use_ridge, config.min_obs_for_regression
    )
    mean_r2 = neutralize_meta["mean_r2"]
    factor_loadings = neutralize_meta["mean_coefs"]
    r2_per_date = neutralize_meta["r2_per_date"]

    diagnostics["mean_r_squared"] = round(mean_r2, 4)
    diagnostics["r2_per_date"] = [round(r, 4) for r in r2_per_date]
    diagnostics["factor_loadings"] = {k: round(v, 4) for k, v in factor_loadings.items()}

    if factor_loadings:
        dominant = max(factor_loadings, key=lambda f: abs(factor_loadings[f]))
        diagnostics["dominant_factor"] = dominant
    else:
        dominant = None
        diagnostics["dominant_factor"] = None

    if mean_r2 > config.high_r2_threshold:
        warnings.append(
            f"Factor model explains {mean_r2:.1%} of signal cross-sectional variance "
            f"(threshold: {config.high_r2_threshold:.0%}). "
            f"Dominant factor: {dominant}."
        )

    # --- Raw IC time-series ---
    raw_ic_series = _compute_ic_series(
        neutralized, "signal_value", fwd_col, config.min_obs_for_regression
    )
    raw_ic_stats = _compute_ic_stats(raw_ic_series)
    diagnostics["raw_ic"] = round(raw_ic_stats["mean_ic"], 6)
    diagnostics["raw_ic_icir"] = round(raw_ic_stats["icir"], 4)
    diagnostics["raw_ic_p_value"] = round(raw_ic_stats["p_value"], 6)

    # --- Residual IC time-series ---
    residual_ic_series = _compute_ic_series(
        neutralized, "residual_signal", fwd_col, config.min_obs_for_regression
    )
    residual_ic_stats = _compute_ic_stats(residual_ic_series)
    diagnostics["residual_ic"] = round(residual_ic_stats["mean_ic"], 6)
    diagnostics["residual_ic_icir"] = round(residual_ic_stats["icir"], 4)
    diagnostics["residual_ic_t_stat"] = round(residual_ic_stats["t_stat"], 4)
    diagnostics["residual_ic_p_value"] = round(residual_ic_stats["p_value"], 6)
    diagnostics["residual_ic_n_dates"] = residual_ic_stats["n_dates"]

    # --- IC degradation ---
    degradation = _compute_ic_degradation(raw_ic_stats["mean_ic"], residual_ic_stats["mean_ic"])
    diagnostics["ic_degradation_pct"] = degradation["degradation_pct"]
    diagnostics["ic_degradation_interpretable"] = degradation["interpretable"]

    degradation_pct = degradation["degradation_pct"]

    if degradation_pct is not None:
        if degradation_pct > config.ic_degradation_fail * 100:
            warnings.append(
                f"IC degrades by {degradation_pct:.0f}% after factor neutralization — "
                "signal is primarily factor exposure with limited orthogonal alpha."
            )
        elif degradation_pct > config.ic_degradation_warn * 100:
            warnings.append(
                f"IC degrades by {degradation_pct:.0f}% after factor neutralization "
                f"(warn threshold: {config.ic_degradation_warn:.0%})."
            )

    if residual_ic_stats["p_value"] > 0.10 and residual_ic_stats["n_dates"] >= config.min_dates:
        warnings.append(
            f"Residual IC is not statistically significant "
            f"(p={residual_ic_stats['p_value']:.3f}). "
            "Cannot rule out that orthogonal IC is due to chance."
        )

    # --- Score and confidence ---
    score = _compute_orthogonality_score(
        degradation_pct=degradation_pct,
        residual_ic=residual_ic_stats["mean_ic"],
        mean_r2=mean_r2,
        p_value=residual_ic_stats["p_value"],
        config=config,
    )
    confidence = _compute_confidence(
        n_dates=n_dates,
        n_factors_used=len(available_factors),
        n_factors_configured=len(config.factors),
        n_tickers=n_tickers,
        config=config,
    )

    narrative = _generate_narrative(
        raw_ic=raw_ic_stats["mean_ic"],
        residual_ic=residual_ic_stats["mean_ic"],
        degradation_pct=degradation_pct,
        mean_r2=mean_r2,
        factor_loadings=factor_loadings,
        dominant_factor=dominant,
        residual_t_stat=residual_ic_stats["t_stat"],
        residual_p_value=residual_ic_stats["p_value"],
        available_factors=available_factors,
        warnings=warnings,
        config=config,
    )

    return ModuleResult(
        module_name="orthogonality",
        score=score,
        confidence=confidence,
        diagnostics=diagnostics,
        narrative=narrative,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Input validation and setup
# ---------------------------------------------------------------------------


def _find_fwd_col(returns_data: pd.DataFrame, target_horizon: int) -> str | None:
    """Find the forward return column closest to target_horizon_days.

    Args:
        returns_data: DataFrame with columns like fwd_1d, fwd_5d, fwd_21d.
        target_horizon: Desired holding period in days.

    Returns:
        Column name of closest available lag, or None if none found.
    """
    candidates = []
    for col in returns_data.columns:
        if col.startswith("fwd_") and col.endswith("d"):
            try:
                lag = int(col[4:-1])
                candidates.append((abs(lag - target_horizon), col))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _identify_available_factors(
    factor_exposures: pd.DataFrame,
    configured_factors: list[str],
) -> list[str]:
    """Return the subset of configured factors present in factor_exposures.

    Args:
        factor_exposures: DataFrame with factor columns.
        configured_factors: Factor names from config.

    Returns:
        Sorted list of factor column names that exist in factor_exposures.
    """
    return [f for f in configured_factors if f in factor_exposures.columns]


def _merge_all(
    signal_df: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    returns_data: pd.DataFrame,
    factors: list[str],
    fwd_col: str,
) -> pd.DataFrame:
    """Three-way inner join: signal + factor_exposures + returns.

    Args:
        signal_df: DataFrame with [ticker, date, signal_value].
        factor_exposures: DataFrame with [ticker, date, *factors].
        returns_data: DataFrame with [ticker, date, fwd_col].
        factors: Factor column names to retain.
        fwd_col: Forward return column name to retain.

    Returns:
        Merged DataFrame with [ticker, date, signal_value, *factors, fwd_col].
        Rows with NaN in any factor or return column are dropped.
    """
    sig = signal_df[["ticker", "date", "signal_value"]].copy()
    fac = factor_exposures[["ticker", "date"] + factors].copy()
    ret = returns_data[["ticker", "date", fwd_col]].copy()

    for df in (sig, fac, ret):
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

    merged = sig.merge(fac, on=["ticker", "date"], how="inner")
    merged = merged.merge(ret, on=["ticker", "date"], how="inner")
    merged = merged.dropna(subset=factors + [fwd_col, "signal_value"])

    logger.debug(
        "Merged orthogonality data: %d rows, %d dates, %d tickers",
        len(merged),
        merged["date"].nunique(),
        merged["ticker"].nunique(),
    )
    return merged


# ---------------------------------------------------------------------------
# Multicollinearity diagnostics
# ---------------------------------------------------------------------------


def _compute_vif(merged: pd.DataFrame, factors: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factor for each factor (pooled across dates).

    VIF_i = 1 / (1 - R²_i) where R²_i is from regressing factor_i
    on all other factors.

    Args:
        merged: Merged DataFrame with factor columns.
        factors: List of factor column names.

    Returns:
        Dict mapping factor name to VIF score.
    """
    if len(factors) < 2:
        return {f: 1.0 for f in factors}

    x_mat = merged[factors].dropna().values.astype(float)
    if x_mat.shape[0] < len(factors) + 2:
        return {f: 1.0 for f in factors}

    vif: dict[str, float] = {}
    for i, name in enumerate(factors):
        y = x_mat[:, i]
        x_other = np.delete(x_mat, i, axis=1)
        reg = LinearRegression(fit_intercept=True)
        try:
            reg.fit(x_other, y)
            ss_res = float(np.sum((y - reg.predict(x_other)) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif[name] = 1.0 / (1.0 - r2) if r2 < 1.0 else 999.0
        except Exception:
            vif[name] = 1.0

    return vif


def _compute_pairwise_factor_corr(
    merged: pd.DataFrame,
    factors: list[str],
) -> dict[str, dict[str, float]]:
    """Compute Pearson correlation matrix among factor columns (pooled).

    Args:
        merged: Merged DataFrame with factor columns.
        factors: List of factor column names.

    Returns:
        Nested dict: corr[factor_a][factor_b] = correlation coefficient.
    """
    if len(factors) < 2:
        return {}
    corr_matrix = merged[factors].corr(method="pearson")
    return {
        f: {g: round(float(corr_matrix.loc[f, g]), 4) for g in factors if g != f} for f in factors
    }


# ---------------------------------------------------------------------------
# Signal neutralization
# ---------------------------------------------------------------------------


def _neutralize_signal(
    merged: pd.DataFrame,
    factors: list[str],
    use_ridge: bool,
    min_obs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Regress signal on factors cross-sectionally per date.

    For each date with sufficient observations, fits OLS or Ridge to
    remove the factor-explained component of the signal.

    Args:
        merged: DataFrame with [ticker, date, signal_value, *factors].
        factors: Factor column names for the regression.
        use_ridge: If True, use Ridge (alpha=1.0) instead of OLS.
        min_obs: Minimum cross-section size to include a date.

    Returns:
        Tuple of:
        - merged copy with added ``residual_signal`` column (NaN where
          regression was skipped)
        - meta dict with r2_per_date, mean_r2, mean_coefs, n_dates_regressed
    """
    merged = merged.copy()
    merged["residual_signal"] = np.nan

    r2_list: list[float] = []
    coef_sum: dict[str, float] = {f: 0.0 for f in factors}
    n_reg = 0

    for date, grp in merged.groupby("date"):
        subset = grp[["signal_value"] + factors].dropna()
        if len(subset) < min_obs:
            continue

        x_date = subset[factors].values.astype(float)
        y = subset["signal_value"].values.astype(float)

        # Drop zero-variance factor columns for this date
        std_mask = x_date.std(axis=0) > 1e-8
        if not std_mask.any():
            continue
        x_use = x_date[:, std_mask]
        factors_use = [f for f, m in zip(factors, std_mask) if m]

        try:
            if use_ridge:
                model = Ridge(alpha=1.0, fit_intercept=True)
                model.fit(x_use, y)
                y_hat = model.predict(x_use)
                coefs = model.coef_
            else:
                model = LinearRegression(fit_intercept=True)
                model.fit(x_use, y)
                y_hat = model.predict(x_use)
                coefs = model.coef_

            residuals = y - y_hat

            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            r2_list.append(r2)
            for f, c in zip(factors_use, coefs):
                coef_sum[f] += c
            n_reg += 1

            # Write residuals back to the merged DataFrame
            merged.loc[subset.index, "residual_signal"] = residuals

        except Exception as exc:
            logger.debug("Neutralization failed for date %s: %s", date, exc)
            continue

    mean_r2 = float(np.mean(r2_list)) if r2_list else 0.0
    mean_coefs = {f: coef_sum[f] / n_reg if n_reg > 0 else 0.0 for f in factors}

    return merged, {
        "r2_per_date": r2_list,
        "mean_r2": mean_r2,
        "mean_coefs": mean_coefs,
        "n_dates_regressed": n_reg,
    }


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------


def _compute_ic_series(
    df: pd.DataFrame,
    signal_col: str,
    fwd_col: str,
    min_obs: int,
) -> list[float]:
    """Compute cross-sectional Spearman IC per date.

    Args:
        df: DataFrame with [date, signal_col, fwd_col].
        signal_col: Column name of signal or residual signal.
        fwd_col: Forward return column.
        min_obs: Minimum cross-section size to include a date.

    Returns:
        List of per-date IC values (dates with insufficient data omitted).
    """
    ic_values = []
    for _, grp in df.groupby("date"):
        clean = grp[[signal_col, fwd_col]].dropna()
        if len(clean) < min_obs:
            continue
        if clean[signal_col].std() < 1e-8 or clean[fwd_col].std() < 1e-8:
            continue
        corr, _ = stats.spearmanr(clean[signal_col], clean[fwd_col])
        if not np.isnan(corr):
            ic_values.append(float(corr))
    return ic_values


def _compute_ic_stats(ic_series: list[float]) -> dict[str, float]:
    """Compute summary statistics from an IC time-series.

    Args:
        ic_series: List of per-date Spearman IC values.

    Returns:
        Dict with mean_ic, std_ic, icir, t_stat, p_value, n_dates.
    """
    if not ic_series:
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "n_dates": 0,
        }

    arr = np.array(ic_series)
    n = len(arr)
    mean_ic = float(np.mean(arr))
    std_ic = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if std_ic > 0 and n > 1:
        icir = mean_ic / std_ic
        t_stat = mean_ic / (std_ic / np.sqrt(n))
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    else:
        icir = 0.0
        t_stat = 0.0
        p_value = 1.0

    return {
        "mean_ic": round(mean_ic, 6),
        "std_ic": round(std_ic, 6),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "n_dates": n,
    }


# ---------------------------------------------------------------------------
# IC degradation
# ---------------------------------------------------------------------------


def _compute_ic_degradation(raw_ic: float, residual_ic: float) -> dict[str, Any]:
    """Compute the fraction of predictive power lost after factor neutralization.

    Uses absolute IC values so that negative signals are handled correctly
    (a negatively predictive signal inverted is still useful, and we measure
    whether factor neutralization reduces that predictive power regardless of
    sign).

    Degradation < 0 means the residual IC is *larger* in absolute terms than
    the raw IC — neutralization improved the signal (e.g., a contaminating
    factor was hurting raw IC).

    Args:
        raw_ic: Mean IC of the raw signal.
        residual_ic: Mean IC of the factor-neutralized signal.

    Returns:
        Dict with degradation_pct (float or None) and interpretable (bool).
    """
    if abs(raw_ic) < 1e-6:
        return {"degradation_pct": None, "interpretable": False}

    degradation = 1.0 - abs(residual_ic) / abs(raw_ic)
    return {
        "degradation_pct": round(degradation * 100, 1),
        "interpretable": True,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _compute_orthogonality_score(
    degradation_pct: float | None,
    residual_ic: float,
    mean_r2: float,
    p_value: float,
    config: OrthogonalityConfig,
) -> float:
    """Compute composite 0-100 orthogonality score.

    Weighting:
    - IC degradation: 35%  (lower degradation → higher score)
    - Residual IC magnitude: 25%
    - Factor model R²: 20%  (lower R² → more orthogonal → higher score)
    - Residual IC significance: 20%

    Args:
        degradation_pct: IC degradation in percentage points (can be None
            if raw IC was near zero).
        residual_ic: Mean IC of the factor-neutralized signal.
        mean_r2: Average cross-sectional R² of the factor regression.
        p_value: p-value of the residual IC t-test.
        config: Scoring configuration.

    Returns:
        Score in [0, 100].
    """
    # 1. IC degradation (35%)
    if degradation_pct is None:
        degradation_score = 40.0  # uncertain; give partial credit
    else:
        d = degradation_pct / 100.0
        if d <= 0.0:
            degradation_score = 100.0
        elif d <= 0.20:
            degradation_score = 90.0 + 10.0 * (0.20 - d) / 0.20
        elif d <= config.ic_degradation_warn:  # 0.40
            degradation_score = 60.0 + 30.0 * (config.ic_degradation_warn - d) / 0.20
        elif d <= config.ic_degradation_fail:  # 0.70
            degradation_score = 10.0 + 50.0 * (config.ic_degradation_fail - d) / 0.30
        else:
            degradation_score = max(0.0, 10.0 * (1.0 - d))

    # 2. Residual IC magnitude (25%)
    ic = abs(residual_ic)
    if ic >= 0.10:
        residual_ic_score = 100.0
    elif ic >= 0.05:
        residual_ic_score = 60.0 + 40.0 * (ic - 0.05) / 0.05
    elif ic >= 0.02:
        residual_ic_score = 20.0 + 40.0 * (ic - 0.02) / 0.03
    else:
        residual_ic_score = max(0.0, ic / 0.02 * 20.0)

    # 3. Factor model R² (20%) — lower R² means signal is more orthogonal
    if mean_r2 <= 0.05:
        r2_score = 100.0
    elif mean_r2 <= config.high_r2_threshold:  # 0.30
        r2_score = 100.0 - 50.0 * (mean_r2 - 0.05) / (config.high_r2_threshold - 0.05)
    elif mean_r2 <= 0.50:
        r2_score = 50.0 - 50.0 * (mean_r2 - config.high_r2_threshold) / (
            0.50 - config.high_r2_threshold
        )
    else:
        r2_score = max(0.0, (1.0 - mean_r2) * 10.0)

    # 4. Residual IC significance (20%)
    if p_value <= 0.01:
        sig_score = 100.0
    elif p_value <= 0.05:
        sig_score = 60.0 + 40.0 * (0.05 - p_value) / 0.04
    elif p_value <= 0.10:
        sig_score = 30.0 + 30.0 * (0.10 - p_value) / 0.05
    else:
        sig_score = max(0.0, (1.0 - p_value) * 30.0)

    composite = (
        0.35 * degradation_score + 0.25 * residual_ic_score + 0.20 * r2_score + 0.20 * sig_score
    )

    # Hard cap: severe degradation + insignificant residual IC → capped at 25
    if (
        degradation_pct is not None
        and degradation_pct > config.ic_degradation_fail * 100
        and p_value > 0.10
    ):
        composite = min(composite, 25.0)

    return round(min(100.0, max(0.0, composite)), 1)


def _compute_confidence(
    n_dates: int,
    n_factors_used: int,
    n_factors_configured: int,
    n_tickers: int,
    config: OrthogonalityConfig,
) -> float:
    """Compute confidence in the module's score.

    Low confidence when:
    - Fewer dates than min_dates
    - Only a subset of configured factors available
    - Few tickers in the cross-section

    Args:
        n_dates: Number of distinct evaluation dates.
        n_factors_used: Number of factors actually in the regression.
        n_factors_configured: Total factors in the config.
        n_tickers: Average cross-section size (rough proxy: total unique tickers).
        config: Configuration.

    Returns:
        Confidence in [0, 1].
    """
    # History confidence
    if n_dates < 3:
        time_conf = 0.1
    elif n_dates < config.min_dates:
        time_conf = 0.3 + 0.6 * (n_dates / config.min_dates)
    else:
        time_conf = 1.0

    # Factor completeness confidence
    factor_conf = n_factors_used / max(1, n_factors_configured) if n_factors_configured > 0 else 1.0

    # Cross-section confidence
    cs_conf = min(1.0, n_tickers / 50.0)

    confidence = 0.5 * time_conf + 0.3 * factor_conf + 0.2 * cs_conf
    return round(min(1.0, max(0.0, confidence)), 2)


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------


def _generate_narrative(
    raw_ic: float,
    residual_ic: float,
    degradation_pct: float | None,
    mean_r2: float,
    factor_loadings: dict[str, float],
    dominant_factor: str | None,
    residual_t_stat: float,
    residual_p_value: float,
    available_factors: list[str],
    warnings: list[str],
    config: OrthogonalityConfig,
) -> str:
    """Generate a PM-readable summary of orthogonality analysis.

    Args:
        raw_ic: Raw signal IC at target horizon.
        residual_ic: Factor-neutralized IC.
        degradation_pct: % IC drop after neutralization (None if not interpretable).
        mean_r2: Average cross-sectional R² of factor regression.
        factor_loadings: Average OLS/Ridge coefficient per factor.
        dominant_factor: Factor with highest absolute loading.
        residual_t_stat: t-statistic of residual IC.
        residual_p_value: p-value of residual IC.
        available_factors: Factors included in the regression.
        warnings: Warnings already generated upstream.
        config: Configuration.

    Returns:
        Multi-sentence plain-English narrative.
    """
    parts = []

    # Lead with the IC degradation finding
    if degradation_pct is None:
        parts.append(
            "The raw signal shows near-zero IC, making it difficult to assess "
            "how much orthogonal alpha remains after factor neutralization."
        )
    elif degradation_pct <= 10:
        parts.append(
            f"After neutralizing against {len(available_factors)} factor(s) "
            f"({', '.join(available_factors[:3])}{'...' if len(available_factors) > 3 else ''}), "
            f"the signal retains {100 - degradation_pct:.0f}% of its raw IC "
            f"(raw IC={raw_ic:.4f}, residual IC={residual_ic:.4f}), "
            "indicating highly orthogonal alpha with minimal factor contamination."
        )
    elif degradation_pct <= config.ic_degradation_warn * 100:
        parts.append(
            f"After factor neutralization, the signal retains "
            f"{100 - degradation_pct:.0f}% of its raw IC "
            f"(raw={raw_ic:.4f}, residual={residual_ic:.4f}). "
            "Most predictive power appears to be vendor-specific rather than factor-driven."
        )
    elif degradation_pct <= config.ic_degradation_fail * 100:
        parts.append(
            f"IC degrades by {degradation_pct:.0f}% after factor neutralization "
            f"(raw={raw_ic:.4f} → residual={residual_ic:.4f}), "
            "suggesting meaningful factor contamination that limits incremental value."
        )
    else:
        parts.append(
            f"IC collapses by {degradation_pct:.0f}% after factor neutralization "
            f"(raw={raw_ic:.4f} → residual={residual_ic:.4f}). "
            "This signal appears to be largely a repackaging of existing factor exposures."
        )

    # Factor R² insight
    if mean_r2 <= 0.10:
        parts.append(
            f"Known factors explain only {mean_r2:.1%} of the signal's cross-sectional "
            "variance, confirming a structurally distinct information source."
        )
    elif mean_r2 <= config.high_r2_threshold:
        parts.append(
            f"Factor exposures account for {mean_r2:.1%} of signal variance — "
            "moderate but not alarming contamination."
        )
    else:
        dominant_str = f" ({dominant_factor} is the primary driver)" if dominant_factor else ""
        parts.append(
            f"Factor exposures explain {mean_r2:.1%} of signal variance"
            f"{dominant_str}, which is above the {config.high_r2_threshold:.0%} threshold."
        )

    # Residual IC significance
    if residual_p_value <= 0.05:
        parts.append(
            f"The residual IC is statistically significant "
            f"(t={residual_t_stat:.2f}, p={residual_p_value:.3f}), "
            "providing statistical confidence in the orthogonal alpha."
        )
    elif residual_p_value <= 0.10:
        parts.append(
            f"The residual IC is marginally significant "
            f"(t={residual_t_stat:.2f}, p={residual_p_value:.3f}). "
            "More data would strengthen this conclusion."
        )
    else:
        parts.append(
            f"The residual IC does not reach statistical significance "
            f"(t={residual_t_stat:.2f}, p={residual_p_value:.3f}). "
            "Proceed with caution — the apparent alpha may not be robust."
        )

    return " ".join(parts)
