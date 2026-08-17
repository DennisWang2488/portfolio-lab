"""Performance metrics with an emphasis on backtest honesty.

All Sharpe-related quantities are computed on per-period returns; annualization
is explicit via `ann_factor` (52 = weekly, 252 = daily, 12 = monthly).

The probabilistic / deflated Sharpe ratios follow Bailey & Lopez de Prado,
"The Sharpe Ratio Efficient Frontier" (2012) and "The Deflated Sharpe Ratio"
(2014).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329  # Bailey-LdP expected-max-Sharpe formula needs this


def ann_return(returns: pd.Series, ann_factor: int) -> float:
    """Annualized geometric return."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    total = float((1.0 + r).prod())
    return total ** (ann_factor / len(r)) - 1.0


def ann_vol(returns: pd.Series, ann_factor: int) -> float:
    return float(returns.dropna().std(ddof=1)) * np.sqrt(ann_factor)


def sharpe(returns: pd.Series, ann_factor: int, rf_per_period: float = 0.0) -> float:
    """Annualized Sharpe ratio on excess returns."""
    r = returns.dropna() - rf_per_period
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(r.mean() / sd) * np.sqrt(ann_factor)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the compounded wealth curve (negative number)."""
    wealth = (1.0 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def avg_turnover(turnover: pd.Series) -> float:
    """Average one-way turnover per rebalance (sum |trade| / 2)."""
    t = turnover.dropna()
    return float(t.mean()) if len(t) else np.nan


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true (per-period) Sharpe exceeds `sr_benchmark`.

    `sr_benchmark` is a per-period (NOT annualized) Sharpe. Accounts for sample
    length, skewness, and kurtosis of the return series.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 4 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # non-excess kurtosis
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return np.nan
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max per-period SR] across `n_trials` independent zero-skill trials.

    `var_sharpe` is the cross-trial variance of the estimated per-period Sharpe.
    This is the deflation benchmark SR* of the Deflated Sharpe Ratio.
    """
    if n_trials < 2 or var_sharpe <= 0:
        return 0.0
    e = np.exp(1.0)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(var_sharpe) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float]) -> float:
    """DSR: PSR against the expected-max Sharpe of the strategies actually tried.

    `trial_sharpes` are the per-period Sharpe estimates of ALL strategy variants
    evaluated during the research process (including the reported one). Guards
    against selection bias from multiple testing.
    """
    trials = [s for s in trial_sharpes if np.isfinite(s)]
    if len(trials) < 2:
        return probabilistic_sharpe(returns, 0.0)
    sr_star = expected_max_sharpe(len(trials), float(np.var(trials, ddof=1)))
    return probabilistic_sharpe(returns, sr_star)


def summary(returns: pd.Series, ann_factor: int,
            turnover: pd.Series | None = None) -> dict:
    """Standard metric block for one strategy's net return series."""
    out = {
        "ann_return": ann_return(returns, ann_factor),
        "ann_vol": ann_vol(returns, ann_factor),
        "sharpe": sharpe(returns, ann_factor),
        "max_drawdown": max_drawdown(returns),
        "psr_vs_zero": probabilistic_sharpe(returns, 0.0),
        "n_periods": int(returns.dropna().shape[0]),
    }
    if turnover is not None:
        out["avg_turnover"] = avg_turnover(turnover)
    return out
