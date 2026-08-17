"""Classical portfolio-construction baselines.

Every baseline has the same signature: it receives a lookback window of past
per-period returns (DataFrame, rows = time, columns = assets) and returns a
long-only, fully-invested weight vector (numpy array summing to 1).

These are the null hypotheses any fancier method must beat out of sample —
in particular equal_weight (DeMiguel, Garlappi & Uppal 2009).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


def _sample_cov(window: pd.DataFrame, shrinkage: float = 0.1) -> np.ndarray:
    """Sample covariance with simple shrinkage toward the scaled identity.

    Not actual Ledoit-Wolf — 0.1 is just a number that keeps the QP from
    falling over on a 104-week window. Fine for a baseline.
    """
    S = np.asarray(window.cov(ddof=1))
    n = S.shape[0]
    target = np.trace(S) / n * np.eye(n)
    return (1.0 - shrinkage) * S + shrinkage * target


def equal_weight(window: pd.DataFrame) -> np.ndarray:
    n = window.shape[1]
    return np.full(n, 1.0 / n)  # ignores the window, that's the point


def min_variance(window: pd.DataFrame, shrinkage: float = 0.1) -> np.ndarray:
    S = _sample_cov(window, shrinkage)
    n = S.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S))),
                      [cp.sum(w) == 1, w >= 0])
    prob.solve()
    if w.value is None:
        return equal_weight(window)
    return _clean(w.value)


def max_sharpe(window: pd.DataFrame, shrinkage: float = 0.1) -> np.ndarray:
    """Long-only maximum-Sharpe portfolio via the standard convex reformulation:
    min y'Σy  s.t.  μ'y = 1, y >= 0, then w = y / sum(y).

    Requires at least one asset with positive estimated mean; otherwise falls
    back to min_variance (the tangency portfolio does not exist long-only).
    """
    mu = window.mean().to_numpy()
    if np.all(mu <= 0):
        return min_variance(window, shrinkage)
    S = _sample_cov(window, shrinkage)
    n = S.shape[0]
    y = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(S))),
                      [mu @ y == 1, y >= 0])
    prob.solve()
    if y.value is None or y.value.sum() <= 1e-12:
        return min_variance(window, shrinkage)
    return _clean(y.value / y.value.sum())


def risk_parity(window: pd.DataFrame, shrinkage: float = 0.1) -> np.ndarray:
    """Equal-risk-contribution portfolio via the convex formulation of
    Spinu (2013): min 0.5 w'Σw - (1/n) Σ log(w_i), then normalize.
    """
    S = _sample_cov(window, shrinkage)
    n = S.shape[0]
    w = cp.Variable(n, pos=True)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(S))
                                  - cp.sum(cp.log(w)) / n))
    prob.solve()
    if w.value is None:
        return equal_weight(window)
    return _clean(w.value / w.value.sum())


def _clean(w: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Clip solver noise, renormalize to the simplex."""
    w = np.asarray(w, dtype=float)
    w[w < tol] = 0.0
    s = w.sum()
    if s <= 0:
        return np.full(len(w), 1.0 / len(w))
    return w / s


BASELINES = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
    "risk_parity": risk_parity,
}
