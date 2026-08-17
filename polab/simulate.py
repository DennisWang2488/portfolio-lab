"""Controlled simulation: returns with a factor structure, and signals of a
prescribed information coefficient.

Why simulate at all. Every empirical result in this project so far has been a
null, for a reason stated in `audits/prereg-spo-retest.md` §4: on one real path
the standard error of a Sharpe difference is 0.2–0.3, so effects the size of
the ones papers report are invisible. Simulation inverts that — **we set the
effect size**, so the design has power by construction, and the question stops
being "can anything beat 1/N" (it cannot, on one path) and becomes the question
institutions actually face: *given a signal of quality IC, what decision rule
extracts the most net-of-cost value?*

Calibration. Monthly moments are set to the sector-ETF panel this project
already uses: market vol ~4.2%/month (~14.5% annual), idiosyncratic ~3.0%/month,
betas ~1. `tests/test_polab.py` checks the simulated moments land in that range.

The one structural simplification, stated plainly: **equal betas**. With equal
betas and `1'w = 1`, the market term `w'ββ'w σ_m²` is constant over the simplex,
so mean–variance reduces *exactly* to the ℓ₂-regularized problem and the target
portfolio is a simplex projection — no QP in the inner loop. That is what makes
a five-dimensional sweep affordable. `stability.no_trade_band` covers the
general case for the smaller comparison grid.
"""

from __future__ import annotations

import numpy as np

# Monthly, calibrated to the sector-ETF panel (see module docstring).
MARKET_VOL = 0.042
IDIO_VOL = 0.030
MARKET_DRIFT = 0.007          # ~8.4%/yr equity risk premium
ANN_FACTOR_MONTHLY = 12


def simulate_returns(n_assets: int, n_periods: int, seed: int = 0,
                     market_vol: float = MARKET_VOL,
                     idio_vol: float = IDIO_VOL,
                     drift: float = MARKET_DRIFT) -> np.ndarray:
    """One-factor monthly returns, shape (n_periods, n_assets).

    r_it = drift + beta_i * f_t + e_it,  beta_i = 1 (see docstring).
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, market_vol, n_periods)
    e = rng.normal(0.0, idio_vol, (n_periods, n_assets))
    return drift + f[:, None] + e


def signal_with_ic(returns: np.ndarray, ic: float, seed: int = 0) -> np.ndarray:
    """Predictions whose CROSS-SECTIONAL correlation with the realized return is
    `ic`, period by period.

    Construction: standardize the realized cross-section to z, draw independent
    noise standardized to u, and form `ic*z + sqrt(1-ic^2)*u`. That combination
    has unit variance and correlation exactly `ic` with z in the population, so
    the realized per-period IC fluctuates around `ic` with sampling error — as a
    real signal does. Rescaled to the cross-section's own location and spread so
    the predictions live in return units.

    `ic = 0` gives a pure-noise signal: the null that any decision rule must not
    profit from.
    """
    if not 0.0 <= ic <= 1.0:
        raise ValueError(f"ic must be in [0, 1], got {ic}")
    rng = np.random.default_rng(seed + 10_000)  # offset so this isn't the same stream as simulate_returns
    R = np.asarray(returns, float)
    mu = R.mean(axis=1, keepdims=True)
    sd = R.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0

    z = (R - mu) / sd
    u = rng.normal(size=R.shape)
    u = (u - u.mean(axis=1, keepdims=True)) / np.where(
        u.std(axis=1, keepdims=True) < 1e-12, 1.0, u.std(axis=1, keepdims=True))
    mixed = ic * z + np.sqrt(max(0.0, 1.0 - ic**2)) * u
    return mu + sd * mixed


def realized_ic(signal: np.ndarray, returns: np.ndarray) -> float:
    """Mean per-period cross-sectional correlation — the achieved IC."""
    S, R = np.asarray(signal, float), np.asarray(returns, float)
    out = []
    for s, r in zip(S, R):
        if s.std() > 1e-12 and r.std() > 1e-12:
            out.append(np.corrcoef(s, r)[0, 1])
    return float(np.mean(out)) if out else np.nan


def grinold_ir(ic: float, breadth: int) -> float:
    """Fundamental Law of Active Management: IR ~= IC * sqrt(breadth).

    Used as a CALIBRATION CHECK on the simulator, not as a claim: a correctly
    built signal-plus-unconstrained-portfolio pipeline should track this before
    costs and constraints bite. Long-only and turnover control both push the
    achieved IR below it.
    """
    return ic * np.sqrt(breadth)
