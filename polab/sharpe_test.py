"""Tests for the difference between two Sharpe ratios on paired return series.

The gap this closes: `metrics.py` scores each strategy on its own (PSR, DSR),
but the claim under test here is always comparative -- "DFL beats 1/N", "E2E
beats predict-then-optimize". A ranking is not evidence that the gap is real.
One OOS path of 454 weekly observations supports far less than the headline
Sharpe differences suggest.

Three procedures, in increasing order of how little they assume:

1. `jobson_korkie` -- Jobson & Korkie (1981) with the Memmel (2003) correction.
   Closed form, but assumes i.i.d. normal returns. Reported only as the
   textbook reference point; weekly equity returns are neither.
2. `ledoit_wolf` -- Ledoit & Wolf (2008) HAC-robust delta method. Drops
   normality and independence: allows serial correlation and fat tails. This
   is the number to quote.
3. `ledoit_wolf_bootstrap` -- their studentized circular block bootstrap,
   which is what LW actually recommend at these sample sizes; the asymptotic
   version over-rejects when T is a few hundred.

All three test H0: SR(a) == SR(b) against a two-sided alternative, on the SAME
dates (paired). Annualization scales the difference and its standard error by
sqrt(ann_factor) and therefore leaves every t-statistic and p-value unchanged.

References
----------
Jobson & Korkie (1981), J. Finance 36(4).
Memmel (2003), Finance Letters 1.
Ledoit & Wolf (2008), "Robust performance hypothesis testing with the Sharpe
ratio", J. Empirical Finance 15(5) -- equations (4)-(9) for the delta method
and section 3.1 for the bootstrap.
Andrews (1991), Econometrica 59(3) -- automatic bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SharpeDiff:
    """Result of one paired Sharpe-difference test (all values annualized)."""

    name_a: str
    name_b: str
    sharpe_a: float
    sharpe_b: float
    diff: float                 # sharpe_a - sharpe_b
    std_error: float            # standard error of `diff`
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n_periods: int
    method: str

    @property
    def significant(self) -> bool:
        """Two-sided rejection at the 5% level."""
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    def as_dict(self) -> dict:
        d = {f: getattr(self, f) for f in self.__dataclass_fields__}
        d["significant_5pct"] = self.significant
        return d


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

def _align(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Restrict both series to their common dates, dropping any NaN row.

    Pairing is the entire point -- the two strategies share market shocks, and
    an unpaired test throws away the covariance that makes the comparison
    informative.
    """
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 8:
        raise ValueError(f"need at least 8 paired observations, got {len(df)}")
    return df["a"].to_numpy(float), df["b"].to_numpy(float)


def _is_degenerate(x: np.ndarray, y: np.ndarray) -> bool:
    """True when the two series are the same series to numerical precision.

    Not hypothetical here: our 1/N reproduces their `ew_net` to 1e-17, so that
    pair appears in every benchmark sweep. The difference is identically zero
    and its standard error is zero, which is a degenerate input to every test
    below, not a failure of them.
    """
    scale = max(float(np.abs(x).max()), float(np.abs(y).max()), 1e-300)
    return bool(np.abs(x - y).max() < 1e-12 * scale)


def _degenerate_result(x: np.ndarray, ann_factor: int, name_a: str,
                       name_b: str, method: str) -> SharpeDiff:
    mu, g = float(x.mean()), float((x**2).mean())
    sr = mu / np.sqrt(g - mu**2) * np.sqrt(ann_factor)
    return SharpeDiff(name_a=name_a, name_b=name_b, sharpe_a=sr, sharpe_b=sr,
                      diff=0.0, std_error=0.0, t_stat=0.0, p_value=1.0,
                      ci_low=0.0, ci_high=0.0, n_periods=len(x),
                      method=method + "_identical_series")


def _moments(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """(mu_a, mu_b, gamma_a, gamma_b) -- LW parameterize by the raw second
    moment gamma = E[r^2], not the variance, so the gradient below is theirs."""
    return float(x.mean()), float(y.mean()), float((x**2).mean()), float((y**2).mean())


def _sharpe_diff(mu_a: float, mu_b: float, g_a: float, g_b: float) -> float:
    va, vb = g_a - mu_a**2, g_b - mu_b**2
    if va <= 0 or vb <= 0:
        return np.nan
    return mu_a / np.sqrt(va) - mu_b / np.sqrt(vb)


def _gradient(mu_a: float, mu_b: float, g_a: float, g_b: float) -> np.ndarray:
    """d(Delta)/d(mu_a, mu_b, gamma_a, gamma_b) -- LW (2008) eq. (8)."""
    va, vb = g_a - mu_a**2, g_b - mu_b**2
    return np.array([
        g_a / va**1.5,
        -g_b / vb**1.5,
        -mu_a / (2.0 * va**1.5),
        mu_b / (2.0 * vb**1.5),
    ])


def _parzen(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    out = np.zeros_like(ax)
    m = ax <= 0.5
    out[m] = 1.0 - 6.0 * ax[m] ** 2 + 6.0 * ax[m] ** 3
    m = (ax > 0.5) & (ax <= 1.0)
    out[m] = 2.0 * (1.0 - ax[m]) ** 3
    return out


def _andrews_bandwidth(Z: np.ndarray) -> float:
    """Andrews (1991) automatic bandwidth for the Parzen kernel, via univariate
    AR(1) approximations of each column (equal weights)."""
    T, k = Z.shape
    num = den = 0.0
    for j in range(k):
        z = Z[:, j]
        denom = float((z[:-1] ** 2).sum())
        rho = float((z[:-1] * z[1:]).sum() / denom) if denom > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))
        resid = z[1:] - rho * z[:-1]
        s2 = float((resid**2).mean())
        num += 4.0 * rho**2 * s2**2 / (1.0 - rho) ** 8
        den += s2**2 / (1.0 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    if alpha2 <= 0:
        return 1.0
    return float(2.6614 * (alpha2 * T) ** 0.2)


def _hac_cov(Y: np.ndarray, prewhite: bool = True) -> np.ndarray:
    """HAC (Parzen kernel, Andrews bandwidth, VAR(1) prewhitening) estimate of
    the long-run covariance of the demeaned rows of `Y`.

    Prewhitening is what makes this usable on persistent series: the kernel
    estimator is applied to the VAR(1) residuals and the result is recolored,
    which removes most of the bias the kernel would otherwise carry.
    """
    T, k = Y.shape
    Z = Y - Y.mean(axis=0)
    A = np.zeros((k, k))
    if prewhite and T > 2 * k + 2:
        X, Ynext = Z[:-1], Z[1:]
        A = np.linalg.lstsq(X, Ynext, rcond=None)[0].T
        spec = float(np.max(np.abs(np.linalg.eigvals(A))))
        if spec > 0.97:          # keep the recoloring matrix invertible
            A *= 0.97 / spec
        Z = Ynext - X @ A.T

    n = Z.shape[0]
    S = _andrews_bandwidth(Z)
    Psi = (Z.T @ Z) / n
    for lag in range(1, n):
        w = float(_parzen(np.array([lag / S]))[0])
        if w == 0.0:
            break
        G = (Z[lag:].T @ Z[:-lag]) / n
        Psi += w * (G + G.T)

    if prewhite:
        M = np.linalg.inv(np.eye(k) - A)
        Psi = M @ Psi @ M.T
    return Psi


def _lw_stat(x: np.ndarray, y: np.ndarray,
             prewhite: bool = True) -> tuple[float, float]:
    """(per-period Sharpe difference, its HAC standard error)."""
    T = len(x)
    mu_a, mu_b, g_a, g_b = _moments(x, y)
    delta = _sharpe_diff(mu_a, mu_b, g_a, g_b)
    if not np.isfinite(delta):
        return np.nan, np.nan
    grad = _gradient(mu_a, mu_b, g_a, g_b)
    Y = np.column_stack([x, y, x**2, y**2])
    Psi = _hac_cov(Y, prewhite=prewhite)
    var = float(grad @ Psi @ grad) / T
    return delta, float(np.sqrt(var)) if var > 0 else np.nan


# --------------------------------------------------------------------------
# the three tests
# --------------------------------------------------------------------------

def jobson_korkie(a: pd.Series, b: pd.Series, ann_factor: int,
                  name_a: str = "a", name_b: str = "b") -> SharpeDiff:
    """Jobson-Korkie / Memmel test. Assumes i.i.d. normal returns.

    Kept for comparison only: on weekly equity returns the normality assumption
    is false in the direction that matters (fat tails inflate significance), so
    a JK p-value smaller than the `ledoit_wolf` one is expected, not reassuring.
    """
    x, y = _align(a, b)
    if _is_degenerate(x, y):
        return _degenerate_result(x, ann_factor, name_a, name_b,
                                  "jobson_korkie_memmel")
    T = len(x)
    mu_a, mu_b = x.mean(), y.mean()
    sa, sb = x.std(ddof=1), y.std(ddof=1)
    sab = float(np.cov(x, y, ddof=1)[0, 1])

    theta = (2.0 * sa**2 * sb**2 - 2.0 * sa * sb * sab
             + 0.5 * mu_a**2 * sb**2 + 0.5 * mu_b**2 * sa**2
             - (mu_a * mu_b) / (sa * sb) * sab**2) / T
    sr_a, sr_b = mu_a / sa, mu_b / sb
    diff = sr_a - sr_b
    # the statistic is built on (sb*mu_a - sa*mu_b); rescale its SE onto `diff`
    se = float(np.sqrt(theta)) / (sa * sb) if theta > 0 else np.nan
    return _finish(diff, se, sr_a, sr_b, ann_factor, T,
                   name_a, name_b, "jobson_korkie_memmel")


def ledoit_wolf(a: pd.Series, b: pd.Series, ann_factor: int,
                name_a: str = "a", name_b: str = "b",
                prewhite: bool = True) -> SharpeDiff:
    """Ledoit-Wolf (2008) HAC delta-method test. Robust to serial correlation
    and to fat tails; asymptotic, so read it alongside the bootstrap below."""
    x, y = _align(a, b)
    if _is_degenerate(x, y):
        return _degenerate_result(x, ann_factor, name_a, name_b, "ledoit_wolf_hac")
    delta, se = _lw_stat(x, y, prewhite=prewhite)
    mu_a, mu_b, g_a, g_b = _moments(x, y)
    sr_a = mu_a / np.sqrt(g_a - mu_a**2)
    sr_b = mu_b / np.sqrt(g_b - mu_b**2)
    return _finish(delta, se, sr_a, sr_b, ann_factor, len(x),
                   name_a, name_b, "ledoit_wolf_hac")


def ledoit_wolf_bootstrap(a: pd.Series, b: pd.Series, ann_factor: int,
                          name_a: str = "a", name_b: str = "b",
                          block_size: int = 5, n_boot: int = 4999,
                          seed: int = 0) -> SharpeDiff:
    """Studentized circular block bootstrap (Ledoit & Wolf 2008, sec. 3.1).

    Resamples whole blocks of *paired* observations, so the cross-strategy
    covariance and within-strategy dependence both survive resampling. The
    studentization (dividing each bootstrap difference by its own HAC standard
    error) is what gives the interval its accuracy at T in the hundreds.

    `block_size` is fixed rather than calibrated -- LW's calibration procedure
    is a bootstrap-within-bootstrap and is not worth its cost here. 5 weeks is
    a deliberate, reportable choice; `scripts/sharpe_tests.py` reports the
    sensitivity to it.
    """
    x, y = _align(a, b)
    if _is_degenerate(x, y):
        return _degenerate_result(x, ann_factor, name_a, name_b,
                                  f"lw_bootstrap_b{block_size}")
    T = len(x)
    delta, se = _lw_stat(x, y)
    if not np.isfinite(se):
        return _finish(delta, np.nan, np.nan, np.nan, ann_factor, T,
                       name_a, name_b, f"lw_bootstrap_b{block_size}")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_size))
    offsets = np.arange(block_size)
    t_boot = np.empty(n_boot)
    n_bad = 0
    for i in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % T).ravel()[:T]
        d_i, se_i = _lw_stat(x[idx], y[idx])
        if np.isfinite(d_i) and np.isfinite(se_i) and se_i > 0:
            t_boot[i] = (d_i - delta) / se_i
        else:
            t_boot[i] = np.nan
            n_bad += 1
    t_boot = t_boot[np.isfinite(t_boot)]
    if len(t_boot) < n_boot // 2:
        raise RuntimeError("bootstrap degenerate: over half the resamples failed")

    t_obs = delta / se
    p = float((np.abs(t_boot) >= abs(t_obs)).mean())
    q_lo, q_hi = np.quantile(t_boot, [0.025, 0.975])
    scale = np.sqrt(ann_factor)
    mu_a, mu_b, g_a, g_b = _moments(x, y)
    return SharpeDiff(
        name_a=name_a, name_b=name_b,
        sharpe_a=float(mu_a / np.sqrt(g_a - mu_a**2) * scale),
        sharpe_b=float(mu_b / np.sqrt(g_b - mu_b**2) * scale),
        diff=float(delta * scale),
        std_error=float(se * scale),
        t_stat=float(t_obs),
        p_value=p,
        # percentile-t: the interval is inverted, hence the crossed quantiles
        ci_low=float((delta - q_hi * se) * scale),
        ci_high=float((delta - q_lo * se) * scale),
        n_periods=T,
        method=f"lw_bootstrap_b{block_size}_n{len(t_boot)}",
    )


def _finish(diff: float, se: float, sr_a: float, sr_b: float,
            ann_factor: int, T: int, name_a: str, name_b: str,
            method: str) -> SharpeDiff:
    """Annualize a (difference, standard error) pair into a normal-theory result."""
    scale = np.sqrt(ann_factor)
    t = diff / se if (np.isfinite(se) and se > 0) else np.nan
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan
    z = stats.norm.ppf(0.975)
    return SharpeDiff(
        name_a=name_a, name_b=name_b,
        sharpe_a=float(sr_a * scale), sharpe_b=float(sr_b * scale),
        diff=float(diff * scale), std_error=float(se * scale),
        t_stat=float(t), p_value=p,
        ci_low=float((diff - z * se) * scale),
        ci_high=float((diff + z * se) * scale),
        n_periods=T, method=method,
    )


def compare_to_benchmark(returns: pd.DataFrame, benchmark: str, ann_factor: int,
                         method: str = "bootstrap", **kwargs) -> pd.DataFrame:
    """Test every column of `returns` against one benchmark column.

    No multiple-testing adjustment is applied here -- these p-values are
    marginal. Use `metrics.deflated_sharpe` for the selection-bias side; the
    two answer different questions and neither substitutes for the other.
    """
    fns = {"jk": jobson_korkie, "lw": ledoit_wolf,
           "bootstrap": ledoit_wolf_bootstrap}
    if method not in fns:
        raise ValueError(f"method must be one of {sorted(fns)}, got {method!r}")
    fn = fns[method]
    rows = [fn(returns[c], returns[benchmark], ann_factor,
               name_a=c, name_b=benchmark, **kwargs).as_dict()
            for c in returns.columns if c != benchmark]
    return (pd.DataFrame(rows).set_index("name_a")
            .sort_values("diff", ascending=False))
