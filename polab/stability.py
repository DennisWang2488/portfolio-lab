"""Decision rules parameterized by how STABLE the decision is over time.

The thesis this module exists to test. Across both replications in this project
(iterations 2, 6, 7) survival net of costs was determined by turnover, not by
prediction quality: E2E-DRO's variants ranked by turnover, and our SPO+/PtO —
whose layer is `argmax r̂ᵀw` over the simplex — turned over 100% of the book
every month and finished last after costs despite topping the table gross.

Turnover is the temporal instability of the decision map: how much the decision
moves when the inputs move. Stability is also what generalization bounds are
written in (Bousquet & Elisseeff 2002). So the empirical regularity "high
turnover → poor out-of-sample" has a statistical reading, not only an accounting
one, and it predicts something testable: **the weaker the signal, the more
stability should be imposed.** `scripts/run_ic_breadth_stability.py` tests it.

Two mechanisms, because they are not the same thing and practitioners use both:

- `blend` — partial rebalancing, `w = (1-a)·w_drifted + a·w_target`. Always
  trades, but only a fraction of the way. Closed form, so the big sweep is cheap.
- `no_trade_band` — an ℓ₁ penalty on trading, which produces a genuine *no-trade
  region*: small edges are not acted on at all. Needs a solver, so it is used on
  the smaller comparison grid.

Classically the no-trade region is justified by transaction costs (Davis &
Norman 1990; Gârleanu & Pedersen 2013). The point of interest here is whether
estimation error alone justifies one — i.e. whether the optimal band is wider
than costs alone imply, and widens as IC falls.
"""

from __future__ import annotations

import numpy as np

from .spo import project_simplex


def target_weights(r_hat: np.ndarray, concentration: float = 1.0) -> np.ndarray:
    """The unconstrained-by-history target: `argmax_w r̂ᵀw − λ‖w‖²` on the simplex.

    With equal betas this IS the mean–variance solution (see `simulate` docstring),
    and it equals `proj_simplex(r̂ / 2λ)`.

    λ is set per period from the cross-sectional spread of `r̂`:
    `λ = concentration · σ(r̂) · n / 2`. That holds the TARGET's concentration
    roughly fixed as IC varies — without it, a stronger signal would mechanically
    produce a more concentrated portfolio and confound signal quality with
    concentration, which is exactly the confound that sank the SPO+ comparison.
    """
    r = np.asarray(r_hat, float)
    n = len(r)
    sigma = r.std()
    if sigma < 1e-12:
        return np.full(n, 1.0 / n)
    lam = concentration * sigma * n / 2.0
    return project_simplex(r / (2.0 * lam))


def blend(w_prev: np.ndarray, w_target: np.ndarray, alpha: float) -> np.ndarray:
    """Partial rebalancing. `alpha=1` trades all the way, `alpha→0` never trades."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    w = (1.0 - alpha) * np.asarray(w_prev, float) + alpha * np.asarray(w_target, float)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


def no_trade_band(r_hat: np.ndarray, w_prev: np.ndarray, tau: float,
                  concentration: float = 1.0) -> np.ndarray:
    """`argmax_w r̂ᵀw − λ‖w‖² − τ‖w − w_prev‖₁` on the simplex.

    The ℓ₁ term makes not-trading optimal for small edges: a genuine no-trade
    region rather than a scaled-down trade. `tau=0` recovers `target_weights`.
    """
    import cvxpy as cp  # local: only this path needs a solver

    r = np.asarray(r_hat, float)
    n = len(r)
    sigma = r.std()
    if sigma < 1e-12:
        return np.asarray(w_prev, float)
    lam = concentration * sigma * n / 2.0

    w = cp.Variable(n)
    obj = cp.Maximize(r @ w - lam * cp.sum_squares(w)
                      - tau * cp.norm1(w - np.asarray(w_prev, float)))
    prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0])
    prob.solve()
    if w.value is None:
        return np.asarray(w_prev, float)
    v = np.maximum(np.asarray(w.value, float), 0.0)
    s = v.sum()
    return v / s if s > 0 else np.full(n, 1.0 / n)


def run_path(returns: np.ndarray, signal: np.ndarray, alpha: float,
             cost_bps: float = 50.0, concentration: float = 1.0,
             tau: float | None = None) -> dict:
    """Walk one simulated path. Returns gross/net series and turnover.

    `signal[t]` is the prediction for period `t`, acted on at the START of `t`;
    `returns[t]` is what then happens. Causality is by construction here, unlike
    the real-data path where `run_spo_retest.training_slice` has to enforce it.

    Weights drift with returns between decisions, and cost is charged on
    `‖w_new − w_drifted‖₁` — the same convention as `backtest.cost_overlay`, so
    simulated and real numbers are comparable.

    Returns `traded` as well as `net`, because the blend rule does not consult
    the cost when deciding: net at ANY cost level is `gross - rate*traded` on
    the same path. That makes the cost axis free to sweep instead of requiring
    a re-run per level — and cost turns out to be the axis that decides
    everything.
    """
    R, S = np.asarray(returns, float), np.asarray(signal, float)
    T, n = R.shape
    rate = cost_bps / 1e4
    w = np.full(n, 1.0 / n)
    gross = np.zeros(T)
    net = np.zeros(T)
    traded_arr = np.zeros(T)

    for t in range(T):
        w_target = target_weights(S[t], concentration)
        if tau is None:
            w_new = blend(w, w_target, alpha)
        else:
            w_new = no_trade_band(S[t], w, tau, concentration)
        traded = np.abs(w_new - w).sum()
        traded_arr[t] = traded
        gross[t] = float(w_new @ R[t])
        net[t] = gross[t] - rate * traded
        # drift to the start of the next period
        grown = w_new * (1.0 + R[t])
        tot = grown.sum()
        w = grown / tot if tot > 0 else np.full(n, 1.0 / n)

    return {"gross": gross, "net": net, "traded": traded_arr,
            "turnover": traded_arr / 2.0}


def equal_weight_path(returns: np.ndarray, cost_bps: float = 50.0) -> dict:
    """1/N held with drift and rebalanced each period — the null, same engine."""
    T, n = returns.shape
    return run_path(returns, np.zeros((T, n)), alpha=1.0, cost_bps=cost_bps)
