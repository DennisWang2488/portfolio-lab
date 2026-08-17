"""Smart Predict-then-Optimize (SPO+) on a max-return portfolio layer.

Implements the decision layer and surrogate loss of arXiv:2601.04062 so we can
test its claim ourselves. Written against Elmachtoub & Grigas (2022), not
against the paper's printed formula -- see `Note on eq. (4)` below.

The decision layer maximizes a linear return term plus a concave, prediction-
independent penalty:

    g(c) = max_{w in W} { c'w + h(w) },     W = {w >= 0, 1'w = 1}
    h(w) = -gamma*||w - w_prev||_1 - lam*||w||_2^2

With `h = 0` this is a linear program over the simplex, so the solution is a
vertex: all capital in the single asset with the highest predicted return. That
degeneracy is not our modelling choice -- it is what `argmax r'w` means on a
simplex, and it is the layer both SPO+ and the paper's "PtO Markowitz" baseline
use. It is also why `polab/audits/prereg-spo-retest.md` puts these strategies in
the low-correlation, high-standard-error row of the power table.

Note on eq. (4)
---------------
The paper prints the SPO+ loss as `max_{w in W} (2r_hat - r)'w - r'w*`. That is
not the SPO+ loss. The correct max-form surrogate is

    L(r_hat, r) = g(2*r_hat - r) - 2*r_hat'w* + r'w* - h(w*)

The two agree at `r_hat = r` (both are 0), which makes the discrepancy easy to
miss, but they differ by `2(r_hat - r)'w*`, which *depends on r_hat*. The
printed version therefore has gradient `2*w_tilde` instead of the correct
`2*(w_tilde - w*)` -- a different algorithm. The paper uses PyEPO, which
implements SPO+ correctly, so this is almost certainly a typesetting error; we
note it because anyone implementing from the paper alone would not get SPO+.

Derivation of the `h != 0` case (the paper asserts gradients still flow but does
not show the bound survives): with `w_hat` optimal for `r_hat` and `w*` optimal
for `r`,

    g(2r_hat - r) >= (2r_hat - r)'w_hat + h(w_hat)                [feasibility]
                   = [2r_hat'w_hat + 2h(w_hat)] - r'w_hat - h(w_hat)
                  >= [2r_hat'w*     + 2h(w*)  ] - r'w_hat - h(w_hat)  [optimality]

so `L >= g(r) - (r'w_hat + h(w_hat))` = regret. `test_spo_plus_upper_bounds_regret`
checks this numerically, including with `h != 0`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# decision layer
# --------------------------------------------------------------------------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, 1'w = 1} (Duchi et al. 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def decide(c: np.ndarray, lam: float = 0.0, gamma: float = 0.0,
           w_prev: np.ndarray | None = None) -> np.ndarray:
    """argmax_{w in W} { c'w - gamma*||w - w_prev||_1 - lam*||w||^2 }.

    Closed form in the two cases we actually use:
      * `lam = gamma = 0` -> a vertex of the simplex (ties broken by lowest index)
      * `lam > 0, gamma = 0` -> `proj_simplex(c / (2*lam))`, because
        `c'w - lam||w||^2 = -lam(||w - c/(2 lam)||^2 - ||c/(2 lam)||^2)`

    The `gamma > 0` case is a small LP and is solved with scipy.
    """
    c = np.asarray(c, dtype=float)
    if gamma > 0.0:
        return _decide_with_fee(c, lam, gamma, w_prev)
    if lam > 0.0:
        return project_simplex(c / (2.0 * lam))
    w = np.zeros_like(c)
    w[int(np.argmax(c))] = 1.0
    return w


def _preferred_solver(cp) -> str | None:
    """First installed solver from our preference order, or None for cvxpy's default.

    cvxpy dropped ECOS from its default install in 1.6, so pinning it by name
    makes a fresh clone fail on the first run. Preference order is
    accuracy-for-this-problem first, availability second; falling through to
    None lets cvxpy pick, which is always better than raising.
    """
    for name in ("ECOS", "CLARABEL", "SCS", "OSQP"):
        if name in cp.installed_solvers():
            return name
    return None


def _decide_with_fee(c: np.ndarray, lam: float, gamma: float,
                     w_prev: np.ndarray | None) -> np.ndarray:
    """Transaction-fee variant. Requires cvxpy (already a project dependency)."""
    import cvxpy as cp

    n = len(c)
    prev = np.zeros(n) if w_prev is None else np.asarray(w_prev, float)
    w = cp.Variable(n)
    obj = c @ w - gamma * cp.norm1(w - prev)
    if lam > 0.0:
        obj = obj - lam * cp.sum_squares(w)
    prob = cp.Problem(cp.Maximize(obj), [w >= 0, cp.sum(w) == 1])
    solver = _preferred_solver(cp)
    prob.solve() if solver is None else prob.solve(solver=solver)
    if w.value is None:
        raise RuntimeError("decision layer failed to solve")
    return project_simplex(np.asarray(w.value).ravel())


def layer_value(c: np.ndarray, w: np.ndarray, lam: float = 0.0,
                gamma: float = 0.0, w_prev: np.ndarray | None = None) -> float:
    """The layer's objective `c'w + h(w)` evaluated at a given `w`."""
    val = float(c @ w)
    if lam > 0.0:
        val -= lam * float(w @ w)
    if gamma > 0.0:
        prev = np.zeros_like(w) if w_prev is None else w_prev
        val -= gamma * float(np.abs(w - prev).sum())
    return val


def _penalty(w: np.ndarray, lam: float, gamma: float,
             w_prev: np.ndarray | None) -> float:
    """h(w) alone."""
    return layer_value(np.zeros_like(w), w, lam, gamma, w_prev)


# --------------------------------------------------------------------------
# SPO+ surrogate
# --------------------------------------------------------------------------

def decision_regret(r_hat: np.ndarray, r: np.ndarray, lam: float = 0.0,
                    gamma: float = 0.0,
                    w_prev: np.ndarray | None = None) -> float:
    """The quantity SPO+ upper-bounds: value lost by deciding on `r_hat`."""
    w_star = decide(r, lam, gamma, w_prev)
    w_hat = decide(r_hat, lam, gamma, w_prev)
    return (layer_value(r, w_star, lam, gamma, w_prev)
            - layer_value(r, w_hat, lam, gamma, w_prev))


def spo_plus(r_hat: np.ndarray, r: np.ndarray, lam: float = 0.0,
             gamma: float = 0.0, w_prev: np.ndarray | None = None
             ) -> tuple[float, np.ndarray]:
    """SPO+ loss and its subgradient with respect to `r_hat`.

    Returns `(loss, grad)` where `grad = 2*(w_tilde - w_star)`.
    """
    w_star = decide(r, lam, gamma, w_prev)
    w_tilde = decide(2.0 * r_hat - r, lam, gamma, w_prev)
    loss = (layer_value(2.0 * r_hat - r, w_tilde, lam, gamma, w_prev)
            - 2.0 * float(r_hat @ w_star)
            + float(r @ w_star)
            - _penalty(w_star, lam, gamma, w_prev))
    return loss, 2.0 * (w_tilde - w_star)


# --------------------------------------------------------------------------
# linear predictors
# --------------------------------------------------------------------------

@dataclass
class LinearPredictor:
    """r_hat = ((x - x_mean)/x_std) @ B + b.

    Standardization is part of the model, fitted on the training window only, so
    a predictor carries its own scaler and cannot be applied to a later window
    using statistics computed from that window. Without it the useful learning
    rate would depend on the units of whichever feature happened to be largest.
    """

    B: np.ndarray               # (d, n)
    b: np.ndarray               # (n,)
    x_mean: np.ndarray          # (d,)
    x_std: np.ndarray           # (d,)
    loss_history: list = None

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.x_mean) / self.x_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.standardize(X) @ self.B + self.b


def _scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm = X.mean(0)
    xs = X.std(0)
    xs[xs < 1e-12] = 1.0        # a constant feature contributes nothing, not NaN
    return xm, xs


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-4) -> LinearPredictor:
    """The PtO baseline: minimize ||XB + b - Y||^2 (their eq. 16), closed form.

    Ridge rather than plain OLS because the design matrix is rank-deficient
    whenever a training window is shorter than the feature count; `alpha` is
    fixed in advance (pre-registration §2) and never tuned per window.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    xm, xs = _scaler(X)
    Xs = (X - xm) / xs
    ym = Y.mean(0)
    d = Xs.shape[1]
    B = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(d), Xs.T @ (Y - ym))
    return LinearPredictor(B=B, b=ym, x_mean=xm, x_std=xs, loss_history=[])


# Fixed by pre-registration; NOT searched per window. Chosen on synthetic data
# by the only criterion available before seeing any real result: the SPO+
# training loss must fall monotonically. The paper's optimizer is Adam (Table 2)
# and its learning-rate range is 1e-4 to 5e-2 -- with standardized features and
# return-scale targets, the low end of that range is the usable one, because an
# Adam step moves each coefficient by about `lr` while the coefficients
# themselves are O(1e-3).
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 63          # their Table 2


def fit_spo_plus(X: np.ndarray, Y: np.ndarray, lam: float = 0.0,
                 gamma: float = 0.0, lr: float = DEFAULT_LR,
                 epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH,
                 seed: int = 0,
                 init: LinearPredictor | None = None) -> LinearPredictor:
    """Train the linear predictor against the SPO+ surrogate, Adam, minibatch.

    Warm-started at the ridge solution unless `init` is given -- the same start
    PyEPO recommends. It also makes the SPO+ vs PtO comparison a question of
    what the decision-focused training *adds*, rather than of where each run
    happened to begin.

    Adam rather than plain SGD because the SPO+ subgradient is `2(w_tilde - w*)`,
    whose entries are O(1) regardless of the return scale, while the coefficients
    it updates are O(1e-3). Under plain SGD any learning rate that trains at all
    is a property of the data's units; Adam normalizes that away, which is also
    what the paper does.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    T, n = Y.shape
    model = init if init is not None else fit_ridge(X, Y)
    Xs = model.standardize(X)
    B, b = model.B.copy(), model.b.copy()

    mB, vB = np.zeros_like(B), np.zeros_like(B)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        order = rng.permutation(T)
        epoch_loss = 0.0
        for s in range(0, T, batch_size):
            idx = order[s:s + batch_size]
            gB, gb = np.zeros_like(B), np.zeros_like(b)
            for i in idx:
                r_hat = Xs[i] @ B + b
                loss, g = spo_plus(r_hat, Y[i], lam=lam, gamma=gamma)
                epoch_loss += loss
                gB += np.outer(Xs[i], g)
                gb += g
            m = max(len(idx), 1)
            gB /= m
            gb /= m

            step += 1
            mB = beta1 * mB + (1 - beta1) * gB
            vB = beta2 * vB + (1 - beta2) * gB**2
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * gb**2
            c1, c2 = 1 - beta1**step, 1 - beta2**step
            B -= lr * (mB / c1) / (np.sqrt(vB / c2) + eps)
            b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        history.append(epoch_loss / T)

    return LinearPredictor(B=B, b=b, x_mean=model.x_mean, x_std=model.x_std,
                           loss_history=history)
