"""Offline test suite — runs on synthetic data, no network needed.

The load-bearing test is test_no_lookahead: it mechanically verifies the
backtest engine cannot see the future.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polab import backtest, baselines, data, metrics, rolls, sharpe_test, spo  # noqa: E402

AF = data.ANN_FACTOR_WEEKLY


def _synthetic():
    return data.synthetic_returns(n_periods=400, n_assets=6, seed=42)


def test_baseline_weights_valid():
    R = _synthetic()
    window = R.iloc[:104]
    for name, fn in baselines.BASELINES.items():
        w = fn(window)
        assert w.shape == (6,), name
        assert np.isclose(w.sum(), 1.0), name
        assert (w >= -1e-12).all(), f"{name} not long-only"


def test_risk_parity_equalizes_risk():
    R = _synthetic()
    w = baselines.risk_parity(R.iloc[:200], shrinkage=0.0)
    S = np.asarray(R.iloc[:200].cov())
    rc = w * (S @ w)  # risk contributions
    assert rc.max() / rc.min() < 1.10, "risk contributions should be ~equal"


def test_walk_forward_runs_and_shapes():
    R = _synthetic()
    res = backtest.walk_forward(R, baselines.equal_weight,
                                lookback=104, rebalance_every=4, cost_bps=10)
    assert len(res.returns) == len(R) - 104
    assert res.weights.shape == (len(R) - 104, 6)
    assert not res.returns.isna().any()
    assert (res.turnover >= 0).all()


def test_equal_weight_first_period_matches_manual():
    R = _synthetic()
    res = backtest.walk_forward(R, baselines.equal_weight,
                                lookback=104, rebalance_every=4, cost_bps=0)
    manual = R.iloc[104].mean()  # 1/N portfolio return, no cost
    assert np.isclose(res.returns.iloc[0], manual)


def test_costs_reduce_returns():
    R = _synthetic()
    free = backtest.walk_forward(R, baselines.min_variance, lookback=104,
                                 rebalance_every=4, cost_bps=0)
    paid = backtest.walk_forward(R, baselines.min_variance, lookback=104,
                                 rebalance_every=4, cost_bps=50)
    assert (1 + free.returns).prod() > (1 + paid.returns).prod()


def test_no_lookahead():
    """Corrupting the future must not change past decisions or past returns."""
    R = _synthetic()
    T = 300
    R_corrupt = R.copy()
    # anything loud enough that a look-ahead leak would show up in the weights
    R_corrupt.iloc[T:] = R_corrupt.iloc[T:] * 5.0 + 0.02

    for fn in (baselines.equal_weight, baselines.min_variance):
        a = backtest.walk_forward(R, fn, lookback=104, rebalance_every=4)
        b = backtest.walk_forward(R_corrupt, fn, lookback=104, rebalance_every=4)
        # everything strictly before T must be identical
        pd.testing.assert_frame_equal(a.weights.loc[:R.index[T - 1]],
                                      b.weights.loc[:R.index[T - 1]])
        pd.testing.assert_series_equal(a.returns.loc[:R.index[T - 1]],
                                       b.returns.loc[:R.index[T - 1]])


def test_psr_sane():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=500, freq="W-FRI")
    good = pd.Series(rng.normal(0.004, 0.02, 500), index=idx)
    noise = pd.Series(rng.normal(0.0, 0.02, 500), index=idx)
    assert metrics.probabilistic_sharpe(good) > 0.95
    assert 0.05 < metrics.probabilistic_sharpe(noise) < 0.95


def test_dsr_deflates():
    """DSR against many trials must be <= PSR against zero."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-01", periods=300, freq="W-FRI")
    r = pd.Series(rng.normal(0.002, 0.02, 300), index=idx)
    trials = list(rng.normal(0.0, 0.1, 20))
    trials.append(r.mean() / r.std(ddof=1))
    assert metrics.deflated_sharpe(r, trials) <= metrics.probabilistic_sharpe(r) + 1e-12


def test_cost_overlay():
    """0 bps overlay is identity; higher costs monotonically hurt; matches engine."""
    R = _synthetic()
    res = backtest.walk_forward(R, baselines.min_variance, lookback=104,
                                rebalance_every=4, cost_bps=0.0)
    zero, _ = backtest.cost_overlay(res.returns, res.weights, R, 0.0)
    pd.testing.assert_series_equal(zero, res.returns)
    net10, turn = backtest.cost_overlay(res.returns, res.weights, R, 10.0)
    net25, _ = backtest.cost_overlay(res.returns, res.weights, R, 25.0)
    assert (net10 <= res.returns + 1e-15).all()
    assert (1 + net25).prod() < (1 + net10).prod()
    assert (turn >= 0).all()
    # overlay must agree with the engine's own cost accounting (post-initial):
    engine10 = backtest.walk_forward(R, baselines.min_variance, lookback=104,
                                     rebalance_every=4, cost_bps=10.0)
    diff = (net10.iloc[1:] - engine10.returns.iloc[1:]).abs().max()
    assert diff < 1e-12


def test_roll_plan_matches_their_backtest():
    """Our roll split arithmetic must tile their OOS window exactly.

    Ground truth: their cached nets have 454 OOS weeks (observed in iter 2).
    """
    plan = rolls.plan()
    assert len(plan) == 4
    assert sum(r.n_test_windows for r in plan) == 454
    assert rolls.total_test_windows() == 454
    # offsets must be contiguous and start at 0
    off = 0
    for r in plan:
        assert r.offset == off
        off += r.n_test_windows
    # training sets grow monotonically across roll windows
    sizes = [r.n_train_windows for r in plan]
    assert sizes == sorted(sizes) and len(set(sizes)) == 4


def _paired_pair(seed=7, edge=0.0025, n=454):
    """Two correlated weekly return series sharing a market factor; `edge` is
    the per-period mean advantage given to the first."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-25", periods=n, freq="W")
    mkt = rng.normal(0.0, 0.018, n)
    a = 0.0010 + edge + 0.9 * mkt + rng.normal(0.0, 0.008, n)
    b = 0.0010 + 1.0 * mkt + rng.normal(0.0, 0.008, n)
    return pd.Series(a, idx), pd.Series(b, idx)


def test_sharpe_diff_identical_series_is_degenerate_not_an_error():
    """Our 1/N reproduces their ew_net exactly, so this pair is in every sweep."""
    a, _ = _paired_pair()
    for fn in (sharpe_test.jobson_korkie, sharpe_test.ledoit_wolf,
               sharpe_test.ledoit_wolf_bootstrap):
        r = fn(a, a.copy(), AF)
        assert r.diff == 0.0 and r.p_value == 1.0
        assert not r.significant
        assert r.method.endswith("identical_series")


def test_sharpe_diff_detects_a_real_gap_and_is_antisymmetric():
    a, b = _paired_pair(edge=0.0025)
    fwd = sharpe_test.ledoit_wolf(a, b, AF)
    rev = sharpe_test.ledoit_wolf(b, a, AF)
    assert fwd.diff > 0.5 and fwd.significant
    assert np.isclose(fwd.diff, -rev.diff)
    assert np.isclose(fwd.p_value, rev.p_value)
    assert np.isclose(fwd.diff, fwd.sharpe_a - fwd.sharpe_b)


def test_sharpe_diff_no_gap_is_not_significant():
    a, b = _paired_pair(edge=0.0)
    assert not sharpe_test.ledoit_wolf(a, b, AF).significant


def test_sharpe_diff_pvalue_invariant_to_annualization():
    """Annualization scales diff and its SE by the same sqrt(ann_factor)."""
    a, b = _paired_pair()
    wk = sharpe_test.ledoit_wolf(a, b, 1)
    yr = sharpe_test.ledoit_wolf(a, b, AF)
    assert np.isclose(wk.p_value, yr.p_value)
    assert np.isclose(yr.diff, wk.diff * np.sqrt(AF))
    assert np.isclose(yr.std_error, wk.std_error * np.sqrt(AF))


def test_sharpe_diff_bootstrap_ci_brackets_the_estimate():
    a, b = _paired_pair()
    r = sharpe_test.ledoit_wolf_bootstrap(a, b, AF, n_boot=299, seed=1)
    assert r.ci_low < r.diff < r.ci_high
    assert 0.0 <= r.p_value <= 1.0


def test_compare_to_benchmark_covers_every_other_column():
    a, b = _paired_pair()
    panel = pd.DataFrame({"a": a, "b": b, "c": a * 0.5 + b * 0.5})
    tab = sharpe_test.compare_to_benchmark(panel, "b", AF, method="lw")
    assert set(tab.index) == {"a", "c"}
    assert (tab["name_b"] == "b").all()


# --- SPO+ : the five checks pre-registered in audits/prereg-spo-retest.md -----
# Correctness is tied to these, NOT to whether SPO+ wins a backtest.

def test_spo_oracle_prediction_is_lossless():
    """Check 1. Fed the realized returns, the layer returns the oracle portfolio
    and the SPO+ loss is exactly zero."""
    rng = np.random.default_rng(0)
    for lam, gamma in [(0.0, 0.0), (0.42, 0.0), (0.0, 0.005)]:
        for _ in range(20):
            r = rng.normal(0, 0.02, 9)
            loss, grad = spo.spo_plus(r, r, lam=lam, gamma=gamma)
            assert abs(loss) < 1e-8, (lam, gamma, loss)
            assert np.abs(grad).max() < 1e-7
            assert abs(spo.decision_regret(r, r, lam=lam, gamma=gamma)) < 1e-9


def test_spo_plus_upper_bounds_regret():
    """Check 2. The theorem the surrogate exists for (Elmachtoub & Grigas 2022),
    verified including the h != 0 case the paper does not derive."""
    rng = np.random.default_rng(1)
    for lam, gamma in [(0.0, 0.0), (0.42, 0.0), (0.0, 0.005), (0.42, 0.005)]:
        for _ in range(40):
            r = rng.normal(0, 0.02, 9)
            r_hat = r + rng.normal(0, 0.02, 9)
            loss, _ = spo.spo_plus(r_hat, r, lam=lam, gamma=gamma)
            regret = spo.decision_regret(r_hat, r, lam=lam, gamma=gamma)
            assert regret >= -1e-9
            assert loss >= regret - 1e-9, (lam, gamma, loss, regret)


def test_spo_subgradient_matches_finite_differences():
    """Check 3. Away from ties the loss is differentiable, so the analytic
    subgradient must agree with a central difference."""
    rng = np.random.default_rng(2)
    eps = 1e-6
    for lam in (0.0, 0.42):
        for _ in range(10):
            r = rng.normal(0, 0.02, 9)
            r_hat = r + rng.normal(0, 0.05, 9)   # well away from ties
            _, grad = spo.spo_plus(r_hat, r, lam=lam)
            num = np.empty_like(grad)
            for j in range(len(r_hat)):
                e = np.zeros_like(r_hat)
                e[j] = eps
                hi, _ = spo.spo_plus(r_hat + e, r, lam=lam)
                lo, _ = spo.spo_plus(r_hat - e, r, lam=lam)
                num[j] = (hi - lo) / (2 * eps)
            assert np.abs(num - grad).max() < 1e-4, (lam, num, grad)


def test_spo_degenerate_prediction_stays_feasible():
    """Check 5. Constant predictions make every vertex optimal; the layer must
    still return a feasible portfolio rather than crash or emit NaN."""
    for lam in (0.0, 0.42):
        w = spo.decide(np.zeros(9), lam=lam)
        assert np.isfinite(w).all()
        assert np.isclose(w.sum(), 1.0) and (w >= -1e-12).all()


def test_spo_layer_shapes_and_closed_forms():
    """The l2 layer must agree with a brute-force solve, and the plain layer
    must be the argmax vertex."""
    rng = np.random.default_rng(3)
    c = rng.normal(0, 0.02, 9)
    w = spo.decide(c)
    assert w.sum() == 1.0 and w[int(np.argmax(c))] == 1.0

    lam = 0.42
    w2 = spo.decide(c, lam=lam)
    best = spo.layer_value(c, w2, lam=lam)
    for _ in range(2000):                       # random feasible points
        g = rng.random(9)
        cand = g / g.sum()
        assert spo.layer_value(c, cand, lam=lam) <= best + 1e-9


def test_spo_training_reduces_its_own_loss():
    """Sanity on the optimizer itself: SPO+ training must lower in-sample SPO+
    loss relative to the ridge start it is warm-started from.

    In-sample only, and it says nothing about out-of-sample performance -- it is
    not evidence for the method, only that the gradient and step size are wired
    up. The epoch-loss path is minibatch-noisy, so the assertion is on the level
    reached, not on monotonicity.
    """
    rng = np.random.default_rng(4)
    T, d, n = 300, 4, 9
    X = rng.normal(0, 1, (T, d))
    Btrue = rng.normal(0, 0.004, (d, n))
    Y = X @ Btrue + rng.normal(0, 0.01, (T, n))

    ridge = spo.fit_ridge(X, Y)
    trained = spo.fit_spo_plus(X, Y, seed=0)

    def mean_loss(model):
        P = model.predict(X)
        return float(np.mean([spo.spo_plus(P[i], Y[i])[0] for i in range(T)]))

    assert mean_loss(trained) < mean_loss(ridge)
    assert len(trained.loss_history) == spo.DEFAULT_EPOCHS


def test_l2_layer_collapses_to_equal_weight_at_the_papers_lambda():
    """arXiv:2601.04062 sets lambda = 0.42 but never states the units of r.

    At decimal-return scale the penalty term (0.42 * ||w||^2 ~ 0.047 at w = 1/N)
    is several times the typical |r| (~0.009 daily), so the layer minimizes
    ||w||^2 and returns essentially the equal-weight portfolio regardless of
    what the predictor says. Their "SPO+ with l2" variant is then 1/N wearing a
    decision-focused label. Scaled by 100 (percent returns) the same lambda
    concentrates instead -- so the specification is ambiguous by a factor that
    changes the strategy completely. Pinned here because it is a finding.
    """
    rng = np.random.default_rng(5)
    n = 9
    decimal_scale, percent_scale = [], []
    for _ in range(200):
        r = rng.normal(0.0, 0.009, n)
        decimal_scale.append(spo.decide(r, lam=0.42))
        percent_scale.append(spo.decide(r * 100.0, lam=0.42))

    d_eff = np.mean([1.0 / np.sum(w**2) for w in decimal_scale])
    p_eff = np.mean([1.0 / np.sum(w**2) for w in percent_scale])
    assert d_eff > 8.5, f"decimal-scale layer should be ~1/N, got n_eff={d_eff}"
    assert p_eff < 4.0, f"percent-scale layer should concentrate, got n_eff={p_eff}"


def test_compare_table():
    R = _synthetic()
    table = backtest.compare(R, baselines.BASELINES, ann_factor=AF,
                             lookback=104, rebalance_every=4, cost_bps=10)
    assert set(table.index) == set(baselines.BASELINES)
    assert {"sharpe", "max_drawdown", "avg_turnover", "dsr"} <= set(table.columns)


# ---------------------------------------------------------------------------
# features + the SPO re-test causality gate (iteration 7)
# ---------------------------------------------------------------------------

def _toy_prices(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = ["A", "B", "C"]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)),
                      index=idx, columns=cols)
    vol = pd.DataFrame(rng.lognormal(15, 0.3, (n, 3)), index=idx, columns=cols)
    return px, vol


def test_features_are_causal():
    """Every feature row must be unchanged by anything after that row."""
    from polab import features as F
    px, vol = _toy_prices()
    full = F.panel_features(px, vol)
    cut = full.index[250]
    corrupted = px.copy()
    corrupted.loc[corrupted.index > cut] *= 3.0
    part = F.panel_features(corrupted, vol)
    common = full.index[full.index <= cut]
    pd.testing.assert_frame_equal(full.loc[common], part.loc[common])


def test_features_shape_and_finiteness():
    from polab import features as F
    px, vol = _toy_prices()
    w = F.panel_features(px, vol)
    assert w.shape[1] == 3 * len(F.FEATURE_NAMES)
    assert np.isfinite(w.to_numpy()).all()
    assert not w.empty


def test_retest_training_slice_never_uses_unrealized_targets():
    """The load-bearing gate: a training sample dated s is admissible at t only
    if its forward-HORIZON target is fully realized, i.e. s + HORIZON <= t.
    Violating this is the classic overlapping-target look-ahead."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "spo_retest", Path(__file__).resolve().parent.parent
        / "scripts" / "run_spo_retest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from polab import features as F
    px, vol = _toy_prices()
    feats = F.panel_features(px, vol)
    fwd = px.shift(-mod.HORIZON) / px - 1.0

    for t in [feats.index[120], feats.index[200], feats.index[-1]]:
        X, Y, dates = mod.training_slice(feats, fwd, t)
        assert len(dates) == len(X) == len(Y)
        for s in dates:
            pos = fwd.index.get_loc(s)
            assert fwd.index[pos + mod.HORIZON] <= t, (
                f"sample {s.date()} used at {t.date()} but its target only "
                f"resolves at {fwd.index[pos + mod.HORIZON].date()}")
        assert not np.isnan(Y).any()


def test_retest_archetype_thresholds_match_prereg():
    """MDE assignment must follow the pre-registered power table, not a
    blanket number: power depends on correlation with 1/N, i.e. concentration."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "spo_retest2", Path(__file__).resolve().parent.parent
        / "scripts" / "run_spo_retest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.archetype(9.0) == "diversified"
    assert mod.archetype(3.8) == "concentrated"
    assert mod.archetype(1.0) == "corner"
    assert mod.MDE_BY_ARCHETYPE["corner"] > mod.MDE_BY_ARCHETYPE["concentrated"]
    assert mod.MDE_BY_ARCHETYPE["concentrated"] > mod.MDE_BY_ARCHETYPE["diversified"]


# ---------------------------------------------------------------------------
# simulator + stability rules (iteration 8)
# ---------------------------------------------------------------------------

def test_simulator_moments_match_calibration():
    """Simulated monthly moments must sit near the sector-ETF panel they claim
    to be calibrated to, else every IC on this grid is measured off-scale."""
    from polab import simulate as sim
    R = sim.simulate_returns(50, 2000, seed=0)
    per_asset_vol = R.std(axis=0).mean()
    target = np.sqrt(sim.MARKET_VOL**2 + sim.IDIO_VOL**2)
    assert abs(per_asset_vol - target) < 0.004, (per_asset_vol, target)
    # a single common factor must dominate the cross-sectional correlation
    corr = np.corrcoef(R.T)
    off = corr[~np.eye(len(corr), dtype=bool)].mean()
    assert 0.5 < off < 0.85, off


def test_signal_achieves_requested_ic():
    """The whole design rests on IC being what we asked for."""
    from polab import simulate as sim
    R = sim.simulate_returns(40, 3000, seed=1)
    for ic in (0.0, 0.05, 0.20):
        S = sim.signal_with_ic(R, ic, seed=1)
        assert abs(sim.realized_ic(S, R) - ic) < 0.02, ic


def test_zero_ic_signal_is_uninformative():
    from polab import simulate as sim
    R = sim.simulate_returns(30, 3000, seed=2)
    assert abs(sim.realized_ic(sim.signal_with_ic(R, 0.0, seed=2), R)) < 0.02


def test_stability_blend_endpoints():
    from polab import stability as stab
    prev = np.array([0.7, 0.2, 0.1])
    target = np.array([0.1, 0.1, 0.8])
    assert np.allclose(stab.blend(prev, target, 1.0), target)
    assert np.allclose(stab.blend(prev, target, 0.0), prev)
    mid = stab.blend(prev, target, 0.5)
    assert np.isclose(mid.sum(), 1.0) and (mid >= 0).all()


def test_alpha_zero_never_trades():
    """alpha=0 must be buy-and-hold: zero turnover, hence zero cost. This is
    the null the whole sweep is measured against."""
    from polab import simulate as sim, stability as stab
    R = sim.simulate_returns(12, 200, seed=3)
    S = sim.signal_with_ic(R, 0.2, seed=3)
    p = stab.run_path(R, S, alpha=0.0, cost_bps=50.0)
    assert np.allclose(p["traded"], 0.0)
    assert np.allclose(p["gross"], p["net"])


def test_turnover_increases_with_alpha():
    from polab import simulate as sim, stability as stab
    R = sim.simulate_returns(20, 200, seed=4)
    S = sim.signal_with_ic(R, 0.1, seed=4)
    turns = [stab.run_path(R, S, alpha=a)["turnover"].mean()
             for a in (0.0, 0.25, 0.5, 1.0)]
    assert turns == sorted(turns) and turns[-1] > turns[0]


def test_target_concentration_is_stable_across_ic():
    """lambda is set from the signal's own spread precisely so that a stronger
    signal does not mechanically concentrate the target -- otherwise IC and
    concentration are confounded and alpha* is uninterpretable."""
    from polab import simulate as sim, stability as stab
    R = sim.simulate_returns(25, 400, seed=5)
    eff = []
    for ic in (0.02, 0.20):
        S = sim.signal_with_ic(R, ic, seed=5)
        w = np.array([stab.target_weights(s) for s in S])
        eff.append(float((1.0 / (w**2).sum(axis=1)).mean()))
    assert abs(eff[0] - eff[1]) / eff[0] < 0.15, eff


def test_no_trade_band_reduces_turnover_monotonically():
    from polab import simulate as sim, stability as stab
    R = sim.simulate_returns(10, 60, seed=6)
    S = sim.signal_with_ic(R, 0.1, seed=6)
    turns = [stab.run_path(R, S, alpha=1.0, tau=t)["turnover"].mean()
             for t in (0.0, 0.01, 0.05)]
    assert turns[0] >= turns[1] >= turns[2]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
