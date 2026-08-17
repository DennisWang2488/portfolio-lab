"""Execute the pre-registered SPO+ re-test (audits/prereg-spo-retest.md).

Nothing here is chosen after seeing a result. Universe, window, costs, protocol,
hypotheses and the decision rule are fixed in that document; the two parameters
it left open (technical-indicator windows, and lambda for the l2 variant) are
fixed in `polab/features.py` and in LAMBDA_L2 below, with their rationale, and
are reported whatever happens.

Design rule carried from the pre-registration (§4): this panel cannot reliably
resolve a Sharpe gap below ~0.5 for corner-solution strategies. Any smaller gap
is reported as NOT DETECTABLE regardless of which way the point estimate falls.

Usage:  python scripts/run_spo_retest.py            (~2-5 min)
        python scripts/run_spo_retest.py --quick    (fewer bootstrap resamples)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polab import (backtest, baselines, features as F, marketdata as md,  # noqa: E402
                   metrics as M, sharpe_test, spo)

# ---- fixed by pre-registration ------------------------------------------------
EVAL_START, EVAL_END = "2016-01-01", "2024-12-31"
TRAIN_DAYS = 252          # "trailing 12 months"
HORIZON = 21              # decision held ~1 month; target must match the layer
COST_BPS = 50.0           # their value (prereg §2)

# Pre-registration §4 gives the minimum detectable effect BY ARCHETYPE, not one
# blanket number: a strategy's power depends on how correlated it is with 1/N,
# which is set by how concentrated it is. Assigned from each strategy's measured
# effective number of positions (1 / sum w^2), not by assumption.
MDE_BY_ARCHETYPE = {"diversified": 0.21, "concentrated": 0.37, "corner": 0.54}


def archetype(eff_n: float) -> str:
    if eff_n >= 4.0:
        return "diversified"
    return "concentrated" if eff_n >= 2.0 else "corner"

# Fixed in advance, not searched. Maximizing c'w - lam*||w||^2 over the simplex
# is proj_simplex(c / (2*lam)): lam sets how much predicted-return spread is
# needed to leave equal weight. Monthly cross-sectional spread here is ~0.02, so
# lam = 0.1 makes c/(2 lam) spread ~0.1 ~ 1/n -- i.e. genuinely diversified,
# which is the whole point of the variant (prereg puts it in the top power row).
LAMBDA_L2 = 0.1


def build_panel():
    """Prices, volumes, daily returns, and the causal feature matrix."""
    close = md.load_prices(md.SECTOR_ETFS, "adjclose")
    volume = md.load_prices(md.SECTOR_ETFS, "volume")
    rets = close.pct_change().dropna(how="any")
    feats = F.panel_features(close, volume)
    # forward HORIZON-day simple return, the quantity the layer optimizes
    fwd = close.shift(-HORIZON) / close - 1.0
    return close, rets, feats, fwd


def rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each month inside the evaluation window."""
    idx = index[(index >= EVAL_START) & (index <= EVAL_END)]
    return list(pd.Series(idx, index=idx).groupby(idx.to_period("M")).last())


def training_slice(feats, fwd, t):
    """Samples whose targets are FULLY REALIZED by `t` — the causality gate.

    A sample dated s has target over (s, s+HORIZON]. Using it at time t requires
    s + HORIZON <= t. Dropping this condition is the classic overlapping-target
    look-ahead; it is the single easiest way to fabricate alpha here.
    """
    usable = feats.index[feats.index <= t]
    pos = fwd.index.get_indexer(usable)
    horizon_ok = pos + HORIZON < len(fwd.index)
    realized = np.array([fwd.index[p + HORIZON] <= t if ok else False
                         for p, ok in zip(pos, horizon_ok)])
    dates = usable[realized][-TRAIN_DAYS:]
    X = feats.loc[dates].to_numpy()
    Y = fwd.loc[dates].to_numpy()
    ok = ~np.isnan(Y).any(axis=1)
    return X[ok], Y[ok], dates[ok]


def run_strategies(quick: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    close, rets, feats, fwd = build_panel()
    dates = rebalance_dates(rets.index)
    eval_idx = rets.index[(rets.index >= dates[0]) & (rets.index <= EVAL_END)]
    n = rets.shape[1]
    print(f"universe {list(rets.columns)}")
    print(f"eval {eval_idx[0].date()} .. {eval_idx[-1].date()} "
          f"({len(eval_idx)} days, {len(dates)} rebalances)")

    learned = ["PtO", "SPO+", "SPO+ (l2)"]
    classical = {"1/N": baselines.equal_weight,
                 "min_variance": baselines.min_variance,
                 "risk_parity": baselines.risk_parity,
                 "max_sharpe": baselines.max_sharpe}
    names = learned + list(classical)
    targets = {nm: {} for nm in names}

    t0 = time.time()
    for k, t in enumerate(dates):
        X, Y, _ = training_slice(feats, fwd, t)
        x_now = feats.loc[[t]].to_numpy()
        if len(X) < 60:                     # not enough realized targets yet
            for nm in names:
                targets[nm][t] = np.full(n, 1.0 / n)
            continue

        ridge = spo.fit_ridge(X, Y)
        m_spo = spo.fit_spo_plus(X, Y, lam=0.0, seed=0, init=ridge)
        m_l2 = spo.fit_spo_plus(X, Y, lam=LAMBDA_L2, seed=0, init=ridge)
        targets["PtO"][t] = spo.decide(ridge.predict(x_now)[0])
        targets["SPO+"][t] = spo.decide(m_spo.predict(x_now)[0])
        targets["SPO+ (l2)"][t] = spo.decide(m_l2.predict(x_now)[0], lam=LAMBDA_L2)

        window = rets.loc[:t].iloc[-TRAIN_DAYS:]
        for nm, fn in classical.items():
            targets[nm][t] = fn(window)

        if k % 12 == 0:
            print(f"  {t.date()} ({k + 1}/{len(dates)}) "
                  f"train={len(X)} [{time.time() - t0:.0f}s]", flush=True)

    # ---- daily holdings path: rebalance on schedule, drift in between --------
    gross, weights = {}, {}
    R = rets.loc[eval_idx].to_numpy()
    for nm in names:
        W = np.zeros((len(eval_idx), n))
        w = None
        for i, d in enumerate(eval_idx):
            if d in targets[nm]:
                w = np.asarray(targets[nm][d], float)
            elif w is None:
                w = np.full(n, 1.0 / n)
            else:
                w = backtest.drift(w, R[i - 1])
            W[i] = w
        weights[nm] = pd.DataFrame(W, index=eval_idx, columns=rets.columns)
        gross[nm] = pd.Series((W * R).sum(axis=1), index=eval_idx, name=nm)

    # SPY buy-and-hold: outside the 9-asset space, zero turnover by construction
    spy = md.load_prices(["SPY"], "adjclose")["SPY"].pct_change()
    gross["SPY B&H"] = spy.reindex(eval_idx).fillna(0.0).rename("SPY B&H")

    # ---- uniform cost treatment (prereg: 50 bps on ||w_t - w_{t-1}||_1) -----
    net, turn = {}, {}
    for nm in names:
        net[nm], turn[nm] = backtest.cost_overlay(gross[nm], weights[nm],
                                                  rets.loc[eval_idx], COST_BPS)
    net["SPY B&H"] = gross["SPY B&H"]
    turn["SPY B&H"] = pd.Series(0.0, index=eval_idx)

    return pd.DataFrame(gross), pd.DataFrame(net), turn, weights


def verdict(row, mde: dict) -> str:
    """The decision rule, fixed in advance (prereg §4 consequence 1).

    A gap smaller than what this design can resolve is reported as not
    detectable WHICHEVER WAY the point estimate falls -- a favourable point
    estimate is not a finding.
    """
    thr = mde.get(row["name_a"], 0.54)
    if abs(row["diff"]) < thr:
        return f"NOT DETECTABLE (|gap| < {thr:.2f})"
    return "significant" if row["p_value"] < 0.05 else "not significant"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_boot = 999 if args.quick else 4999

    gross, net, turn, weights = run_strategies(args.quick)
    af = md.ANN_FACTOR_DAILY
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # effective number of positions -> archetype -> MDE (prereg §4)
    eff_n, mde = {}, {}
    for nm, W in weights.items():
        e = float((1.0 / (W.to_numpy() ** 2).sum(axis=1)).mean())
        eff_n[nm] = e
        mde[nm] = MDE_BY_ARCHETYPE[archetype(e)]
    eff_n["SPY B&H"], mde["SPY B&H"] = 1.0, MDE_BY_ARCHETYPE["corner"]

    rows = []
    for nm in net.columns:
        rows.append({"name": nm,
                     "sharpe_gross": M.sharpe(gross[nm], af),
                     "sharpe_net50": M.sharpe(net[nm], af),
                     "ann_return_net": M.ann_return(net[nm], af),
                     "max_drawdown": M.max_drawdown(net[nm]),
                     "turnover_rebal": float(turn[nm][turn[nm] > 0].mean()
                                             if (turn[nm] > 0).any() else 0.0),
                     "eff_positions": eff_n.get(nm, np.nan),
                     "mde": mde.get(nm, np.nan)})
    perf = pd.DataFrame(rows).set_index("name").sort_values("sharpe_net50",
                                                            ascending=False)
    print("\n=== performance (50 bps unless noted) ===")
    print(perf.to_string(float_format=lambda v: f"{v:.3f}"))
    perf.to_csv(out / "spo_retest_performance.csv")

    tests = []
    for label, panel in (("gross", gross), ("net50", net)):
        for bench, hyp in (("1/N", "H1"), ("PtO", "H2")):
            tbl = sharpe_test.compare_to_benchmark(
                panel, bench, af, method="bootstrap", n_boot=n_boot, block_size=5)
            tbl = tbl.reset_index()
            tbl["cost"], tbl["hypothesis"], tbl["benchmark"] = label, hyp, bench
            tests.append(tbl)
    tests = pd.concat(tests, ignore_index=True)
    tests["mde"] = tests.name_a.map(mde)
    tests["verdict"] = tests.apply(lambda r: verdict(r, mde), axis=1)
    tests.to_csv(out / "spo_retest_tests.csv", index=False)

    print("\n=== hypothesis tests (paired LW studentized block bootstrap) ===")
    for hyp, bench in (("H1", "1/N"), ("H2", "PtO")):
        for cost in ("gross", "net50"):
            sub = tests[(tests.hypothesis == hyp) & (tests.cost == cost)]
            sub = sub[sub.name_a.isin(["SPO+", "SPO+ (l2)", "PtO"])]
            if sub.empty:
                continue
            print(f"\n{hyp}: vs {bench} ({cost})")
            for _, r in sub.iterrows():
                print(f"  {r.name_a:12s} dSharpe {r['diff']:+.3f} "
                      f"SE {r['std_error']:.3f}  p={r['p_value']:.3f}  -> {r.verdict}")

    print(f"\nsaved: results/spo_retest_performance.csv, results/spo_retest_tests.csv")
    print("\ndecision rule (fixed in advance, prereg §4): a gap below this "
          "design's\nminimum detectable effect for that strategy's archetype is "
          "reported as NOT\nDETECTABLE, whichever way the point estimate falls. "
          "Per-strategy MDE:")
    for nm in sorted(mde):
        print(f"  {nm:14s} eff_positions {eff_n[nm]:4.1f} -> "
              f"{archetype(eff_n[nm]):12s} MDE {mde[nm]:.2f}")


if __name__ == "__main__":
    main()
