"""IC x breadth x stability: what decision rule extracts a signal of known quality?

The question this replaces. Every empirical result here has been a null because
on one real path a Sharpe difference has SE 0.2-0.3 (audit prereg §4). So stop
asking "can anything beat 1/N" -- on one path, no -- and ask the question
institutions actually face: given a signal of quality IC over breadth N, and a
cost of c, how much should the decision move?

What is measured:
  IC        signal quality (cross-sectional corr of prediction with realization)
  breadth   number of assets
  alpha     stability strength: fraction of the way to the target we trade
            (alpha=1 full rebalance; alpha->0 never trade)

Pre-declared hypotheses (fixed before running, same discipline as the SPO re-test):
  S1  For each IC>0 the net-Sharpe-maximizing alpha* is INTERIOR (0 < alpha* < 1):
      neither full rebalancing nor never trading is optimal.
  S2  alpha* is INCREASING in IC: the weaker the signal, the more stability
      should be imposed. This is the claim the project's two nulls point to.
  S3  At IC=0 no alpha beats 1/N net of costs, and alpha* -> 0. A rule that
      profits from a pure-noise signal would indict the harness, not the market.

Validation independent of the outcome (mirrors prereg §5):
  V1  At alpha=1, gross IR should track Grinold's IC*sqrt(breadth) in ORDER OF
      MAGNITUDE and rise in both IC and breadth. Long-only caps it below the
      unconstrained law, so this is a sanity band, not an equality.
  V2  At IC=0, gross Sharpe must be statistically indistinguishable from 1/N.

Usage:  python scripts/run_ic_breadth_stability.py [--quick] [--bands]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polab import simulate as sim, stability as stab  # noqa: E402

AF = sim.ANN_FACTOR_MONTHLY
IC_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]
BREADTH_GRID = [10, 25, 50, 100]
ALPHA_GRID = [0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
# Cost is FREE to sweep: the blend rule never consults it, so net at any level
# is `gross - rate*traded` on the same simulated path. It is also the axis that
# decides everything -- see the break-even table.
COST_GRID = [0.0, 10.0, 25.0, 50.0]
TAU_GRID = [0.0, 0.002, 0.005, 0.010, 0.020]     # l1 no-trade band, --bands only
N_PERIODS = 240        # 20 years monthly
N_SEEDS = 40
COST_BPS = 50.0


def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(AF)) if sd > 1e-15 else np.nan


def sweep(ic_grid, breadth_grid, alpha_grid, cost_grid, n_seeds, n_periods):
    """One simulated path per (breadth, seed, ic, alpha); every cost level is
    then read off it without re-running."""
    rows = []
    for N in breadth_grid:
        for seed in range(n_seeds):
            R = sim.simulate_returns(N, n_periods, seed=seed)
            zero_sig = np.zeros_like(R)
            # THE null: the same investor without a signal. Buy-and-hold 1/N
            # (alpha=0 => never trades, so it pays no cost). Using
            # cost-paying rebalanced 1/N instead makes a zero-signal rule look
            # significantly profitable at 50 bps purely by not churning -- an
            # artifact of the benchmark, not an effect. Rebalanced 1/N is kept
            # as a reported reference (it is the DeMiguel benchmark).
            base = stab.run_path(R, zero_sig, alpha=0.0)
            rebal = stab.run_path(R, zero_sig, alpha=1.0)
            for ic in ic_grid:
                # common random numbers: the same noise draw across alpha, cost
                # and IC, so every comparison below is paired
                S = sim.signal_with_ic(R, ic, seed=seed)
                achieved = sim.realized_ic(S, R)
                for a in alpha_grid:
                    p = stab.run_path(R, S, alpha=a)
                    for c in cost_grid:
                        rate = c / 1e4
                        net = p["gross"] - rate * p["traded"]
                        b_net = base["gross"] - rate * base["traded"]
                        rb_net = rebal["gross"] - rate * rebal["traded"]
                        rows.append({
                            "breadth": N, "seed": seed, "ic": ic,
                            "ic_achieved": achieved, "alpha": a, "cost_bps": c,
                            "sharpe_gross": sharpe(p["gross"]),
                            "sharpe_net": sharpe(net),
                            "turnover": float(p["turnover"].mean()),
                            "excess_net": sharpe(net) - sharpe(b_net),
                            "excess_vs_rebal": sharpe(net) - sharpe(rb_net),
                            "excess_gross": sharpe(p["gross"]) - sharpe(base["gross"]),
                        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and paired SE across seeds for every (breadth, ic, alpha) cell."""
    g = df.groupby(["breadth", "ic", "alpha", "cost_bps"])
    out = g.agg(excess_net=("excess_net", "mean"),
                se=("excess_net", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                sharpe_net=("sharpe_net", "mean"),
                sharpe_gross=("sharpe_gross", "mean"),
                excess_vs_rebal=("excess_vs_rebal", "mean"),
                turnover=("turnover", "mean"),
                ic_achieved=("ic_achieved", "mean"),
                n=("excess_net", "size")).reset_index()
    # alpha=0 at IC=0 IS the benchmark, so its difference is identically zero
    # for every seed: report t=0 rather than 0/0.
    out["t"] = np.where(out.se > 1e-15, out.excess_net / out.se.replace(0, np.nan), 0.0)
    return out


def optimal_alpha(summary: pd.DataFrame) -> pd.DataFrame:
    """alpha* per (breadth, ic, cost), by mean net excess Sharpe over 1/N."""
    idx = summary.groupby(["breadth", "ic", "cost_bps"]).excess_net.idxmax()
    best = summary.loc[idx, ["breadth", "ic", "cost_bps", "alpha", "excess_net",
                             "se", "t", "turnover"]]
    return best.rename(columns={"alpha": "alpha_star"}).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bands", action="store_true",
                    help="also run the l1 no-trade-band mechanism (slower)")
    args = ap.parse_args()

    n_seeds = 8 if args.quick else N_SEEDS
    n_periods = 120 if args.quick else N_PERIODS
    print(f"grid: IC {IC_GRID} x breadth {BREADTH_GRID}")
    print(f"      alpha {ALPHA_GRID} x cost {COST_GRID} bps")
    print(f"{n_seeds} seeds x {n_periods} months\n")

    df = sweep(IC_GRID, BREADTH_GRID, ALPHA_GRID, COST_GRID, n_seeds, n_periods)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "ic_breadth_stability_raw.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "ic_breadth_stability_summary.csv", index=False)

    # ---- V1: does the simulator track the Fundamental Law? -----------------
    print("=== V1 calibration: gross IR at alpha=1 vs Grinold IC*sqrt(breadth) ===")
    full = summary[(summary.alpha == 1.0) & (summary.cost_bps == 0.0)]
    v1 = []
    for _, r in full[full.ic > 0].iterrows():
        v1.append({"ic": r.ic, "breadth": int(r.breadth),
                   "gross_excess_IR": r.sharpe_gross - summary[
                       (summary.breadth == r.breadth) & (summary.ic == 0.0)
                       & (summary.alpha == 1.0)
                       & (summary.cost_bps == 0.0)].sharpe_gross.iloc[0],
                   "grinold": sim.grinold_ir(r.ic, int(r.breadth))})
    v1 = pd.DataFrame(v1)
    v1["ratio"] = v1.gross_excess_IR / v1.grinold
    print(v1.pivot(index="ic", columns="breadth", values="ratio")
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print("(long-only caps the achievable IR below the unconstrained law, so "
          "ratios < 1 are expected; what matters is that they are stable and "
          "the IR rises in both IC and breadth)")

    # ---- S3 / V2: the pure-noise null --------------------------------------
    print("\n=== S3/V2 null check: IC = 0 ===")
    null = summary[(summary.ic == 0.0) & (summary.cost_bps == COST_BPS)]
    worst = null.loc[null.t.abs().idxmax()]
    print(f"largest |t| vs buy-and-hold over all (breadth, alpha) cells: "
          f"{worst.t:+.2f} at breadth={int(worst.breadth)}, alpha={worst.alpha}")
    print(f"best alpha at IC=0 by mean excess: "
          f"{null.loc[null.excess_net.idxmax(), 'alpha']}")

    # ---- S1 / S2: the main result ------------------------------------------
    best = optimal_alpha(summary)
    best.to_csv(out / "ic_breadth_stability_alpha_star.csv", index=False)

    print("\n=== S1/S2: alpha* (how far to trade toward the target) ===")
    for c in COST_GRID:
        sub = best[best.cost_bps == c]
        print(f"\ncost {c:.0f} bps:")
        print(sub.pivot(index="ic", columns="breadth", values="alpha_star")
              .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== net excess Sharpe over buy-and-hold 1/N at alpha*, t-stat in parens ===")
    for c in COST_GRID:
        sub = best[best.cost_bps == c].copy()
        sub["cell"] = sub.apply(lambda r: f"{r.excess_net:+.3f}({r.t:+.1f})", axis=1)
        print(f"\ncost {c:.0f} bps:")
        print(sub.pivot(index="ic", columns="breadth", values="cell").to_string())

    # ---- the practically useful number -------------------------------------
    print("\n=== break-even IC: lowest IC whose best alpha beats buy-and-hold at t>2 ===")
    print(f"{'cost':>6} " + " ".join(f"{('N=' + str(N)):>9}" for N in BREADTH_GRID))
    for c in COST_GRID:
        cells = []
        for N in BREADTH_GRID:
            sub = best[(best.breadth == N) & (best.cost_bps == c) & (best.t > 2.0)]
            cells.append(f"{sub.ic.min():.2f}" if len(sub) else f">{max(IC_GRID):.2f}")
        print(f"{c:>5.0f}b " + " ".join(f"{v:>9}" for v in cells))

    if args.bands:
        print("\n=== l1 no-trade band vs partial rebalancing (reduced grid) ===")
        rows = []
        for N in [25]:
            for seed in range(min(n_seeds, 12)):
                R = sim.simulate_returns(N, n_periods, seed=seed)
                base = stab.equal_weight_path(R, cost_bps=COST_BPS)
                b = sharpe(base["net"])
                for ic in [0.02, 0.05, 0.10]:
                    S = sim.signal_with_ic(R, ic, seed=seed)
                    for tau in TAU_GRID:
                        p = stab.run_path(R, S, alpha=1.0, cost_bps=COST_BPS,
                                          tau=tau)
                        rows.append({"ic": ic, "tau": tau,
                                     "excess_net": sharpe(p["net"]) - b,
                                     "turnover": float(p["turnover"].mean())})
        bands = pd.DataFrame(rows)
        bands.to_csv(out / "ic_stability_bands.csv", index=False)
        print(bands.groupby(["ic", "tau"])
              .agg(excess_net=("excess_net", "mean"),
                   turnover=("turnover", "mean")).to_string(
                       float_format=lambda v: f"{v:.3f}"))

    print(f"\nsaved: results/ic_breadth_stability_{{raw,summary,alpha_star}}.csv")


if __name__ == "__main__":
    main()
