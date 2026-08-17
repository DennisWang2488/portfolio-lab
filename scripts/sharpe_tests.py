"""Is any of the E2E-DRO ranking statistically distinguishable from 1/N?

`compare_e2edro_cache.py` produces the ranking and the cost stress. This script
asks the question that ranking cannot answer: on ONE out-of-sample path of 454
weekly observations, which of those Sharpe gaps survives a paired test?

Two benchmarks, because they answer different questions:
  * `polab_equal_weight` -- the DeMiguel et al. (2009) null. "Did any of this
    beat doing nothing?"
  * `po_net` -- their own feature-based predict-then-optimize net. "Did
    end-to-end training beat two-stage?", i.e. the paper's actual claim.

Each at three cost levels, since iteration 2 established that the ranking is
turnover-fragile and the gross comparison is close to meaningless.

Usage: python scripts/sharpe_tests.py [--n-boot 4999]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polab import backtest, baselines, data, e2edro_io, sharpe_test as ST  # noqa: E402

AF = data.ANN_FACTOR_WEEKLY
COST_LEVELS = [0.0, 10.0, 25.0]
BENCHMARKS = ["polab_equal_weight", "po_net"]


def build_return_panels() -> dict[float, pd.DataFrame]:
    """{cost_bps: DataFrame of per-strategy net returns} on their OOS window.

    Same construction as `compare_e2edro_cache.py` -- their cached nets plus our
    baselines, with costs applied by the uniform overlay from each strategy's
    own weight path, so the only thing that differs across panels is the cost.
    """
    rets, wts = {}, {}
    for name in e2edro_io.MAIN_NETS:
        try:
            net = e2edro_io.load_net(name)
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")
            continue
        rets[name] = e2edro_io.net_returns(net)
        wts[name] = e2edro_io.net_weights(net)
    if not rets:
        sys.exit("no cached nets could be loaded")

    oos = next(iter(rets.values())).index
    R = data.load_e2edro("asset")
    for nm, fn in baselines.BASELINES.items():
        res = backtest.walk_forward(R, fn, lookback=104, rebalance_every=1,
                                    cost_bps=0.0, name=f"polab_{nm}")
        rets[f"polab_{nm}"] = res.returns.loc[oos[0]:oos[-1]]
        wts[f"polab_{nm}"] = res.weights.loc[oos[0]:oos[-1]]

    Rw = R.copy()
    Rw.columns = range(R.shape[1])
    panels = {}
    for bps in COST_LEVELS:
        cols = {}
        for nm, gross in rets.items():
            if bps == 0.0:
                cols[nm] = gross
            else:
                W = wts[nm].copy()
                W.columns = range(W.shape[1])
                cols[nm], _ = backtest.cost_overlay(gross, W, Rw, bps)
        panels[bps] = pd.DataFrame(cols)
    print(f"OOS window {oos[0].date()} .. {oos[-1].date()} ({len(oos)} weeks), "
          f"{len(rets)} strategies, costs {COST_LEVELS} bps\n")
    return panels


def run(panels: dict[float, pd.DataFrame], n_boot: int) -> pd.DataFrame:
    rows = []
    for bps, panel in panels.items():
        for bench in BENCHMARKS:
            for method in ("lw", "bootstrap"):
                kw = {"n_boot": n_boot} if method == "bootstrap" else {}
                tab = ST.compare_to_benchmark(panel, bench, AF,
                                              method=method, **kw)
                tab = tab.reset_index()
                tab.insert(0, "cost_bps", bps)
                rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def block_size_sensitivity(panel: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """The bootstrap block size is a judgement call; show it does not drive the
    verdict. Reported for the headline contrast only (dr_net_learn_theta vs 1/N)."""
    rows = []
    for b in (1, 3, 5, 10, 20):
        r = ST.ledoit_wolf_bootstrap(panel["dr_net_learn_theta"],
                                     panel["polab_equal_weight"], AF,
                                     name_a="dr_net_learn_theta",
                                     name_b="polab_equal_weight",
                                     block_size=b, n_boot=n_boot)
        rows.append({"block_size": b, "diff": r.diff, "p_value": r.p_value,
                     "ci_low": r.ci_low, "ci_high": r.ci_high})
    return pd.DataFrame(rows).set_index("block_size")


def _show(df: pd.DataFrame, cost: float, bench: str, method: str) -> None:
    sub = df[(df.cost_bps == cost) & (df.name_b == bench)
             & (df.method.str.startswith(method))]
    if sub.empty:
        return
    print(f"--- vs {bench} @ {cost:.0f} bps "
          f"({'HAC asymptotic' if method == 'ledoit' else 'block bootstrap'}) ---")
    out = sub[["name_a", "sharpe_a", "sharpe_b", "diff", "std_error",
               "p_value", "ci_low", "ci_high", "significant_5pct"]].copy()
    for c in ("sharpe_a", "sharpe_b", "diff", "std_error", "ci_low", "ci_high"):
        out[c] = out[c].map("{:+.3f}".format)
    out["p_value"] = out["p_value"].map("{:.3f}".format)
    print(out.to_string(index=False), "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=4999)
    args = ap.parse_args()

    panels = build_return_panels()
    df = run(panels, args.n_boot)

    for cost in COST_LEVELS:
        for bench in BENCHMARKS:
            _show(df, cost, bench, "lw_bootstrap")
    print("=== HAC asymptotic (compare with the bootstrap above) ===\n")
    _show(df, 10.0, "polab_equal_weight", "ledoit")
    _show(df, 10.0, "po_net", "ledoit")

    sens = block_size_sensitivity(panels[10.0], args.n_boot)
    print("--- bootstrap block-size sensitivity "
          "(dr_net_learn_theta vs 1/N @ 10 bps) ---")
    print(sens.to_string(float_format=lambda v: f"{v:+.3f}"), "\n")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "sharpe_tests.csv", index=False)
    sens.to_csv(out_dir / "sharpe_tests_block_sensitivity.csv")
    print("saved: results/sharpe_tests.csv, "
          "results/sharpe_tests_block_sensitivity.csv")
    print("note: p-values are marginal (no multiple-testing adjustment); the "
          "selection-bias side is the DSR column of e2edro_cache_comparison.csv.")


if __name__ == "__main__":
    main()
