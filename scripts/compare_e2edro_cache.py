"""Cross-validate our harness against E2E-DRO's cached (pre-trained) results,
then stress the paper's gross-return ranking with a uniform transaction-cost
overlay.

Their backtest charges no costs and rebalances weekly (n_roll=4 retrains).
We (a) verify our engine reproduces their equal-weight numbers exactly,
(b) put their 8 nets and our 4 baselines in one honest table, and
(c) recompute Sharpe net of 10/25 bps from each strategy's own weight paths.

Usage: python scripts/compare_e2edro_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polab import backtest, baselines, data, e2edro_io, metrics as M  # noqa: E402


def main() -> None:
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
    print(f"their OOS window: {oos[0].date()} .. {oos[-1].date()} "
          f"({len(oos)} weeks)\n")

    # Our baselines under matched conditions: weekly rebalancing, zero cost,
    # identical OOS window. Costs are applied by the uniform overlay below.
    R = data.load_e2edro("asset")
    for nm, fn in baselines.BASELINES.items():
        res = backtest.walk_forward(R, fn, lookback=104, rebalance_every=1,
                                    cost_bps=0.0, name=f"polab_{nm}")
        rets[f"polab_{nm}"] = res.returns.loc[oos[0]:oos[-1]]
        wts[f"polab_{nm}"] = res.weights.loc[oos[0]:oos[-1]]

    # sanity: our 1/N must reproduce their ew_net to numerical precision
    diff = (rets["ew_net"] - rets["polab_equal_weight"]).abs().max()
    print(f"engine cross-check |their ew - our 1/N|_max = {diff:.2e} "
          f"({'OK' if diff < 1e-10 else 'MISMATCH — investigate'})\n")

    Rw = R.copy()
    Rw.columns = range(R.shape[1])  # align with anonymous net weight columns
    trial_srs = [s.mean() / s.std(ddof=1) for s in rets.values()]
    rows = []
    for nm, gross in rets.items():
        W = wts[nm]
        W.columns = range(W.shape[1])
        net10, turn = backtest.cost_overlay(gross, W, Rw, 10.0)
        net25, _ = backtest.cost_overlay(gross, W, Rw, 25.0)
        rows.append({
            "name": nm,
            "ann_return": M.ann_return(gross, 52),
            "ann_vol": M.ann_vol(gross, 52),
            "sharpe_gross": M.sharpe(gross, 52),
            "sharpe_10bps": M.sharpe(net10, 52),
            "sharpe_25bps": M.sharpe(net25, 52),
            "turnover_wk": float(turn.mean()),
            "max_drawdown": M.max_drawdown(gross),
            "dsr_gross": M.deflated_sharpe(gross, trial_srs),
        })
    table = (pd.DataFrame(rows).set_index("name")
             .sort_values("sharpe_10bps", ascending=False))

    fmt = {"ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe_gross": "{:.3f}",
           "sharpe_10bps": "{:.3f}", "sharpe_25bps": "{:.3f}",
           "turnover_wk": "{:.3f}", "max_drawdown": "{:.2%}", "dsr_gross": "{:.3f}"}
    out = table.copy()
    for col, f in fmt.items():
        out[col] = out[col].map(f.format)
    print(out.to_string())

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    table.to_csv(results_dir / "e2edro_cache_comparison.csv")
    pd.DataFrame(rets).to_csv(results_dir / "e2edro_cache_oos_returns.csv")
    print("\nsaved: results/e2edro_cache_comparison.csv, "
          "results/e2edro_cache_oos_returns.csv")
    print(f"notes: gross = their no-cost convention; net columns charge each "
          f"strategy's own weight path; DSR deflates across the "
          f"{len(rets)} strategies here.")

    diff_retrained(rets)


def diff_retrained(cached_rets: dict) -> None:
    """If Colab-retrained pickles exist (vendor/E2E-DRO/new_cache/exp), diff
    them against the shipped cache: gross Sharpe side by side."""
    if not e2edro_io.NEW_CACHE.exists():
        return
    print("\n=== retrained (new_cache) vs shipped cache ===")
    for name in e2edro_io.MAIN_NETS:
        try:
            net = e2edro_io.load_net(name, cache_dir=e2edro_io.NEW_CACHE)
        except FileNotFoundError:
            continue
        s_new = e2edro_io.net_returns(net)
        sr_new = M.sharpe(s_new, 52)
        if name in cached_rets:
            sr_old = M.sharpe(cached_rets[name], 52)
            print(f"{name:24s} cache {sr_old:6.3f} | retrained {sr_new:6.3f} "
                  f"| diff {sr_new - sr_old:+.3f}")
        else:
            print(f"{name:24s} retrained {sr_new:6.3f} (no cached counterpart)")


if __name__ == "__main__":
    main()
