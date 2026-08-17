"""Stitch per-roll training chunks into a full backtest and score it.

Reassembles the chunks written by `train_roll.py` in offset order, verifies the
assembled length matches their `pc.backtest` sizing, and reports the honest
metric block against the shipped cache for the same net.

Usage:
    python scripts/combine_rolls.py                # all nets found
    python scripts/combine_rolls.py --net nom_net
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polab import e2edro_io, metrics as M, rolls as R  # noqa: E402

CHUNKS = ROOT / "results" / "rolls"


def load_chunks(net: str) -> list[dict]:
    out = []
    for p in sorted(CHUNKS.glob(f"{net}_roll*.pkl")):
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return sorted(out, key=lambda c: c["offset"])


def assemble(chunks: list[dict]) -> tuple[pd.Series, pd.DataFrame]:
    """Concatenate chunks in offset order. Each chunk carries only ITS OWN test
    dates (114/113/114/113), so the full index is their concatenation."""
    rets = np.concatenate([c["rets"] for c in chunks])
    weights = np.vstack([c["weights"] for c in chunks])
    idx = pd.DatetimeIndex(np.concatenate([np.asarray(c["dates"]) for c in chunks]))
    assert idx.is_monotonic_increasing, "roll windows overlap or are out of order"
    return (pd.Series(rets, index=idx, name="rets"),
            pd.DataFrame(weights, index=idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="e.g. nom_net; default: all found")
    args = ap.parse_args()

    if not CHUNKS.exists():
        sys.exit(f"no chunks in {CHUNKS} — run scripts/train_roll.py first")

    groups = defaultdict(list)
    for p in CHUNKS.glob("*_roll*.pkl"):
        groups[p.name.rsplit("_roll", 1)[0]].append(p)
    nets = [args.net] if args.net else sorted(groups)

    expected_total = R.total_test_windows()
    expected_rolls = len(R.plan())
    rows = []
    for net in nets:
        chunks = load_chunks(net)
        if not chunks:
            print(f"[skip] {net}: no chunks")
            continue
        have = {c["roll"] for c in chunks}
        missing = sorted(set(range(expected_rolls)) - have)
        if missing:
            print(f"[partial] {net}: {len(chunks)}/{expected_rolls} rolls, "
                  f"missing {missing} — skipping (rerun those jobs)")
            continue

        rets, weights = assemble(chunks)
        if len(rets) != expected_total:
            print(f"[warn] {net}: assembled {len(rets)} weeks, "
                  f"expected {expected_total}")
        hours = sum(c["train_seconds"] for c in chunks) / 3600
        row = {"name": net, "sharpe_retrained": M.sharpe(rets, 52),
               "ann_return": M.ann_return(rets, 52),
               "max_drawdown": M.max_drawdown(rets),
               "n_weeks": len(rets), "train_hours": hours}

        try:
            cached = e2edro_io.net_returns(e2edro_io.load_net(net))
            row["sharpe_cached"] = M.sharpe(cached, 52)
            row["delta"] = row["sharpe_retrained"] - row["sharpe_cached"]
        except Exception:  # noqa: BLE001 — no cached counterpart is fine
            row["sharpe_cached"] = np.nan
            row["delta"] = np.nan
        rows.append(row)

        out = ROOT / "results" / f"retrained_{net}.csv"
        pd.DataFrame({"rets": rets}).to_csv(out)
        weights.to_csv(ROOT / "results" / f"retrained_{net}_weights.csv")

    if not rows:
        sys.exit("nothing complete to combine yet")
    table = pd.DataFrame(rows).set_index("name")
    print("\n" + table.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nsuccess criterion: ranking preserved and sharpe_retrained close to "
          "sharpe_cached. Exact reproduction is not expected — library versions "
          "moved since 2022 even with their seed (1000). Record either way.")


if __name__ == "__main__":
    main()
