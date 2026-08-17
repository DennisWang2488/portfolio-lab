"""Run the four classical baselines on the vendored E2E-DRO real dataset.

Usage:  python scripts/run_baselines.py [--cost-bps 10]

Setup mirrors Costa & Iyengar (2023): weekly returns, 20 US large-caps,
2-year estimation window, ~monthly rebalancing. The question the table
answers: does anything beat equal_weight after costs, with statistical
honesty (PSR / DSR)?
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polab import backtest, baselines, data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--lookback", type=int, default=104)   # 2y of weeks
    ap.add_argument("--rebalance-every", type=int, default=4)  # ~monthly
    args = ap.parse_args()

    R = data.load_e2edro("asset")
    print(f"data: {R.shape[1]} assets, {R.shape[0]} weeks, "
          f"{R.index[0].date()} .. {R.index[-1].date()}")
    print(f"cost={args.cost_bps}bps  lookback={args.lookback}w  "
          f"rebalance every {args.rebalance_every}w\n")

    table = backtest.compare(
        R, baselines.BASELINES, ann_factor=data.ANN_FACTOR_WEEKLY,
        lookback=args.lookback, rebalance_every=args.rebalance_every,
        cost_bps=args.cost_bps)

    fmt = {"ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.3f}",
           "max_drawdown": "{:.2%}", "psr_vs_zero": "{:.3f}", "dsr": "{:.3f}",
           "avg_turnover": "{:.3f}"}
    out = table.copy()
    for col, f in fmt.items():
        out[col] = out[col].map(f.format)
    print(out.to_string())
    print("\nReminder: PSR/DSR quantify confidence the true Sharpe exceeds the "
          "benchmark; DSR deflates for the 4 strategies tried in this table.")


if __name__ == "__main__":
    main()
