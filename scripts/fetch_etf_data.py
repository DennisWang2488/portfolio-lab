"""Populate the daily ETF price cache used by the arXiv:2601.04062 re-test.

Run once; the CSVs are committed and every later run is offline.

Universe choice (stated here because the paper's is unknowable -- it names
neither a count nor a ticker): the nine original SPDR select-sector ETFs, plus
SPY held out as a passive benchmark. Window starts a year before the paper's
2015 so that trailing features (SMA, RSI, MACD, Bollinger) are computable at
the start of the evaluation period without borrowing from it.

Survivorship: all ten instruments still trade today, so this universe is
survivorship-biased by construction. Sector ETFs are far less exposed to it
than a stock universe would be -- none was delisted or merged over the window --
but the bias is not zero and is reported rather than papered over.

Usage: python scripts/fetch_etf_data.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polab import marketdata as md  # noqa: E402

START, END = "2014-01-01", "2025-01-01"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-download even if cached (changes the data vintage)")
    args = ap.parse_args()

    tickers = md.SECTOR_ETFS + [md.BENCHMARK]
    print(f"universe: {tickers}\nwindow:   {START} .. {END}\n")
    md.download_universe(tickers, START, END, force=args.force)

    R = md.daily_returns()
    px = md.load_prices(tickers)
    print(f"\nreturn panel: {R.shape[0]} days x {R.shape[1]} assets, "
          f"{R.index[0].date()} .. {R.index[-1].date()}")
    print(f"aligned calendar: {len(px)} trading days across all {len(tickers)}")
    print("\nannualized summary (full window, buy and hold):")
    ann = (R.mean() * 252 * 100).round(2)
    vol = (R.std() * (252 ** 0.5) * 100).round(2)
    print((ann.to_frame("ann_ret_%").join(vol.to_frame("ann_vol_%"))).to_string())


if __name__ == "__main__":
    main()
