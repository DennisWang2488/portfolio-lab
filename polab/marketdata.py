"""Daily US ETF price data, fetched once and cached to disk.

Why this exists rather than `yfinance`: the only thing we need is Yahoo's public
chart endpoint, which returns split/dividend-adjusted closes and volume in one
JSON document. Reading it directly costs ~60 lines, adds **no new dependency**
(D-17 forbids pip-installing into the anaconda base env), and pins the exact
request we made. The cached CSVs are committed, so every downstream number
traces to a file in the repo and every rerun is offline.

The cache is the source of truth. `load_prices` never hits the network unless a
ticker's CSV is missing, so a rerun cannot silently pick up a different data
vintage than the one the results were computed from.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "yahoo_daily"

CHART_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
             "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")

# The nine original SPDR select-sector ETFs. All have traded continuously since
# December 1998, so the panel is rectangular over any window we care about --
# unlike XLRE (listed 2015-10) and XLC (listed 2018-06), whose inclusion would
# make the universe change size mid-backtest. Stated, not silently chosen.
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

# Held out of the investable universe; used only as a passive benchmark, which
# is the comparison arXiv:2601.04062 never reports.
BENCHMARK = "SPY"


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def fetch_yahoo_daily(ticker: str, start: str, end: str,
                      retries: int = 4) -> pd.DataFrame:
    """One ticker of daily OHLCV + adjusted close. Network call.

    Yahoo intermittently truncates a chunked response mid-document; that shows
    up as `IncompleteRead`, not as an HTTP error, so it must be retried rather
    than trusted. Reading the body in full before parsing makes the failure
    surface here instead of as a silently short price series.
    """
    url = CHART_URL.format(ticker=ticker, p1=_epoch(start), p2=_epoch(end))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            payload = json.loads(body)
            break
        except Exception as exc:  # noqa: BLE001 — retry any transport failure
            last = exc
            if attempt == retries - 1:
                raise RuntimeError(
                    f"{ticker}: failed after {retries} attempts ({last!r})") from last
            time.sleep(2.0 * (attempt + 1))

    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "adjclose": result["indicators"]["adjclose"][0]["adjclose"],
        "volume": quote["volume"],
    }, index=pd.to_datetime(result["timestamp"], unit="s", utc=True)
        .tz_convert(None).normalize())
    df.index.name = "date"
    return df.dropna(subset=["adjclose"])


def download_universe(tickers: list[str], start: str, end: str,
                      cache_dir: Path = CACHE_DIR,
                      pause: float = 0.5, force: bool = False) -> None:
    """Populate the on-disk cache. Skips tickers already cached unless `force`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        path = cache_dir / f"{t}.csv"
        if path.exists() and not force:
            print(f"[cached] {t}")
            continue
        df = fetch_yahoo_daily(t, start, end)
        df.to_csv(path)
        print(f"[fetched] {t}: {len(df)} rows "
              f"{df.index[0].date()} .. {df.index[-1].date()}")
        time.sleep(pause)


def load_prices(tickers: list[str], field: str = "adjclose",
                cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """One field for several tickers, aligned on the common trading calendar.

    Offline: raises if a ticker is not cached rather than reaching for the
    network, so results can never be recomputed against a different vintage.
    """
    cols = {}
    for t in tickers:
        path = cache_dir / f"{t}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{t} not cached at {path}. Run scripts/fetch_etf_data.py first.")
        cols[t] = pd.read_csv(path, index_col="date", parse_dates=True)[field]
    df = pd.DataFrame(cols)
    # Inner join: a date is usable only if every asset traded on it. Dropping
    # partial rows is the conservative choice -- forward-filling a missing close
    # would invent a zero return on a day the asset did not trade.
    return df.dropna(how="any")


def daily_returns(tickers: list[str] | None = None,
                  cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Simple daily returns from adjusted closes (the backtest's return panel)."""
    px = load_prices(tickers or SECTOR_ETFS, "adjclose", cache_dir)
    return px.pct_change().dropna(how="any")


ANN_FACTOR_DAILY = 252
