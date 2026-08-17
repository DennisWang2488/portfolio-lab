"""Data access: vendored E2E-DRO dataset, synthetic generator, yfinance download.

The primary real dataset is the one shipped with the vendored E2E-DRO repo
(Costa & Iyengar 2023): weekly returns 2000-01..2021-10 for 20 large-cap US
stocks plus 8 Fama-French factors. Using their exact data makes our replication
of their experiments apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "E2E-DRO"
ANN_FACTOR_WEEKLY = 52


def load_e2edro(kind: str = "asset") -> pd.DataFrame:
    """Weekly returns from the vendored E2E-DRO cache.

    kind='asset'  -> 20 US large-caps (1135 weeks, 2000-01-07..2021-10-01)
    kind='factor' -> 8 Fama-French factors, same index
    """
    path = VENDOR / "cache" / f"{kind}_weekly.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — re-clone https://github.com/Iyengar-Lab/E2E-DRO "
            f"into {VENDOR}")
    return pd.read_pickle(path)


def synthetic_returns(n_periods: int = 500, n_assets: int = 8,
                      seed: int = 0) -> pd.DataFrame:
    """Multivariate-normal returns with a random correlated covariance.

    For offline tests only — calibrated loosely to weekly equity magnitudes
    (ann. vol ~15-35%, small positive drift).
    """
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.15, 0.35, n_assets) / np.sqrt(ANN_FACTOR_WEEKLY)
    A = rng.normal(size=(n_assets, n_assets))
    corr = A @ A.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    cov = np.outer(vols, vols) * corr
    mu = rng.uniform(0.0, 0.10, n_assets) / ANN_FACTOR_WEEKLY
    R = rng.multivariate_normal(mu, cov, size=n_periods)
    idx = pd.date_range("2000-01-07", periods=n_periods, freq="W-FRI")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(R, index=idx, columns=cols)


def download_yfinance(tickers: list[str], start: str, end: str,
                      cache_path: str | Path | None = None) -> pd.DataFrame:
    """Daily adjusted-close returns via yfinance (network + `pip install yfinance`).

    Note: yfinance adjusted prices are survivorship-clean only for the tickers
    you pass — picking today's index members and backtesting 20 years is
    survivorship bias. Fine for method comparisons on a fixed universe; not
    fine for absolute performance claims.
    """
    import yfinance as yf  # deliberate lazy import: optional dependency
    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"]
    rets = px.pct_change().dropna(how="all")
    if cache_path is not None:
        rets.to_csv(cache_path)
    return rets
