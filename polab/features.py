"""Technical features for the SPO+ re-test.

The paper (arXiv:2601.04062, Table 1) lists its feature families — log returns,
SMA + price bias, RSI + MACD difference, Bollinger band width, volume
indicators — but **not one of their parameters** (audit `spo-2601.04062.md` §38).
The pre-registration inherited that gap. So the windows below are fixed HERE,
before any strategy was run, at the textbook defaults every charting package
ships with (RSI 14, MACD 12/26/9, Bollinger 20/2, SMA 20). They are never tuned
per window and never revisited after seeing a result; see `notes.md` iteration 7
for this recorded as a specification decision rather than a silent choice.

Every feature at row `t` is computed from data at or before `t`. Alignment to
targets is the caller's job — `spo_retest.build_panel` shifts targets forward
and drops any sample whose target is not fully realized by the decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed in advance. Textbook defaults; not searched.
SMA_WINDOW = 20
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLL_WINDOW, BOLL_K = 20, 2.0
VOL_WINDOW = 20
RET_LAGS = (1, 5, 21)

# Longest lookback any feature needs; callers must discard this warm-up.
WARMUP = max(SMA_WINDOW, RSI_PERIOD, MACD_SLOW + MACD_SIGNAL,
             BOLL_WINDOW, VOL_WINDOW, max(RET_LAGS)) + 5

FEATURE_NAMES = ([f"logret{k}" for k in RET_LAGS]
                 + ["sma_bias", "rsi", "macd_diff", "boll_width", "vol_ratio"])


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI, scaled to [-1, 1] so it sits on the other features' scale."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.fillna(50.0)          # flat stretch -> neutral, not NaN
    return rsi / 50.0 - 1.0


def asset_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """The 8 features for one asset. Index = `close`'s index."""
    logp = np.log(close)
    out = {f"logret{k}": logp.diff(k) for k in RET_LAGS}

    sma = close.rolling(SMA_WINDOW).mean()
    out["sma_bias"] = close / sma - 1.0

    out["rsi"] = _rsi(close)

    ema_f = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_diff"] = (macd - signal) / close      # price-scale free

    sd = close.rolling(BOLL_WINDOW).std(ddof=0)
    out["boll_width"] = (2.0 * BOLL_K * sd) / sma

    vsma = volume.rolling(VOL_WINDOW).mean()
    out["vol_ratio"] = np.log(volume.replace(0, np.nan) / vsma)

    return pd.DataFrame(out, index=close.index)[FEATURE_NAMES]


def panel_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Features for every asset, concatenated into one wide design matrix.

    Columns are `('<TICKER>', '<feature>')` flattened to `TICKER__feature`, so
    the linear predictor maps one cross-sectional feature vector to the whole
    return vector — the structure `spo.LinearPredictor` expects.
    """
    frames = []
    for tic in close.columns:
        f = asset_features(close[tic], volume[tic])
        f.columns = [f"{tic}__{c}" for c in f.columns]
        frames.append(f)
    wide = pd.concat(frames, axis=1)
    return wide.replace([np.inf, -np.inf], np.nan).dropna(how="any")
