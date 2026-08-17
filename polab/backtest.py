"""Strictly causal walk-forward backtest engine.

Design invariants (the whole point of this module):

1. NO LOOK-AHEAD. The weights applied over period t are a function of returns
   strictly before t (`returns.iloc[i - lookback : i]` for a rebalance at i).
   `tests/test_polab.py::test_no_lookahead` enforces this mechanically.
2. Transaction costs are charged on every unit of turnover against the
   *drifted* weights (holdings move with returns between rebalances).
3. One strategy = one pure function of the lookback window. Anything stateful
   (fitted models) must be wrapped so that fitting also only sees the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class BacktestResult:
    returns: pd.Series          # net per-period portfolio returns
    weights: pd.DataFrame       # weights in force at the START of each period
    turnover: pd.Series         # one-way turnover at each rebalance date
    cost_bps: float
    name: str = "strategy"
    extras: dict = field(default_factory=dict)

    def summary(self, ann_factor: int) -> dict:
        return {"name": self.name,
                **M.summary(self.returns, ann_factor, self.turnover)}


def drift(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Weights after one period of returns r, no trading."""
    grown = w * (1.0 + r)
    total = grown.sum()
    if total <= 0:  # portfolio wiped out; degenerate but keep defined
        return np.full_like(w, 1.0 / len(w))
    return grown / total


def walk_forward(returns: pd.DataFrame,
                 strategy,
                 lookback: int = 104,
                 rebalance_every: int = 4,
                 cost_bps: float = 10.0,
                 name: str = "strategy") -> BacktestResult:
    """Run `strategy` through an out-of-sample walk-forward loop.

    returns          : per-period asset returns, rows = time ascending.
    strategy         : callable(window: DataFrame) -> weight vector (sum 1).
    lookback         : estimation window length in periods.
    rebalance_every  : trade every k periods (e.g. 4 on weekly data ~ monthly).
    cost_bps         : proportional cost per unit traded, in basis points.
    """
    R = returns.dropna(how="any")
    n_periods, n_assets = R.shape
    if n_periods <= lookback:
        raise ValueError(f"need more than lookback={lookback} periods, got {n_periods}")
    rate = cost_bps / 1e4

    dates = R.index[lookback:]
    port_ret = np.zeros(len(dates))
    weights_hist = np.zeros((len(dates), n_assets))
    turnover_dates, turnover_vals = [], []

    w = None  # holdings entering the period
    for k, i in enumerate(range(lookback, n_periods)):
        cost = 0.0
        if (i - lookback) % rebalance_every == 0 or w is None:
            window = R.iloc[i - lookback:i]           # strictly past data
            target = np.asarray(strategy(window), dtype=float)
            if target.shape != (n_assets,) or not np.isclose(target.sum(), 1.0, atol=1e-6):
                raise ValueError(f"{name}: invalid weights at {R.index[i]}")
            prev = target if w is None else w
            traded = np.abs(target - prev).sum()
            cost = rate * traded  # first entry: prev == target, so this is 0
            turnover_dates.append(R.index[i])
            turnover_vals.append(traded / 2.0)        # one-way. some people report two-way (= this*2)
            w = target
        r = R.iloc[i].to_numpy()
        port_ret[k] = float(w @ r) - cost
        weights_hist[k] = w
        w = drift(w, r)

    return BacktestResult(
        returns=pd.Series(port_ret, index=dates, name=name),
        weights=pd.DataFrame(weights_hist, index=dates, columns=R.columns),
        turnover=pd.Series(turnover_vals, index=turnover_dates, name="turnover"),
        cost_bps=cost_bps,
        name=name,
    )


def cost_overlay(gross: pd.Series,
                 weights: pd.DataFrame,
                 asset_returns: pd.DataFrame,
                 cost_bps: float) -> tuple[pd.Series, pd.Series]:
    """Apply proportional transaction costs to an externally produced backtest.

    Uniform post-hoc treatment for comparing strategies whose engines differ
    (e.g. E2E-DRO's cached results vs ours): given gross per-period returns and
    the weights in force at the START of each period, charge `cost_bps` per
    unit traded relative to the drifted previous weights. The initial position
    is not charged (same convention for every strategy).

    Returns (net_returns, per_period_one_way_turnover).
    """
    R = asset_returns.loc[gross.index].to_numpy()
    W = weights.loc[gross.index].to_numpy()
    rate = cost_bps / 1e4
    net = gross.to_numpy().astype(float).copy()
    turn = np.zeros(len(net))
    for t in range(1, len(net)):
        drifted = drift(W[t - 1], R[t - 1])
        traded = np.abs(W[t] - drifted).sum()
        turn[t] = traded / 2.0
        net[t] -= rate * traded
    return (pd.Series(net, index=gross.index, name=gross.name),
            pd.Series(turn, index=gross.index, name="turnover"))


def compare(returns: pd.DataFrame,
            strategies: dict,
            ann_factor: int,
            **kwargs) -> pd.DataFrame:
    """Run several strategies under identical conditions; one metrics row each.

    Adds `dsr` — each strategy's Deflated Sharpe treating this whole comparison
    as the trial set, so the table itself accounts for multiple testing.
    """
    results = {nm: walk_forward(returns, fn, name=nm, **kwargs)
               for nm, fn in strategies.items()}
    trial_srs = [res.returns.mean() / res.returns.std(ddof=1)
                 for res in results.values()]
    rows = []
    for nm, res in results.items():
        row = res.summary(ann_factor)
        row["dsr"] = M.deflated_sharpe(res.returns, trial_srs)
        rows.append(row)
    return pd.DataFrame(rows).set_index("name")
