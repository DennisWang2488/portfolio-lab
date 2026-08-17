"""Roll-window bookkeeping for divide-and-conquer retraining of E2E-DRO.

`e2edro.e2e_net.net_roll_test` loops over `n_roll` windows. Each iteration
reloads the SAME saved init state and re-fits the prediction layer to OLS on
that window's training data, so **no state carries between windows** — they are
independent jobs that can run in separate processes, in parallel, and be
resumed individually.

This module reproduces their split arithmetic in pure pandas/numpy (no torch,
no cvxpylayers) so the plan can be verified offline and each worker knows where
its slice belongs in the assembled backtest.

Reference (their code, verbatim logic):
    win_size = init_split[1] / n_roll
    split[0] = init_split[0] + win_size * i
    split[1] = win_size            if i < n_roll-1  else  1 - split[0]
    numel    = round(n_total * cumsum(split))
    train    = data[:numel[0]]
    test     = data[numel[0]-n_obs : numel[1]]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RollWindow:
    index: int              # roll number, 0-based
    split: tuple            # (train_frac, test_frac) passed to split_update
    train_end: int          # numel[0]: rows of training data
    test_end: int           # numel[1]
    n_train_windows: int    # sliding windows the trainer iterates per epoch
    n_test_windows: int     # OOS weeks this roll contributes
    offset: int             # where this roll's returns start in the full backtest


def _numel(n_total: int, split) -> list[int]:
    return [round(v) for v in n_total * np.cumsum(split)]


def plan(n_total: int = 1134, init_split=(0.6, 0.4), n_roll: int = 4,
         n_obs: int = 104, perf_period: int = 13) -> list[RollWindow]:
    """Enumerate the roll windows exactly as their loop would."""
    win_size = init_split[1] / n_roll
    out, offset = [], 0
    for i in range(n_roll):
        s0 = init_split[0] + win_size * i
        s1 = win_size if i < n_roll - 1 else 1 - s0
        numel = _numel(n_total, (s0, s1))
        n_test = (numel[1] - (numel[0] - n_obs)) - n_obs
        # SlidingWindow(train, n_obs, perf_period) yields this many batches
        n_train = numel[0] - n_obs - perf_period
        out.append(RollWindow(i, (s0, s1), numel[0], numel[1],
                              n_train, n_test, offset))
        offset += n_test
    return out


def total_test_windows(n_total: int = 1134, init_split=(0.6, 0.4),
                       n_obs: int = 104) -> int:
    """Length of the assembled backtest (their `pc.backtest(...)` sizing)."""
    numel = _numel(n_total, init_split)
    return (numel[1] - (numel[0] - n_obs)) - n_obs


def summary(rolls: list[RollWindow], epochs: int = 50) -> str:
    lines = [f"{'roll':>4} {'train_win':>9} {'test_win':>8} {'offset':>6} "
             f"{'steps@%d ep' % epochs:>12}"]
    for r in rolls:
        lines.append(f"{r.index:>4} {r.n_train_windows:>9} {r.n_test_windows:>8} "
                     f"{r.offset:>6} {r.n_train_windows * epochs:>12,}")
    tot_steps = sum(r.n_train_windows for r in rolls) * epochs
    lines.append(f"{'ALL':>4} {'':>9} {sum(r.n_test_windows for r in rolls):>8} "
                 f"{'':>6} {tot_steps:>12,}")
    return "\n".join(lines)
