# portfolio-lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)
![Tests](https://img.shields.io/badge/tests-36%20offline-green.svg)

Replications of two published end-to-end portfolio studies, plus a simulation that relates signal quality, breadth, and costs to when a decision rule can beat buy-and-hold.

The backtester is causal and tested for look-ahead. Equal-weight (1/N) is included in every comparison. Transaction costs and turnover are reported throughout. The minimum detectable effect is computed from the return panel before each run.

[CONCEPTS.md](CONCEPTS.md) is a short background note: seven facts about portfolio optimization, each with a number from this repo.

## Study 1 — Costa & Iyengar (2023)

Replication of Costa & Iyengar, *Distributionally Robust End-to-End Portfolio Construction*, on their published out-of-sample window: 454 weeks, 20 US large-caps, and 8 Fama–French factors. We evaluate their eight cached networks and four classical baselines on one engine.

As a pipeline check, the engine matches their `ew_net` weights to a maximum absolute error of 1.4 × 10⁻¹⁷ across all 454 weeks.

On this window, none of the twelve strategies is statistically distinguishable from equal weighting.

| | Sharpe gross | 10 bps | 25 bps | weekly turnover |
|---|---:|---:|---:|---:|
| `dr_net_learn_theta` (best network) | 1.415 | 1.311 | 1.157 | 0.178 |
| `dr_net` | 1.315 | 1.238 | 1.124 | 0.124 |
| 1/N | 1.052 | 1.044 | 1.034 | 0.010 |
| `dr_net_learn_gamma` | 1.023 | 0.675 | 0.150 | 0.450 |
| `po_net` (two-stage baseline) | 0.899 | 0.508 | −0.078 | 0.576 |

The largest Sharpe gap in the table (1.415 vs. 1.052) has HAC *p* = 0.180 and studentized block-bootstrap *p* = 0.202 (Ledoit–Wolf, 4999 resamples, block 5). At 10 bps the same pair is *p* = 0.33. Varying the bootstrap block from 1 to 20 leaves that in 0.34–0.45. Standard errors on annualized Sharpe differences are 0.19–0.32, so this window has limited power for gaps of this size.

Relative to 1/N we do not find significant outperformance at any cost level. After costs, a few strategies underperform: `po_net` at 10 bps (*p* = 0.011); at 25 bps, `po_net` (*p* < 0.001), `dr_net_learn_gamma` (*p* = 0.001), and `dr_net_learn_delta` (*p* = 0.011).

The paper’s comparison of end-to-end networks to `po_net` coincides with a large turnover difference (`po_net` 0.576 per week vs. 0.178 for `dr_net_learn_theta`). On this panel, ranking methods by turnover is close to ranking them by net Sharpe.

Raw: [`results/e2edro_cache_comparison.csv`](results/e2edro_cache_comparison.csv) · [`results/sharpe_tests.csv`](results/sharpe_tests.csv) · [`results/e2edro_cache_oos_returns.csv`](results/e2edro_cache_oos_returns.csv)

## Study 2 — pre-registered re-test of arXiv:2601.04062 (SPO+)

Wang & Hasuike report that SPO+ outperforms predict-then-optimize on US ETF data. Code and the exact universe are not released, so this is a pre-registered implementation on a documented panel rather than a numerical reproduction of their tables.

Design: [`audits/prereg-spo-retest.md`](audits/prereg-spo-retest.md) (2026-08-02). Before any strategy ran, we recorded that `argmax r̂ᵀw` over a simplex has limited power for Sharpe gaps below about 0.5 on this panel. The paper reports +0.126, which is below that threshold. Results smaller than the minimum detectable effect are reported as not detectable in either direction.

Nine SPDR sector ETFs, 2016–2024, 108 monthly rebalances, 50 bps (the cost figure used in the paper).

| | Sharpe gross | Sharpe net 50 bps | turnover / rebalance | effective positions |
|---|---:|---:|---:|---:|
| SPY buy & hold | 0.900 | 0.900 | 0.000 | 1.0 |
| risk parity | 0.840 | 0.829 | 0.016 | 8.4 |
| 1/N | 0.831 | 0.822 | 0.013 | 9.0 |
| PtO | 0.915 | 0.490 | 1.000 | 1.00 |
| SPO+ | 0.858 | 0.447 | 1.000 | 1.00 |
| SPO+ (ℓ₂) | 0.701 | 0.333 | 0.611 | 3.8 |

- SPO+ vs. PtO: −0.057 gross (*p* = 0.59), −0.043 net (*p* = 0.67). The point estimate is small relative to the pre-registered minimum detectable effect.
- On the simplex, `argmax r̂ᵀw` concentrates: effective positions `1/Σwᵢ²` = 1.00 and one-way turnover of 1.00, so the portfolio is typically a single ETF each month. After 50 bps both learned rules rank near the bottom of the table; 1/N moves from 0.831 to 0.822.
- SPO+(ℓ₂) trails 1/N by −0.489 net (*p* = 0.012).

Raw: [`results/spo_retest_performance.csv`](results/spo_retest_performance.csv) · [`results/spo_retest_tests.csv`](results/spo_retest_tests.csv). Notes on the paper: [`audits/spo-2601.04062.md`](audits/spo-2601.04062.md).

## Study 3 — signal quality, breadth, and costs

Studies 1 and 2 are each one historical path. This study asks a more general question: given a signal of quality IC over breadth *N* at cost *c*, how large a position change is justified?

Break-even IC — the lowest signal quality at which the best rule in the set beats buy-and-hold:

| cost | N = 10 | N = 25 | N = 50 | N = 100 |
|---|---:|---:|---:|---:|
| 0 bps | 0.02 | 0.02 | 0.02 | 0.02 |
| 10 bps | 0.10 | 0.05 | 0.05 | 0.05 |
| 25 bps | 0.20 | 0.10 | 0.10 | 0.10 |
| 50 bps | > 0.20 | 0.20 | 0.20 | 0.20 |

Typical monthly equity IC estimates are around 0.02–0.05. Study 2 sits near the high-cost, low-breadth corner of the table, where the break-even IC is well above that range. That is one reason a small Sharpe gap is hard to detect there.

On Grinold’s IR ≈ IC·√BR, the ratio of achieved to predicted IR is roughly constant across IC but declines with breadth: 0.53 → 0.39 → 0.29 → 0.21 at N = 10 → 25 → 50 → 100, consistent with a long-only concentration constraint.

Raw: [`results/ic_breadth_stability_summary.csv`](results/ic_breadth_stability_summary.csv) · [`results/ic_breadth_stability_alpha_star.csv`](results/ic_breadth_stability_alpha_star.csv)

## What the code enforces

| | |
|---|---|
| No look-ahead | `tests/test_polab.py::test_no_lookahead` corrupts future observations and asserts past decisions are bit-identical. |
| 1/N as a baseline | DeMiguel, Garlappi & Uppal (2009). Included in every table. |
| Power first | Minimum detectable effect from the actual return panel, written down before the run. |
| Multiple testing | `backtest.compare()` reports Deflated Sharpe (Bailey & López de Prado) against the full set of strategies tried. |
| Costs | Default 10 bps proportional. Turnover reported per strategy. |
| Sharpe inference | Paired Ledoit–Wolf studentized block bootstrap (4999 resamples) and a HAC test, for overlapping, autocorrelated returns. |
| Implementation checks | SPO+ is checked by oracle-losslessness, the surrogate’s upper-bound property, and a finite-difference gradient check, as specified in the pre-registration. |

## Quickstart

```bash
pip install -r requirements.txt
```

Offline suite, no data, no network (36 tests on the synthetic generator):

```bash
python tests/test_polab.py
```

```bash
python scripts/run_ic_breadth_stability.py --quick    # Study 3, ~3 s
```

Studies 1 and 2 need data that is fetched, not redistributed (see [NOTICE.md](NOTICE.md)):

```bash
bash scripts/setup_vendor.sh        # Study 1: clone E2E-DRO (Apache 2.0) + their cache
```

```bash
python scripts/fetch_etf_data.py    # Study 2: daily sector-ETF panel from Yahoo
```

```bash
python scripts/run_baselines.py          # 4 classical baselines on the weekly panel
python scripts/compare_e2edro_cache.py   # Study 1: their nets + ours, cost stress
python scripts/sharpe_tests.py           # Study 1: 132 paired Sharpe tests (~2 min)
python scripts/run_spo_retest.py         # Study 2: the pre-registered re-test (~3 min)
```

Every number in this README comes from a committed file under `results/`.

## Layout

```
polab/
  backtest.py      walk-forward engine; compare() with Deflated Sharpe
  baselines.py     1/N, min-variance, max-Sharpe, risk parity
  metrics.py       PSR / Deflated Sharpe
  sharpe_test.py   paired Ledoit–Wolf Sharpe-difference test (HAC + block bootstrap)
  spo.py           SPO+ surrogate and decision layer
  stability.py     no-trade band / partial-rebalance rule (Study 3)
  simulate.py      signals of known IC over known breadth
  features.py      causal technical features
  e2edro_io.py     unpickle cached E2E-DRO nets without optional dependencies
  marketdata.py    Yahoo chart endpoint via stdlib urllib
audits/
  spo-2601.04062.md       notes on arXiv:2601.04062
  prereg-spo-retest.md    pre-registration
results/         one CSV per study
notes.md         iteration log
CONCEPTS.md      seven facts, each with a number from this repo
```

## Limitations

- Survivorship. The 20-stock universe was fixed in 2021 and consists of names that still trade; the nine sector ETFs also still trade. That is acceptable for comparing methods on a fixed universe, not for claims about absolute performance.
- Costs are proportional only. No market-impact, borrow, or lot-size model.
- Studies 1 and 2 are single historical paths. Sharpe differences on those paths have standard errors of about 0.2–0.3. Study 3 is there to put those sample sizes in context.
- Study 2 uses a sector-ETF universe specified here. The original paper does not identify its universe, so we do not claim to match its reported numbers.
- Study 1 evaluates the authors’ published cached networks rather than retraining. Training scripts exist (`scripts/train_roll.py`, `notebooks/`) but were not used for the numbers above; see `notes.md`, iteration 6.

## License

MIT for this repo — see [LICENSE](LICENSE). Third-party code and data are fetched from their own sources under their own licenses; see [NOTICE.md](NOTICE.md).
