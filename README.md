# portfolio-lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)
![Tests](https://img.shields.io/badge/tests-36%20offline-green.svg)

Two replications of published "end-to-end beats two-stage" portfolio results. Both came back null. A third study measures when a signal of known IC can beat buy-and-hold, which is why those nulls happen.

The backtester is causal and tested for it. 1/N is in every table. Costs are always on. Turnover is always reported. Minimum detectable effect is computed before the run.

[CONCEPTS.md](CONCEPTS.md) is the short version: seven facts about portfolio optimization, each with a number from this repo.

## Study 1 — Costa & Iyengar (2023)

Their out-of-sample window: 454 weeks, 20 US large-caps and 8 Fama–French factors, 12 strategies (their 8 networks plus 4 classical baselines), one engine.

The engine matches their `ew_net` portfolio to a maximum absolute error of 1.4 × 10⁻¹⁷ across all 454 weeks. Whatever follows is not a backtester mismatch.

None of the 12 is distinguishable from equal weighting.

| | Sharpe gross | 10 bps | 25 bps | weekly turnover |
|---|---:|---:|---:|---:|
| `dr_net_learn_theta` (their best) | 1.415 | 1.311 | 1.157 | 0.178 |
| `dr_net` | 1.315 | 1.238 | 1.124 | 0.124 |
| 1/N | 1.052 | 1.044 | 1.034 | 0.010 |
| `dr_net_learn_gamma` | 1.023 | 0.675 | 0.150 | 0.450 |
| `po_net` (their two-stage baseline) | 0.899 | 0.508 | −0.078 | 0.576 |

The largest gap, 1.415 vs. 1.052, has HAC *p* = 0.180 and studentized block-bootstrap *p* = 0.202 (Ledoit–Wolf, 4999 resamples, block 5). At 10 bps the same pair is *p* = 0.33. Sweeping the bootstrap block from 1 to 20 keeps that in 0.34–0.45. Standard errors on annualized Sharpe differences are 0.19–0.32.

Zero strategies significantly beat 1/N. The significant results are losses: `po_net` at 10 bps (*p* = 0.011); at 25 bps, `po_net` (*p* < 0.001), `dr_net_learn_gamma` (*p* = 0.001), and `dr_net_learn_delta` (*p* = 0.011).

Their headline end-to-end-over-two-stage gap is measured against `po_net`, which turns over 0.576 per week vs. 0.178 for `dr_net_learn_theta`. Sort the table by turnover and the ranking is almost the same.

Raw: [`results/e2edro_cache_comparison.csv`](results/e2edro_cache_comparison.csv) · [`results/sharpe_tests.csv`](results/sharpe_tests.csv) · [`results/e2edro_cache_oos_returns.csv`](results/e2edro_cache_oos_returns.csv)

## Study 2 — pre-registered re-test of arXiv:2601.04062 (SPO+)

The paper claims SPO+ consistently beats predict-then-optimize on US ETF data. No code, and the universe is not named.

Design frozen before any strategy ran: [`audits/prereg-spo-retest.md`](audits/prereg-spo-retest.md) (2026-08-02). Section 4 says in advance that `argmax r̂ᵀw` over a simplex cannot reliably detect a Sharpe gap below about 0.5. The paper reports +0.126. Anything below the threshold is "not detectable", including if SPO+ appears to win.

Nine SPDR sector ETFs, 2016–2024, 108 monthly rebalances, 50 bps (the paper's cost figure).

All three pre-registered hypotheses came back negative.

| | Sharpe gross | Sharpe net 50 bps | turnover / rebalance | effective positions |
|---|---:|---:|---:|---:|
| SPY buy & hold | 0.900 | 0.900 | 0.000 | 1.0 |
| risk parity | 0.840 | 0.829 | 0.016 | 8.4 |
| 1/N | 0.831 | 0.822 | 0.013 | 9.0 |
| PtO | 0.915 (1st) | 0.490 (7th) | 1.000 | 1.00 |
| SPO+ | 0.858 | 0.447 (8th) | 1.000 | 1.00 |
| SPO+ (ℓ₂) | 0.701 | 0.333 | 0.611 | 3.8 |

- SPO+ vs. PtO (the paper's claim): −0.057 gross (*p* = 0.59), −0.043 net (*p* = 0.67). Wrong side, well inside the pre-registered threshold.
- `argmax r̂ᵀw` over a simplex is a single-asset bet. Effective positions `1/Σwᵢ²` = 1.00 and 100% one-way turnover: the whole book moves into one ETF every month. Both learned strategies go from best-in-table gross to worst-in-table net. 1/N goes 0.831 → 0.822.
- Only significant result: SPO+(ℓ₂) trails 1/N by −0.489 net (*p* = 0.012).

Raw: [`results/spo_retest_performance.csv`](results/spo_retest_performance.csv) · [`results/spo_retest_tests.csv`](results/spo_retest_tests.csv). Paper audit: [`audits/spo-2601.04062.md`](audits/spo-2601.04062.md).

## Study 3 — when a signal can beat buy-and-hold

Studies 1 and 2 are one historical path. This one asks a different question: given a signal of quality IC over breadth N at cost c, how far should the decision move?

Break-even IC — lowest signal quality at which the best decision rule beats buy-and-hold:

| cost | N = 10 | N = 25 | N = 50 | N = 100 |
|---|---:|---:|---:|---:|
| 0 bps | 0.02 | 0.02 | 0.02 | 0.02 |
| 10 bps | 0.10 | 0.05 | 0.05 | 0.05 |
| 25 bps | 0.20 | 0.10 | 0.10 | 0.10 |
| 50 bps | > 0.20 | 0.20 | 0.20 | 0.20 |

Realistic monthly equity IC is 0.02–0.05. Study 2 ran at 50 bps on 9 assets — the top-left corner, where the required skill is higher than equity signals usually have.

On Grinold's IR ≈ IC·√BR, the ratio of achieved to predicted IR is constant across IC but falls in breadth: 0.53 → 0.39 → 0.29 → 0.21 at N = 10 → 25 → 50 → 100. Long-only plus a fixed concentration budget is the binding constraint.

Raw: [`results/ic_breadth_stability_summary.csv`](results/ic_breadth_stability_summary.csv) · [`results/ic_breadth_stability_alpha_star.csv`](results/ic_breadth_stability_alpha_star.csv)

## What the code enforces

| | |
|---|---|
| No look-ahead | `tests/test_polab.py::test_no_lookahead` corrupts future observations and asserts past decisions are bit-identical. |
| 1/N is the null | DeMiguel, Garlappi & Uppal (2009). In every table. |
| Power first | Minimum detectable effect from the actual return panel, written down before the run. |
| Multiple testing | `backtest.compare()` reports Deflated Sharpe (Bailey & López de Prado) against the full set of strategies tried. |
| Costs always on | Default 10 bps proportional. Turnover reported per strategy. |
| Sharpe inference | Paired Ledoit–Wolf studentized block bootstrap (4999 resamples) plus HAC, not a *t*-test on overlapping fat-tailed returns. |
| Implementation checked separately | SPO+ is judged by oracle-losslessness, the surrogate's upper-bound property, and a finite-difference gradient check — fixed in the pre-registration. |

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
  e2edro_io.py     unpickle their cached nets without their optional dependencies
  marketdata.py    Yahoo chart endpoint via stdlib urllib
audits/
  spo-2601.04062.md       paper audit
  prereg-spo-retest.md    pre-registration
results/         one CSV per study
notes.md         iteration log
CONCEPTS.md      seven facts, each with a number from this repo
```

## Limitations

- Survivorship, twice. The 20-stock universe was chosen in 2021 and is all survivors. The 9 sector ETFs all still trade. Fine for method comparisons on a fixed universe, not for absolute performance.
- Proportional costs only. No market-impact model, no borrow, no lot sizes.
- One historical path for Studies 1 and 2. A Sharpe difference on that path has SE 0.2–0.3, so the usual conclusion is "cannot tell". That is why Study 3 exists.
- Study 2 uses our universe, not the paper's — theirs is not recoverable from the text. Nothing here reproduces their numbers.
- Study 1 does not retrain their networks. It runs their shipped cache. Retraining exists (`scripts/train_roll.py`, `notebooks/`) and was not run — see `notes.md`, iteration 6.

## License

MIT for this repo — see [LICENSE](LICENSE). Third-party code and data are fetched from their own sources under their own licenses; see [NOTICE.md](NOTICE.md).
