# portfolio-lab

**Two independent replications of published "end-to-end beats two-stage" portfolio results.
Both came back null. This repository is mostly about why that is a finding and not a failure.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)
![Tests](https://img.shields.io/badge/tests-36%20offline-green.svg)

A backtesting lab built so that a null result means something. The engine is strictly causal and
*mechanically* tested for it, 1/N is the null hypothesis in every table, costs are always on,
turnover is always reported, and the minimum detectable effect is computed **before** the run
rather than discovered afterwards.

> **New here?** [**CONCEPTS.md**](CONCEPTS.md) is the 20-minute version: seven facts that form the
> spine of portfolio optimization, each paired with a number measured in this repo. It explains
> every result below, including why so many of them are nulls.

---

## Study 1 — Costa & Iyengar (2023), *Distributionally Robust End-to-End Portfolio Construction*

Their own out-of-sample window: **454 weeks**, 20 US large-cap equities and 8 Fama–French factors,
**12 strategies** — their 8 networks and 4 classical baselines of ours, on one backtest engine.

**Pipeline validation first.** Our engine reproduces their `ew_net` portfolio to a **maximum
absolute error of 1.4 × 10⁻¹⁷** across all 454 weeks — floating-point identity, not agreement.
Whatever the comparison shows afterwards, it isn't an artifact of two different backtesters.

**Finding: not one of the 12 is statistically distinguishable from equal weighting.**

| | Sharpe gross | 10 bps | 25 bps | weekly turnover |
|---|---:|---:|---:|---:|
| `dr_net_learn_theta` (their best) | 1.415 | 1.311 | 1.157 | 0.178 |
| `dr_net` | 1.315 | 1.238 | 1.124 | 0.124 |
| **1/N** | **1.052** | **1.044** | **1.034** | **0.010** |
| `dr_net_learn_gamma` | 1.023 | 0.675 | 0.150 | 0.450 |
| `po_net` (their two-stage baseline) | 0.899 | 0.508 | −0.078 | 0.576 |

The largest gap in the table — 1.415 vs. 1.052 — carries **HAC *p* = 0.180** and **studentized
block-bootstrap *p* = 0.202** (Ledoit–Wolf, 4999 resamples, block 5). Charge 10 bps and the same
pair goes to *p* = 0.33; sweeping the bootstrap block length from 1 to 20 moves that only within
0.34–0.45, so the conclusion is not an artifact of the dependence assumption. Standard errors on
annualized Sharpe differences run 0.19–0.32. The design cannot see effects this small, and says so.

**The one thing that *is* significant is a loss.** Across every test at every cost level, **zero**
strategies significantly beat 1/N; the only significant results are strategies significantly
*losing* to it — `po_net` at 10 bps (*p* = 0.011) and, at 25 bps, `po_net` (*p* < 0.001),
`dr_net_learn_gamma` (*p* = 0.001) and `dr_net_learn_delta` (*p* = 0.011).

**Why their headline gap appears.** Their reported end-to-end-over-two-stage advantage is measured
against `po_net`, and `po_net` turns over **0.576 per week against `dr_net_learn_theta`'s 0.178**.
Sort the table by turnover and it reproduces the ranking almost exactly. The gap is a property of
the baseline they chose, not of end-to-end training.

<sub>Raw: [`results/e2edro_cache_comparison.csv`](results/e2edro_cache_comparison.csv) ·
[`results/sharpe_tests.csv`](results/sharpe_tests.csv) (132 paired tests) ·
[`results/e2edro_cache_oos_returns.csv`](results/e2edro_cache_oos_returns.csv)</sub>

---

## Study 2 — a pre-registered re-test of arXiv:2601.04062 (SPO+)

That paper claims decision-focused (SPO+) training *consistently* beats predict-then-optimize on
real US ETF data. It released no code, and names neither a count nor a ticker for its universe.

**The design was frozen before any strategy ran** — [`audits/prereg-spo-retest.md`](audits/prereg-spo-retest.md),
written 2026-08-02. Universe, costs, protocol, hypotheses, and the decision rule are all fixed
there, and §4 states in advance what the design can and cannot see:

> `argmax r̂ᵀw` over a simplex is a corner solution, so this design **cannot reliably detect a
> Sharpe gap below ≈ 0.5**. The effect the paper reports is **+0.126** — roughly a quarter of that.
> Any gap below the threshold will be reported as "not detectable", **whichever direction the point
> estimate falls**, including if SPO+ appears to win.

Nine SPDR sector ETFs, 2016–2024, 108 monthly rebalances, 50 bps — the paper's own cost figure.

**All three pre-registered hypotheses came back negative.**

| | Sharpe gross | Sharpe net 50 bps | turnover / rebalance | effective positions |
|---|---:|---:|---:|---:|
| SPY buy & hold | 0.900 | **0.900** | 0.000 | 1.0 |
| risk parity | 0.840 | 0.829 | 0.016 | 8.4 |
| **1/N** | 0.831 | 0.822 | 0.013 | 9.0 |
| PtO | **0.915** (1st) | 0.490 (7th) | **1.000** | **1.00** |
| SPO+ | 0.858 | 0.447 (8th) | **1.000** | **1.00** |
| SPO+ (ℓ₂) | 0.701 | 0.333 | 0.611 | 3.8 |

- **SPO+ vs. PtO** — the paper's actual claim — measures **−0.057 gross** (*p* = 0.59) and
  **−0.043 net** (*p* = 0.67). Point estimate on the wrong side, and far inside the threshold
  declared in advance.
- **`argmax r̂ᵀw` over a simplex is a literal single-asset bet.** Measured effective positions
  `1/Σwᵢ²` = **1.00** and **100% one-way turnover**: the entire book moves into one ETF every
  month, on the strength of a noisy forecast. Both learned strategies go from best-in-table gross
  to worst-in-table net, while 1/N moves 0.831 → 0.822.
- **The only significant result in the study is a loss:** SPO+(ℓ₂) trails 1/N by **−0.489 net**
  (*p* = 0.012).

<sub>Raw: [`results/spo_retest_performance.csv`](results/spo_retest_performance.csv) ·
[`results/spo_retest_tests.csv`](results/spo_retest_tests.csv). Audit of the paper itself:
[`audits/spo-2601.04062.md`](audits/spo-2601.04062.md).</sub>

---

## Study 3 — why both nulls were structural

Two nulls on one historical path is weak evidence about methods and strong evidence about *power*.
So the third study stops asking "can anything beat 1/N" and asks the question a desk actually
faces: **given a signal of quality IC over breadth N at cost c, how far should the decision move?**
Signals of known IC are simulated, so the answer is measurable rather than inferred.

**Break-even IC — the lowest signal quality at which the best decision rule beats buy-and-hold:**

| cost | N = 10 | N = 25 | N = 50 | N = 100 |
|---|---:|---:|---:|---:|
| 0 bps | 0.02 | 0.02 | 0.02 | 0.02 |
| 10 bps | 0.10 | 0.05 | 0.05 | 0.05 |
| 25 bps | 0.20 | 0.10 | 0.10 | 0.10 |
| 50 bps | **> 0.20** | 0.20 | 0.20 | 0.20 |

Realistic monthly equity IC is **0.02–0.05**. Study 2 ran at 50 bps on a 9-asset universe — the
top-left corner, where the required skill exceeds anything an equity signal reliably has.
**Its null was structural, not a property of SPO+.** One table retro-explains every null in the
project.

A second measurement, on Grinold's IR ≈ IC·√BR: the ratio of achieved to predicted IR is *constant
across IC* — the law's first half is exact — but **falls in breadth**: 0.53 → 0.39 → 0.29 → 0.21
at N = 10 → 25 → 50 → 100. Long-only plus a fixed concentration budget is the binding constraint.
You cannot spend breadth you are not allowed to take positions in.

<sub>Raw: [`results/ic_breadth_stability_summary.csv`](results/ic_breadth_stability_summary.csv) ·
[`results/ic_breadth_stability_alpha_star.csv`](results/ic_breadth_stability_alpha_star.csv)</sub>

---

## How the lab avoids fooling itself

These are the parts that make the nulls worth reporting, and they are enforced in code, not in prose:

| discipline | enforcement |
|---|---|
| **No look-ahead** | `tests/test_polab.py::test_no_lookahead` **corrupts future observations and asserts past decisions are bit-identical**. Not a code review — a test. |
| **1/N is the null** | DeMiguel, Garlappi & Uppal (2009). It appears in every table, including the ones where it wins. |
| **Power before p-values** | The minimum detectable effect is computed from the actual return panel and written down *before* the run. A favourable point estimate below it is not a finding. |
| **Multiple testing** | `backtest.compare()` reports each strategy's Deflated Sharpe (Bailey & López de Prado) against the whole set of strategies tried, not just the reported one. |
| **Costs always on** | Default 10 bps proportional; turnover reported per strategy. Study 1 shows why this decides everything. |
| **Correct Sharpe inference** | Paired Ledoit–Wolf studentized block bootstrap (4999 resamples) with the HAC asymptotic alongside — not a naive *t*-test on overlapping, autocorrelated, fat-tailed returns. |
| **The implementation is validated separately from its performance** | SPO+ correctness is judged by oracle-losslessness, the surrogate's upper-bound property, and a finite-difference gradient check — decided in the pre-registration, so that "it lost" can never be confused with "it was broken". |
| **Negative results ship** | Three studies, three nulls, all written up. |

---

## Quickstart

```bash
pip install -r requirements.txt
```

The offline suite needs no data and no network — 36 tests on the synthetic generator:

```bash
python tests/test_polab.py
```

```bash
python scripts/run_ic_breadth_stability.py --quick    # Study 3, ~3 s
```

Studies 1 and 2 need data that is **fetched, not redistributed** (see [NOTICE.md](NOTICE.md)):

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

Every number in this README traces to a committed file under `results/`.

---

## Repository map

```
polab/
  backtest.py      strictly causal walk-forward engine; compare() with Deflated Sharpe
  baselines.py     1/N, min-variance, max-Sharpe, risk parity
  metrics.py       PSR / Deflated Sharpe (Bailey & López de Prado)
  sharpe_test.py   paired Ledoit-Wolf Sharpe-difference test, HAC + studentized block bootstrap
  spo.py           SPO+ surrogate and the decision layer, with closed forms where they exist
  stability.py     the no-trade band / partial-rebalance rule of Study 3
  simulate.py      signals of known IC over known breadth
  features.py      causal technical features (SMA, RSI, MACD, Bollinger)
  e2edro_io.py     unpickle their cached nets without their optional dependencies
  marketdata.py    Yahoo chart endpoint via stdlib urllib
audits/
  spo-2601.04062.md       audit of the paper
  prereg-spo-retest.md    pre-registration, frozen before the run
results/         every committed number, one CSV per study
notes.md         append-only iteration log, including the decisions that did not work
CONCEPTS.md      the seven facts, each with a number from this repo
```

---

## Caveats

- **Survivorship bias, twice.** The 20-stock universe was chosen in 2021 and is all survivors.
  The 9 sector ETFs all still trade. Fine for *method comparisons* on a fixed universe; not valid
  for absolute performance claims. Reported, not fixed.
- **Proportional costs only.** No market-impact model (square-root law), no borrow, no lot sizes.
- **One historical path.** This is the binding limitation of Studies 1 and 2, and precisely why
  Study 3 exists: on a single path a Sharpe difference carries SE 0.2–0.3, so the honest
  conclusion is almost always "cannot tell".
- **Study 2 uses our universe, not the paper's** — theirs is not recoverable from the text. No
  result here reproduces their numbers, and none is described as doing so.
- **Study 1 does not retrain their networks from scratch**; it runs off their shipped cache. The
  retraining machinery exists (`scripts/train_roll.py`, `notebooks/`) and was deliberately not
  run — see `notes.md`, iteration 6.

## License

MIT for the code in this repository — see [LICENSE](LICENSE). Third-party code and data are
fetched from their own sources under their own licenses; see [NOTICE.md](NOTICE.md).
