# portfolio-lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)

Backtester I wrote so a null result is actually usable: no look-ahead (there’s a test that corrupts the future and checks past weights don’t change), 1/N in every table, costs on, turnover reported, and I write down the minimum detectable effect before I run anything.

Two replications of published end-to-end portfolio papers, plus a small sim for when a signal of known IC can beat buy-and-hold.

[CONCEPTS.md](CONCEPTS.md) is the short version if you just want the intuition.

## Costa & Iyengar (2023)

*Distributionally Robust End-to-End Portfolio Construction.* I put their published OOS window (454 weeks, 20 US large-caps, 8 Fama–French factors) through this engine — their 8 cached networks and 4 classical baselines.

`ew_net` matches to 1.4e-17 across all 454 weeks, so this isn’t a backtester mismatch.

On this window none of the twelve is statistically distinguishable from equal weighting.

| | Sharpe gross | 10 bps | 25 bps | weekly turnover |
|---|---:|---:|---:|---:|
| `dr_net_learn_theta` (best network) | 1.415 | 1.311 | 1.157 | 0.178 |
| `dr_net` | 1.315 | 1.238 | 1.124 | 0.124 |
| 1/N | 1.052 | 1.044 | 1.034 | 0.010 |
| `dr_net_learn_gamma` | 1.023 | 0.675 | 0.150 | 0.450 |
| `po_net` (two-stage baseline) | 0.899 | 0.508 | −0.078 | 0.576 |

1.415 vs 1.052 is HAC *p* = 0.180, block-bootstrap *p* = 0.202 (Ledoit–Wolf, 4999 resamples, block 5). At 10 bps that pair is *p* = 0.33; changing the block length from 1 to 20 keeps it in 0.34–0.45. SEs on annualized Sharpe diffs are 0.19–0.32, so this window just isn’t long enough to call a gap of that size.

Nothing beats 1/N at conventional significance. After costs a few lose to it: `po_net` at 10 bps (*p* = 0.011); at 25 bps, `po_net` (*p* < 0.001), `dr_net_learn_gamma` (*p* = 0.001), `dr_net_learn_delta` (*p* = 0.011).

Their end-to-end vs `po_net` comparison lines up with a big turnover difference (0.178 / week vs 0.576). Ranking by turnover is almost the same ranking as net Sharpe on this panel.

[`results/e2edro_cache_comparison.csv`](results/e2edro_cache_comparison.csv), [`results/sharpe_tests.csv`](results/sharpe_tests.csv), [`results/e2edro_cache_oos_returns.csv`](results/e2edro_cache_oos_returns.csv)

## SPO+ — arXiv:2601.04062

Wang & Hasuike report SPO+ beating predict-then-optimize on US ETFs. No code, and they don’t name the tickers, so I can’t reproduce their tables. I pre-registered a design on 9 sector SPDRs instead: [`audits/prereg-spo-retest.md`](audits/prereg-spo-retest.md) (2026-08-02).

Before anything ran I wrote down that `argmax r̂ᵀw` over a simplex can’t really see a Sharpe gap below about 0.5 on this panel. The paper reports +0.126. Anything smaller than that threshold I treat as not detectable, either direction.

9 SPDR sector ETFs, 2016–2024, 108 monthly rebalances, 50 bps (their cost figure).

| | Sharpe gross | Sharpe net 50 bps | turnover / rebalance | effective positions |
|---|---:|---:|---:|---:|
| SPY buy & hold | 0.900 | 0.900 | 0.000 | 1.0 |
| risk parity | 0.840 | 0.829 | 0.016 | 8.4 |
| 1/N | 0.831 | 0.822 | 0.013 | 9.0 |
| PtO | 0.915 | 0.490 | 1.000 | 1.00 |
| SPO+ | 0.858 | 0.447 | 1.000 | 1.00 |
| SPO+ (ℓ₂) | 0.701 | 0.333 | 0.611 | 3.8 |

SPO+ vs PtO: −0.057 gross (*p* = 0.59), −0.043 net (*p* = 0.67). Well inside the threshold I set.

`argmax r̂ᵀw` on the simplex just picks one name — effective positions `1/Σwᵢ²` = 1.00, turnover 1.00 every month. After 50 bps both learned rules sit near the bottom; 1/N goes 0.831 → 0.822. SPO+(ℓ₂) vs 1/N is −0.489 net (*p* = 0.012).

This is my universe, not theirs.

[`results/spo_retest_performance.csv`](results/spo_retest_performance.csv), [`results/spo_retest_tests.csv`](results/spo_retest_tests.csv). Notes: [`audits/spo-2601.04062.md`](audits/spo-2601.04062.md).

## IC × breadth × cost

1 and 2 are each one historical path. This one is the more general question: if you have a signal of quality IC over N names at cost c, how much should you actually trade?

Break-even IC (lowest IC where the best rule here beats buy-and-hold):

| cost | N = 10 | N = 25 | N = 50 | N = 100 |
|---|---:|---:|---:|---:|
| 0 bps | 0.02 | 0.02 | 0.02 | 0.02 |
| 10 bps | 0.10 | 0.05 | 0.05 | 0.05 |
| 25 bps | 0.20 | 0.10 | 0.10 | 0.10 |
| 50 bps | > 0.20 | 0.20 | 0.20 | 0.20 |

Monthly equity IC is usually quoted around 0.02–0.05. Study 2 is basically the high-cost, low-N corner, where you need much more than that. That’s one reason a small Sharpe gap is hard to see there.

Grinold’s IR ≈ IC·√BR: the ratio of achieved to predicted IR is pretty flat across IC, but it drops with N (0.53 → 0.39 → 0.29 → 0.21 at 10 / 25 / 50 / 100). Long-only plus a concentration budget — you can’t spend breadth you aren’t allowed to take.

[`results/ic_breadth_stability_summary.csv`](results/ic_breadth_stability_summary.csv), [`results/ic_breadth_stability_alpha_star.csv`](results/ic_breadth_stability_alpha_star.csv)

## Run

```bash
pip install -r requirements.txt
python tests/test_polab.py                              # 36 tests, no data, no network
python scripts/run_ic_breadth_stability.py --quick      # ~3 s
```

Studies 1 and 2 need data I don’t redistribute ([NOTICE.md](NOTICE.md)):

```bash
bash scripts/setup_vendor.sh          # E2E-DRO + their cache
python scripts/fetch_etf_data.py      # Yahoo sector ETFs
python scripts/run_baselines.py
python scripts/compare_e2edro_cache.py
python scripts/sharpe_tests.py        # ~2 min
python scripts/run_spo_retest.py      # ~3 min
```

Numbers in this README are from committed files under `results/`.

Also in here: `backtest.compare()` does Deflated Sharpe against the whole set of strategies, not just the one you wanted to report. SPO+ correctness (oracle-losslessness, upper bound, finite-diff gradient) was fixed in the pre-reg so “it lost” isn’t “it was broken”.

## Caveats

Survivorship twice — the 20 names were picked in 2021 and they’re all still around; the 9 sector ETFs still trade. Fine for comparing methods on a fixed universe, not for “this Sharpe is real”.

Proportional costs only. No impact, no borrow, no lots.

Studies 1 and 2 are one path each. Sharpe diffs on those paths have SE ~0.2–0.3, so most of the time you just can’t tell. That’s why the IC table exists.

Study 2 is not their universe. Study 1 runs their shipped cache; I didn’t retrain. The training scripts are there (`scripts/train_roll.py`, `notebooks/`) if someone wants to. See `notes.md` iteration 6.

MIT for this repo ([LICENSE](LICENSE)). Third-party code and data stay with their own licenses — [NOTICE.md](NOTICE.md).
