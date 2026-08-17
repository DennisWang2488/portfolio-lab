# Pre-registration — SPO+ re-test on US sector ETFs

导读（中文）：这份文件在**跑任何策略之前**写死。目的是防止事后挑参数、挑窗口、挑指标。核心是第 4 节：这个设计的最小可检出效应是 **0.54** 年化 Sharpe，而 arXiv:2601.04062 宣称的差距是 **0.126** —— 差约 4 倍。所以我们**提前声明**：任何小于 0.5 的差距一律报告为「检测不出」，不管点估计站在哪一边。

Written 2026-08-02, **before any strategy was run**. Companion to
[`spo-2601.04062.md`](spo-2601.04062.md), which audits the paper itself.

The audit (§5 there) showed the paper's design lacks the power to detect its own
reported effect. That criticism only has force if we hold ourselves to it first.
Hence this document: the design, the test, and the decision rule are fixed here,
and §4 states in advance what this design can and cannot see.

---

## 1. Data — fixed

| | |
|---|---|
| Universe | 9 original SPDR select-sector ETFs: XLB XLE XLF XLI XLK XLP XLU XLV XLY |
| Benchmark (not investable) | SPY |
| Source | Yahoo chart endpoint, adjusted closes + volume; cached in `data/yahoo_daily/*.csv` and committed |
| Fetched window | 2014-01-02 … 2024-12-31 (2768 trading days, rectangular across all 10) |
| Evaluation window | 2016-01-01 … 2024-12-31, matching the paper's reported period |
| 2014–2015 | feature warm-up and the first training windows only; never evaluated |

**Why this universe.** The paper names neither a count nor a ticker, so its
universe cannot be recovered. These nine have traded continuously since 1998, so
the panel does not change size mid-backtest (XLRE listed 2015-10 and XLC
2018-06 would break that). **This is our universe, not theirs**, and no result
here will be described as reproducing their numbers.

**Survivorship.** All ten still trade. The universe is survivorship-biased by
construction; sector ETFs are much less exposed than a stock universe (none was
delisted or merged over the window) but the bias is not zero. Reported, not fixed.

---

## 2. Strategies — fixed

Everything shares one decision layer and one backtest engine, so differences are
attributable to the training loss and nothing else.

| | |
|---|---|
| **1/N** | the null. DeMiguel et al. (2009). Absent from the paper. |
| **SPY buy-and-hold** | passive benchmark. Also absent from the paper. |
| **PtO** | linear predictor trained by MSE → `argmax_{w∈W} r̂ᵀw`. Their eq. (16)–(17). |
| **SPO+** | same linear predictor, same layer, trained by the SPO+ surrogate, their eq. (4). |
| **SPO+ (ℓ₂)** | same, with `λ‖w‖₂²` in the layer (their eq. 6) — the variant that actually diversifies. |
| **min-variance / risk parity / max-Sharpe** | existing `polab` classical baselines, for context. |

`W = {w : w ≥ 0, 1ᵀw = 1}` — long-only, fully invested. **Declared here because
the paper never defines it**, and under a max-return LP this choice *is* the
strategy.

Costs: proportional **50 bps** on `‖w_t − w_{t−1}‖₁`, their value, applied
uniformly. Turnover reported per strategy — the paper reports none.

Protocol: monthly rebalance; train on the trailing 12 months; strictly causal,
enforced by the existing `test_no_lookahead`. **No per-window hyperparameter
search** (the paper's ~108 Optuna searches are an uncorrected selection process);
learning rate and epochs are fixed in advance at a single value and reported.

---

## 3. Hypotheses — fixed

- **H1 (primary).** SPO+ attains a higher net-of-cost annualized Sharpe than
  **1/N** over 2016–2024.
- **H2 (the paper's actual claim).** SPO+ beats **PtO** — same layer, same
  predictor, MSE loss instead of SPO+.
- **H3.** Any advantage in H1/H2 survives the 50 bps cost, i.e. is not a gross
  artifact reversed by turnover (this is what killed the ranking in iteration 2).

Primary test for all three: paired Ledoit–Wolf studentized block bootstrap
(`polab/sharpe_test.py`, 4999 resamples, block 5), two-sided at 5%, with the HAC
asymptotic reported alongside. Secondary: Deflated Sharpe across the whole
strategy set, for the selection-bias side.

---

## 4. What this design can detect — stated in advance

Measured on the actual return panel, against 1/N, for competing strategies of
increasing concentration:

| competing strategy archetype | corr. with 1/N | SE of ΔSharpe | min. detectable effect (80% power) |
|---|---|---|---|
| diversified (5 of 9, monthly) | 0.973 | 0.074 | **0.21** |
| concentrated (2 of 9, monthly) | 0.914 | 0.133 | **0.37** |
| corner solution (1 of 9, monthly) | 0.820 | 0.191 | **0.54** |
| single-sector buy & hold (XLK) | 0.828 | 0.191 | **0.53** |

`argmax r̂ᵀw` over a simplex is a corner solution, so **SPO+ and PtO sit in the
bottom rows: this design cannot reliably detect a Sharpe gap below ≈ 0.5.**
The `ℓ₂` variant diversifies and sits nearer the top row (≈ 0.2–0.4).

For reference, the effects arXiv:2601.04062 reports: SPO+ vs PtO **+0.126**,
SPO+ with fee vs PtO **+0.056**, SPO+ vs MaxSharpe **+0.211**. **Their headline
effect is roughly a quarter of what a design of this size can see.**

### Consequences accepted in advance

1. A gap below ≈ 0.5 will be reported as **"not detectable at this sample size"**,
   whichever direction the point estimate falls — including if SPO+ looks like it
   wins. A favourable point estimate is not a finding.
2. A **null is the expected outcome and is a complete result.** The precedent is
   iteration 6: nothing beat 1/N there either.
3. Because a null is likely and expected, **it is not evidence that our SPO+ is
   broken.** The implementation is validated separately, by mechanical tests
   (§5), not by whether it wins.

---

## 5. Implementation validation — independent of the outcome

SPO+ is judged correct by these, decided now, not by its backtest performance:

1. **Oracle sanity.** Fed the realized returns as "predictions", the layer must
   return the oracle portfolio and SPO+ loss must be 0.
2. **Upper-bound property.** SPO+ loss ≥ true decision regret, on random draws.
   This is the theorem the surrogate exists for (Elmachtoub & Grigas 2022).
3. **Gradient check.** The subgradient `2(w* − ŵ)` must match a finite-difference
   estimate of the SPO+ loss in `r̂`.
4. **Causality.** The existing `test_no_lookahead` must pass with SPO+ wired in.
5. **Degenerate-input guard.** Constant predictions must give a well-defined
   (arbitrary-but-feasible) vertex, not a crash.

---

## 6. Rules

1. **No tuning until a preferred method wins.** Hyperparameters are fixed before
   the run and reported whatever happens. If we later change one, the change and
   the reason are logged in `notes.md` and both sets of numbers are shown.
2. **No window shopping.** 2016–2024 is the evaluation window. Regime-specific
   cuts (2020, 2024) may be *reported* but carry T ≈ 250, SE ≈ 0.3–0.9 per §4,
   so they will be labelled non-inferential.
3. **Every number traces to a committed file** under `results/`.
4. **The universe is not changed after seeing results.**
5. If any of these is broken, it is recorded in `notes.md` as a deviation from
   this pre-registration rather than quietly absorbed.

---

## 7. What would change our mind about the paper

We are not trying to show the paper is wrong. Concretely, we would conclude the
SPO+ advantage is real if: SPO+ beats 1/N by more than the §4 threshold for its
concentration class, net of 50 bps, with the bootstrap CI excluding zero, and the
gap does not vanish when the ℓ₂ variant diversifies away the corner solution.

That outcome is possible and would be reported as readily as a null.
