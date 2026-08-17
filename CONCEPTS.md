# The spine of portfolio optimization — 7 facts, with our own numbers

This field looks like dozens of competing methods. The load-bearing structure is
the seven claims below; the rest are corollaries. Each one gives: **the claim →
why (the mathematical skeleton) → evidence measured in this repo → how to see it
yourself**. Together they explain every result this project produced, including
why so many of them are nulls. About 20 minutes.

Every number below came out of this repo. Nothing is quoted from a textbook
without our own measurement next to it.

---

## Fact 1 — μ is unestimable, Σ is. Everything follows from this asymmetry.

**The claim.** Expected returns cannot be estimated from history at any sample
size you will ever have. Covariances can.

**Why.** For a return series with volatility σ, the standard error of the sample
mean after T years is σ/√T — and crucially it depends on the *calendar span*,
not the sampling frequency. Sampling daily instead of monthly does not help: you
get more observations of the same total drift. With σ = 20%/yr, to pin the mean
down to ±1% you need

$$T = \left(\frac{\sigma}{\text{SE}}\right)^2 = \left(\frac{0.20}{0.01}\right)^2 = 400 \text{ years.}$$

Variance is the opposite: realized variance converges *in-fill*, so more frequent
sampling genuinely helps. This is Merton's (1980) observation and it is the single
most important fact in the field.

**Consequence.** Any method that needs μ is fighting a lost battle. Any method
that needs only Σ is on solid ground. That is the real dividing line — not
"simple vs sophisticated".

**Our evidence.** In `results/spo_retest_performance.csv`, the Σ-only methods sit
right on top of the null while the μ-dependent ones fall apart after costs:

| | needs μ? | Sharpe net 50bps |
|---|---|---|
| risk_parity | no | 0.829 |
| 1/N | no (needs nothing) | 0.822 |
| min_variance | no | 0.704 |
| max_sharpe | **yes** | 0.670 |
| PtO / SPO+ | **yes** | 0.490 / 0.447 |

**See it:** `python scripts/run_baselines.py`

---

## Fact 2 — The optimizer amplifies estimation error rather than averaging it out

**The claim.** Feeding noisy estimates into mean–variance optimization produces
worse decisions than using cruder inputs. Michaud (1989) called MVO an
"estimation-error maximizer".

**Why.** The unconstrained mean–variance solution is

$$w^\star \propto \Sigma^{-1}\mu.$$

`Σ⁻¹` is the problem. Its action is dominated by the *smallest* eigenvalues of Σ
— exactly the directions estimated least reliably, because they correspond to
near-redundant combinations of assets. The optimizer therefore places its largest
bets along the axes where the data says least. It is not neutral to noise; it
actively seeks it out, because a spuriously high estimated return combined with a
spuriously low estimated variance looks exactly like an opportunity.

**Consequence.** More optimization is not more better. This is why every
practical method is really a way of *limiting* how much the optimizer can act on
its inputs — see Facts 3 and 5.

**Our evidence.** `argmax_w r̂ᵀw` over the simplex is the extreme case: with no
penalty at all the solution is a vertex. We measured effective positions
`1/Σwᵢ²` = **1.00** for both PtO and SPO+ — the optimizer put 100% of the book in
a single ETF, every month, on the strength of a noisy forecast.

**See it:** the `eff_positions` column of `results/spo_retest_performance.csv`

---

## Fact 3 — Constraints are regularization in disguise

**The claim.** A no-short constraint is not (only) a mandate restriction; it is
mathematically equivalent to shrinking the covariance matrix.

**Why.** Jagannathan & Ma (2003): solving min-variance subject to `w ≥ 0` gives
the same answer as solving the *unconstrained* problem with a modified covariance
`Σ̃ = Σ − δ1ᵀ − 1δᵀ`, where δ is the vector of Lagrange multipliers on the
non-negativity constraints. Binding the constraint on asset *i* is exactly
equivalent to reducing its estimated covariances — i.e. shrinking it toward the
rest of the panel.

**Consequence.** The constraint set is a modelling choice with statistical
content, not administrative trim. This also explains why long-only portfolios
often *outperform* their long–short cousins out of sample despite being strictly
less flexible: the flexibility was being spent on noise.

**Our evidence.** Indirect but visible: our long-only simplex caps the achievable
information ratio well below the theoretical law (Fact 4) — a cost in theory that
buys robustness in practice.

---

## Fact 4 — The Fundamental Law: IR ≈ IC · √breadth — but breadth saturates

**The claim.** Grinold's law says the information ratio you can achieve is your
skill per bet times the square root of the number of independent bets.

$$\text{IR} \approx \text{IC}\cdot\sqrt{\text{BR}}$$

IC = the cross-sectional correlation between your forecast and what happens.
Realistic monthly equity ICs are **0.02–0.05**; 0.10 is very good; 0.20 is rare.

**Why it matters.** It says skill and diversification are substitutes. A weak
signal over many names can match a strong signal over few. This is the entire
economic logic of statistical arbitrage.

**But.** We measured it and found the law holds in IC and *fails* in breadth.
From iteration 8's V1 check, the ratio of achieved gross IR to `IC·√N`:

| breadth | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| ratio to Grinold | 0.53 | 0.39 | 0.29 | 0.21 |

The ratio is **constant across every IC** (so IR really is linear in skill — the
law's first half is exact), but it **falls with breadth**: IR grows far more
slowly than √N. The reason is Fact 3 — long-only plus a fixed concentration
budget. *You cannot spend breadth you are not allowed to take positions in.*

**See it:** `python scripts/run_ic_breadth_stability.py` (the V1 block)

---

## Fact 5 — Turnover is the tax that decides everything

**The claim.** In the regime real signals live in, transaction costs are not a
second-order correction. They are the dominant term, and they determine the
ranking.

**Why.** Cost scales with how much the decision *moves*; gross alpha scales with
how *right* it is. When IC is small, "right" is small and "moves" is large,
because a noisy forecast changes a lot between periods. The corner-solution
optimizer of Fact 2 is the worst case: its output is maximally sensitive to its
input.

**Our evidence — this is the project's central empirical finding.** Gross, the
learned strategies are the *best* in the table. Net of the paper's own 50 bps,
they are the *worst*:

| | Sharpe gross | Sharpe net 50bps | turnover/rebal |
|---|---|---|---|
| PtO | **0.915** (1st) | 0.490 (7th) | 1.000 |
| SPO+ | 0.858 | 0.447 (8th) | 1.000 |
| 1/N | 0.831 | **0.822** | 0.013 |

100% one-way turnover means the entire book is replaced every month. And in the
E2E-DRO replication the same pattern sorted *their* variants: the one that
survived 25 bps had turnover 0.18, the one that went negative had 0.58.

**See it:** `python scripts/compare_e2edro_cache.py`

---

## Fact 6 — Why 1/N is hard to beat, and what that does *not* mean

**The claim.** DeMiguel, Garlappi & Uppal (2009) tested 14 optimization methods
and none consistently beat equal weight out of sample. Independent work through
2024 still finds this.

**What it actually means.** It is a statement about **Fact 1**, not about
optimization. Almost all 14 methods needed μ. Methods that need only Σ (risk
parity, min-variance) hold their own — in our table risk_parity 0.829 vs 1/N
0.822 with lower drawdown. The correct lesson is *"do not estimate μ from
historical averages"*, not *"do not optimize"*.

**The second thing it does not mean.** "Beat 1/N" is not the objective an
institution has. They are not choosing between 1/N and optimization; they have a
proprietary alpha signal and need to turn it into positions subject to risk,
cost, capacity, and mandate constraints. Given a real signal, optimization
absolutely adds value. Without one, it cannot.

**Our evidence — the break-even table.** Simulating signals of known quality
(iteration 8), the lowest IC at which the best decision rule beats buy-and-hold:

| cost | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| 0 bps | 0.02 | 0.02 | 0.02 | 0.02 |
| 10 bps | 0.10 | 0.05 | 0.05 | 0.05 |
| 25 bps | 0.20 | 0.10 | 0.10 | 0.10 |
| 50 bps | **>0.20** | 0.20 | 0.20 | 0.20 |

Read it against "real monthly IC is 0.02–0.05". **At 50 bps on a 9-asset universe
you need skill no equity signal reliably has.** Our SPO+ re-test ran at exactly
that corner — so its null was *structural*, not a property of SPO+. This single
table retro-explains every null in this project.

**See it:** `python scripts/run_ic_breadth_stability.py`

---

## Fact 7 — What institutions actually do, and where the money is

**The pipeline.** Not one model — three separable stages:

1. **Alpha.** Proprietary forecasts. *This is the competitive moat.*
2. **Risk model.** Barra / Axioma / Northfield, or in-house. Factor covariance,
   because Fact 1 says Σ is the estimable half.
3. **Optimizer.** MOSEK / Gurobi / OSQP, with turnover limits, position bounds,
   factor-exposure bounds, borrow constraints, lot sizes. Plus a market-impact
   cost model (square-root law), not just proportional costs.

**The key realization.** The optimizer is a **constraint-satisfaction and
cost-control** tool. It is not where alpha comes from. Stage 1 is the moat;
stages 2–3 are engineering that stops you from giving the moat back.

**The tell.** A pattern common enough in practitioner accounts to be worth
naming: desks solve the continuous relaxation and round to integers by hand
rather than running an exact MIP — when the solution says 3.5 shares, a human
decides 3 or 4 — even with a commercial MIP solver already in the stack.

That is not laziness, it is Fact 1 restated at the desk level: **the objective is
flat between 3 and 4 relative to the error in its inputs.** Closing a 0.1%
optimality gap on an objective whose μ carries 50%+ standard error is fitting
noise. The human breaks the tie with information the model does not contain —
borrow availability, upcoming earnings, crossing opportunities.

**Our version of the same phenomenon.** We found no method statistically
distinguishable from 1/N, with standard errors of 0.19–0.32 on every annualized
Sharpe difference. They cannot tell 3 from 4; we cannot tell method A from method
B. Same flat objective, different scale.

---

## The through-line

Fact 1 (μ unestimable) forces Fact 2 (optimizer amplifies noise), which forces
Fact 3 (constrain it) and Fact 5 (do not let the decision move much). Fact 4 says
the only escape is skill or breadth, and Fact 6 quantifies exactly how much of
either you need. Fact 7 says the industry has known all of this for thirty years
and built its process around it.

**Where that leaves a method.** Not "predict better" — Fact 1 says you cannot,
from history. Not "optimize harder" — Fact 2 says that is backwards. The room is
in **how far the decision should move given how good the signal is**, which is
the interior band iteration 8 mapped, and the only region where a new rule could
help.

## How not to fool yourself (what the harness enforces)

- **1/N is the null.** Every table includes it.
- **Costs always on**, turnover always reported. Fact 5 is why.
- **No look-ahead, mechanically tested** — `tests/test_polab.py::test_no_lookahead`
  corrupts the future and asserts past decisions are bit-identical.
- **Power before p-values.** State the minimum detectable effect *before* running;
  a favourable point estimate below it is not a finding.
- **Deflate for multiple testing** — PSR / Deflated Sharpe (Bailey & López de
  Prado) against the whole set of strategies tried, not just the reported one.

## Canonical references

Merton (1980) on estimating the mean · Michaud (1989) estimation-error maximizer ·
Jagannathan & Ma (2003) constraints as shrinkage · Ledoit & Wolf (2004, 2017)
covariance shrinkage · Grinold & Kahn, *Active Portfolio Management* ·
DeMiguel, Garlappi & Uppal (2009) 1/N · Gârleanu & Pedersen (2013) dynamic trading
with costs · Boyd et al., *Multi-Period Trading via Convex Optimization*
(`cvxportfolio`) · Bailey & López de Prado (2012, 2014) PSR / Deflated Sharpe.
