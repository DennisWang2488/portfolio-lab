# portfolio-lab — iteration log (append-only)

Same convention as `theory_notes_c3.md`: never rewrite earlier blocks; append
`## Iteration N` with *What I did / What I learned / What is still open / Verdict*.

## Iteration 1 — 2026-07-07 — scaffold, harness, baselines on real data

**What I did.**
- Scoped the project (conversation with Dennis): product-oriented PO lab; Phase 1
  = honest replication/stress-test of the DFL-beats-PTO-on-real-markets claim
  (arXiv:2601.04062, no code released); Phase 2 candidates = LLM-views→Black-
  Litterman and OPS/regret; RL/GRPO deferred to Phase 3.
- Vendored Costa & Iyengar's E2E-DRO (Apache 2.0) incl. its real dataset
  (20 US large-caps weekly 2000–2021 + FF factors) and their cached experiment
  results (`vendor/E2E-DRO/cache/exp*`, ~69MB, git-ignored).
- Built `polab/`: 4 baselines (1/N, min-var, max-Sharpe via convex tangency
  reformulation, risk parity via Spinu 2013), walk-forward engine (drifting
  weights, proportional costs, strict causality), metrics (PSR, expected-max-SR,
  Deflated Sharpe; Bailey & López de Prado).
- 9 offline tests, all passing — incl. a mechanical no-look-ahead test and a
  risk-contribution-equalization test for risk parity.
- First real-data table (10 bps, 2y lookback, ~monthly rebal, 1031 OOS weeks):

  | strategy | ann_ret | ann_vol | Sharpe | maxDD | turnover/rebal |
  |---|---|---|---|---|---|
  | equal_weight | 13.6% | 17.2% | 0.829 | −47.3% | 0.023 |
  | min_variance | 10.1% | 13.2% | 0.797 | −31.4% | 0.073 |
  | max_sharpe   | 15.7% | 17.2% | 0.934 | −35.7% | 0.165 |
  | risk_parity  | 12.5% | 15.2% | 0.856 | −41.6% | 0.026 |

**What I learned.**
- The vendored dataset removes the AlphaVantage-key dependency entirely —
  replication of their experiments can be fully offline.
- The table is the textbook picture: 1/N is a strong null; max-Sharpe edges it
  on Sharpe but with 7× the turnover (cost-sensitivity to probe later);
  min-variance delivers the drawdown/vol reduction it promises. Sanity high.
- Env: anaconda base has numpy/pandas/scipy/cvxpy/torch; `cvxpylayers` and
  `yfinance` are MISSING. Deliberately did not pip-install into base (atlas
  experiments depend on that env) — the DFL step needs a dedicated env.

**What is still open.**
- Reproduce E2E-DRO's own experiments (their `main.py` + cached exp results
  give a no-retrain comparison path first).
- Then implement/port the SPO(+) loss on this data and run the honest harness
  over it — the arXiv:2601.04062 stress test.
- Cost-sensitivity sweep (0/10/25/50 bps) once DFL numbers exist.
- Statistical test for Sharpe differences (Ledoit-Wolf / Jobson-Korkie) not yet
  in `metrics.py`.

**Verdict.** Skeleton solid, harness tested, real-data baselines sane. Ready for
the replication step.

<promise>PORTFOLIO_LAB_ITER_1_COMPLETE</promise>

## Iteration 2 — 2026-07-07 — cache cross-validation + transaction-cost stress of E2E-DRO

**What I did.**
- New module `polab/e2edro_io.py`: loads Costa & Iyengar's cached pre-trained
  experiment objects WITHOUT their env (sys.modules stubs for cvxpylayers /
  alpha_vantage / pandas_datareader / statsmodels; `pd.read_pickle` for pandas
  pickle-compat; psutil.cpu_count()->None sandbox quirk patched).
- New `backtest.cost_overlay()`: uniform post-hoc proportional-cost treatment
  for any (gross returns, weight path) pair — lets us compare their engine's
  output and ours under identical cost assumptions. Tested against the
  engine's own cost accounting (agree to 1e-12).
- `scripts/compare_e2edro_cache.py`: their 8 main-experiment nets + our 4
  baselines, one table, OOS 2013-01..2021-10 (454 weeks), gross + 10bps + 25bps.
- 10/10 tests pass.

**What I learned.**
- **Engine cross-validation passed at machine precision**: our 1/N reproduces
  their cached ew_net exactly (max |diff| = 1.4e-17). Two independent backtest
  implementations, same numbers. High confidence in the harness.
- **Their headline ranking reproduces from cache (gross)**: dr_net_learn_theta
  1.41 > dr_net 1.31 > nom_net 1.18 > 1/N 1.05 > po_net 0.90 (Sharpe).
- **The cost overlay is brutal and differential** (weekly one-way turnover in
  parens): dr_net_learn_theta *survives* — 1.41 → 1.31 @10bps → 1.16 @25bps
  (0.18); dr_net similar (0.12). But nom_net (0.34) drops 1.18 → 1.00 → 0.73,
  **below 1/N at 25bps**; dr_net_learn_gamma (0.45) collapses 1.02 → 0.15;
  po_net (0.58!) goes 0.90 → 0.51 → **negative** at 25bps. 1/N barely moves
  (1.051 → 1.034).
- Take: *gross-return DFL comparisons are close to meaningless* — turnover
  differences of 5-50x swamp the alpha differences. Any honest replication of
  the arXiv:2601.04062 claim must be net-of-costs with per-strategy turnover
  reported. (They do model costs in-objective; our job is to verify the
  *evaluation* is cost-honest too.) Also: our plain max_sharpe baseline beats
  their nominal E2E net of 10bps.
- Their po_net is a *feature-based* predict-then-optimize (prediction layer +
  fixed decision layer), not our statistical baselines — the right "PTO
  strawman vs PTO done well" distinction to keep in mind when we judge DFL
  papers' baselines.

**What is still open.**
- Caveats to carry: single OOS window; survivorship-biased 20-stock universe;
  their nets retrain only 4x through the window; DSR computed on gross.
- Sharpe-difference test (Ledoit-Wolf / Jobson-Korkie) still missing.
- Next: dedicated conda env (cvxpylayers) → retrain their nets from scratch to
  verify the cache is what training actually produces → then implement the
  SPO loss and confront arXiv:2601.04062 directly.

**Verdict.** Harness independently validated; the paper's gross ranking
reproduces, and the cost stress already yields the project's first real
finding: DFL's advantage here is *variant-dependent and turnover-fragile*,
which sharpens exactly what to test in the SPO replication.

<promise>PORTFOLIO_LAB_ITER_2_COMPLETE</promise>

## Iteration 3 — 2026-07-07 — Colab retraining prep (awaiting user's Colab run)

**What I did.**
- Dennis: no NVIDIA GPU locally (macOS, M5 Pro) → retraining goes to Colab.
  Noted honestly: this workload is CPU-bound anyway (cvxpylayers/diffcp solves
  small conic programs sequentially; the nets are 8→20 linear layers). Colab's
  value = clean disposable env, not GPU.
- Extracted the CV-winning hyperparameters from the cached objects so Colab
  skips the 3×6 grid (~9× compute saved): nom lr=0.02/50ep, dr lr=0.0125/50ep,
  theta lr=0.0125/40ep, base lr=0.005/30ep. po_net has no trained params (OLS).
- Built `notebooks/colab_retrain_e2edro.ipynb` (self-contained: pip installs,
  clones upstream repo — dataset ships in it — smoke tests ew/po first, then
  nom_net and dr_net with reference Sharpe values embedded; zips new_cache for
  download). Runtime estimate: nom 1–3 h, dr 2–6 h on Colab CPU.
- `e2edro_io.load_net()` now takes a cache_dir; `compare_e2edro_cache.py`
  auto-diffs `new_cache/exp` (retrained) vs `cache/exp` (shipped) when present.

**What is still open.**
- User runs the notebook on Colab; retrained pickles go to
  `vendor/E2E-DRO/new_cache/exp/`; rerun the compare script.
- Success criterion: ranking `dr > nom > 1/N > po` preserved, Sharpes near
  reference (exact reproduction not expected — library versions moved since
  2022, even with their seed 1000). Deviation is a finding, not a failure.

**Verdict.** Prep complete; blocked on the Colab run (user action).

<promise>PORTFOLIO_LAB_ITER_3_PREP_COMPLETE</promise>

## Iteration 4 — 2026-07-28 — Colab run diagnosed; divide-and-conquer retraining

**What I did.**
- Diagnosed the failed Colab run from `retrained_nets.zip` alone. Contents:
  `ew_net.pkl` + `po_net.pkl` (both smoke tests PASSED) and
  `nom_initial_state_linear_TrainDeltaFalse` but no `nom_net.pkl` / `dr_net.pkl`.
  Timestamps 01:41–01:42:38 ⇒ **fast deterministic failure**, not a timeout.
  po_net's success rules out: CvxpyLayer *forward*, ecos, diffcp import, the
  pandas OLS/insert logic, SlidingWindow, split_update. Remaining suspects
  (training-only, untested by the smoke tests): (a) `torch.load(init_state_path)`
  under PyTorch>=2.6's `weights_only=True` default, (b) backprop through the
  CvxpyLayer. Added notebook cells 6.5 (isolates both in <1 min, prints the
  traceback, times one fwd+bwd) and 6.6 (torch.load patch).
- **Key structural finding:** `net_roll_test` reloads the SAME init state and
  re-fits the prediction layer to OLS at the top of every roll window ⇒ the 4
  windows share no state and are **embarrassingly parallel**.
- `polab/rolls.py` reproduces their split arithmetic in pure pandas. Verified
  offline: windows tile as 114+113+114+113 = **454**, matching both their
  `pc.backtest` sizing and the 454 OOS weeks observed in iter 2. Test added.
- Real scale exposed: **146,700 fwd+bwd per net** (full-batch accumulation, one
  Adam step per epoch; roll 0 = 28k steps, roll 3 = 45k). At ~0.4 s/step that is
  ~16 h serial — **exceeds Colab free-tier session limits**, which is why the
  run could never have finished even without the crash. Parallel wall-clock is
  the slowest single roll (~45k steps), not the sum.
- `scripts/train_roll.py`: runs ONE (net, roll) job; faithful line-by-line
  reimplementation of `net_train` plus per-epoch checkpointing (crash loses ≤1
  epoch) and resume. `scripts/combine_rolls.py`: stitches chunks, verifies the
  assembled length, scores vs the shipped cache.
- 11/11 tests pass.

**What is still open.**
- Need the actual traceback from cell 6.5 to confirm which suspect it was.
- Decision: local venv (M5 Pro, no disconnect risk, 8 jobs in parallel) vs Colab.
- `train_roll.py` is untested end-to-end (no cvxpylayers locally by design) —
  first real run doubles as its test.

**Verdict.** Failure diagnosed to two candidates without needing the traceback;
the harder problem (the run was never going to fit in a Colab session) is solved
structurally by exploiting roll-window independence.

<promise>PORTFOLIO_LAB_ITER_4_COMPLETE</promise>

## Iteration 5 — 2026-07-28 — parallel Colab (TPU-runtime) notebook

**What I did.**
- Dennis chose Colab over a local venv, using the **TPU runtime for its ~90 vCPUs**
  (the TPU itself is unused — the workload is CPU-bound conic solves; no torch_xla).
  Good call: 12 single-threaded jobs fit easily, and wall-clock collapses to the
  slowest single roll (~45,200 steps) instead of ~16 h serial.
- `scripts/train_roll.py` now honours `POLAB_VENDOR` so a foreign layout
  (Colab: upstream at /content/E2E-DRO) works without mirroring our tree.
- `scripts/make_colab_parallel_notebook.py` generates
  `notebooks/colab_parallel_retrain.ipynb`. It **inlines `polab/rolls.py` and
  `scripts/train_roll.py` by reading them at generation time** — the repo stays
  the single source of truth, no forked copy in the notebook. Verified the
  inlined blobs re-parse with ast.
- Division of labour: Colab produces chunk pickles; **combining/scoring happens
  locally** (`combine_rolls.py` needs the vendored cache + full polab package).
- Resilience: Drive-mounted chunk dir + per-epoch checkpoints ⇒ a dropped session
  costs at most one epoch; re-running the launch cell skips finished jobs and
  resumes partial ones (skip logic lives in the worker, not the notebook).
- Default job set: nom × 4, dr × 4, dr_theta × 4 = 12 (dr_theta included because
  it was the iter-2 cost-stress winner: Sharpe 1.41, survives 25 bps).

**What is still open.**
- The run itself. Cell 6 prints a measured per-step time and a grounded ETA
  before any hours are committed.
- Whether the earlier crash was `torch.load`'s weights_only default or the
  CvxpyLayer backward — the worker pre-empts the former; cell 6 exposes the
  latter immediately if it is still live.

**Verdict.** Parallel path ready; the run is now bounded by one roll window, not
by the serial total, and survives disconnects.

<promise>PORTFOLIO_LAB_ITER_5_COMPLETE</promise>

## Iteration 6 — 2026-08-02 — Sharpe-difference tests: nothing beats 1/N

**What I did.**
- Dropped the from-scratch retrain (Dennis's call). The machinery from iters 4–5
  is committed as a record; the retrain would only have verified that the
  shipped cache equals what training produces, which is a premise of neither
  finding below. Recorded as a known limitation, not a pending task.
- New `polab/sharpe_test.py` — the comparative test that has been open since
  iteration 2. Three paired procedures on the same dates: Jobson–Korkie with
  the Memmel (2003) correction (i.i.d.-normal, reference point only),
  Ledoit–Wolf (2008) HAC delta method (Parzen kernel, Andrews automatic
  bandwidth, VAR(1) prewhitening + recoloring), and their studentized circular
  block bootstrap. Calibration check before use: over 400 replications with
  t(4) returns and a shared market factor, empirical size at nominal 5% was
  4.3% (LW) — the test is not over-rejecting on fat tails.
- `scripts/sharpe_tests.py` — all 12 strategies against two benchmarks at
  0/10/25 bps, 4999 resamples: `polab_equal_weight` (the DeMiguel et al. null)
  and `po_net` (their own predict-then-optimize net, i.e. the paper's claim).
- Degenerate-pair guard: our 1/N equals their `ew_net` to 1e-17, so that pair
  has an identically-zero difference and zero standard error. Returned as a
  labelled degenerate result rather than crashing the bootstrap. 17/17 tests.

**What I learned.**
- **Not one strategy is distinguishable from 1/N, at any cost level.** The best
  case is `dr_net_learn_theta` gross: +0.363 Sharpe over 1/N, SE 0.271,
  p = 0.202, 95% CI [−0.201, +0.961]. At 10 bps it is +0.267, p = 0.352. Every
  standard error on an annualized Sharpe difference here is 0.19–0.32, so the
  paper's headline gaps are on the order of 1–2 SE. Verdict: **the ranking in
  Table 3 is not statistically separable on its own OOS path.**
- **The only significant results against 1/N are negative**: `po_net` at 10 bps
  (−0.537, p = 0.020) and at 25 bps (−1.113, p < 0.001), plus
  `dr_net_learn_delta` (−0.652, p = 0.021) and `dr_net_learn_gamma`
  (−0.885, p = 0.005) at 25 bps. What the data can establish is which methods
  *lose* to doing nothing, not which win.
- **"E2E beats two-stage" is significant, but not for the reason claimed.** At
  10 bps `dr_net_learn_theta` beats `po_net` by +0.804 (p = 0.003) — but 1/N
  beats `po_net` by +0.537 (p = 0.020) on the same path. The significance comes
  from `po_net` being bad (0.58 weekly one-way turnover) rather than from
  end-to-end training being good. Gross, the same comparison is +0.515 with
  p = 0.056 — not significant even before costs. So the paper's central
  comparison clears the bar only against its own weakest baseline, and only
  once costs it does not model are applied.
- The HAC asymptotic and the bootstrap agree closely (dr_theta vs 1/N @10bps:
  p = 0.328 vs 0.352), so the conclusion is not an artifact of either. Block
  size does not drive it either: p ∈ [0.343, 0.447] for b ∈ {1,3,5,10,20}.

**What is still open.**
- p-values here are marginal. The selection-bias side stays with the DSR column
  of `e2edro_cache_comparison.csv`; the two are complementary and neither
  substitutes for the other. A joint treatment (LW's multiple-testing extension)
  is not implemented.
- Everything rests on ONE test path of 454 weeks, 2013–2021, no 2008, single
  seed, survivorship-biased 20-name universe. A wider Sharpe-difference test
  cannot fix a single-path design; only more paths can.
- Retrain not run (see above). We are testing their shipped artifacts.
- Next: the SPO loss and arXiv:2601.04062 — the north-star question, now with
  the comparative test in place to judge it with.

**Verdict.** The methodological finding is complete and it is a negative one:
on this dataset, over this window, no method in Costa & Iyengar's Table 3 —
theirs or ours — is statistically distinguishable from equal weight, and their
E2E-over-PTO gap is a turnover artifact of their own baseline. Combined with
iteration 2's cost stress, that is the whole story the E2E-DRO replication had
to tell, and it is enough to move on.

<promise>PORTFOLIO_LAB_ITER_6_COMPLETE</promise>

## Iteration 7 — 2026-08-10 — the pre-registered SPO+ re-test: executed

**What I did.**
- Ran the design fixed in `audits/prereg-spo-retest.md`. 9 SPDR sector ETFs,
  2016-01-29 … 2024-12-31 (2246 days, 108 monthly rebalances), 12-month trailing
  training, 50 bps (their own cost figure), paired LW studentized block
  bootstrap, 4999 resamples. `scripts/run_spo_retest.py`; every number in
  `results/spo_retest_{performance,tests}.csv`.
- **Two specification gaps closed in advance, not after seeing results.**
  (a) Technical-indicator windows: the paper names the feature families and no
  parameters (audit §38) and the pre-registration inherited that gap. Fixed in
  `polab/features.py` at textbook defaults (RSI 14, MACD 12/26/9, Bollinger
  20/2, SMA 20, volume 20) — declared, never tuned per window.
  (b) `lam` for the l2 variant: 0.1, chosen because maximizing `c'w - lam||w||²`
  over the simplex is `proj_simplex(c / 2lam)`, so lam sets how much predicted
  spread is needed to leave equal weight; 0.1 puts it in the diversified regime,
  which is the point of the variant.
- **Corrected my own first pass.** I applied a blanket MDE of 0.5 to every
  strategy. That is not the pre-registration: §4 gives the minimum detectable
  effect *by archetype* (corner 0.54 / concentrated 0.37 / diversified 0.21),
  because power is set by correlation with 1/N. Now assigned from each
  strategy's measured effective positions `1/Σw²`. This matters: it moved
  SPO+(l2) vs 1/N net-of-cost from "not detectable" to a real, significant
  finding. Recorded here because the fix cut *against* the convenient answer.
- 4 new tests (28/28): feature causality (corrupt the future, past features
  unchanged), the overlapping-target gate, MDE archetype mapping, feature shape.

**What I learned.**
- **`argmax r̂ᵀw` over the simplex is measured to be exactly a single-asset bet.**
  PtO and SPO+ both show effective positions **1.00** and **100% one-way turnover
  every month** — the entire portfolio is replaced at each rebalance. This is not
  a modelling choice of ours; it is what the paper's own layer means, and it is
  why the audit put these strategies in the low-power row.
- **The paper's headline effect does not appear.** It reports SPO+ vs PtO
  **+0.126**. We measure **−0.057** gross (SE 0.105, p = 0.592) and **−0.043**
  net (p = 0.673): the wrong sign, and an order of magnitude inside the noise.
- **Costs annihilate both learned strategies.** Gross, PtO 0.915 and SPO+ 0.858
  look like the best strategies in the table. At the paper's own 50 bps they
  fall to **0.490** and **0.447**, while 1/N goes 0.831 → **0.822**. The entire
  apparent advantage is turnover the paper does not report.
- **Nothing beats 1/N; the only significant result is a loss.** SPO+(l2) loses to
  1/N by **−0.489 net, p = 0.012** — the one variant that diversifies is the one
  we can prove is worse. Best in the table is SPY buy-and-hold (0.900).
- Same shape as iteration 6 on a completely different dataset, universe,
  frequency and method family: what these designs can establish is which methods
  lose to doing nothing.

**What is still open.**
- Our universe is not theirs (unrecoverable from the paper); this is a re-test of
  the method, never described as reproducing their numbers.
- One path, one seed, 9 survivorship-clean ETFs. A null on one path is not proof
  of no effect anywhere — it is proof the reported effect is not visible here at
  a size where it should be, given they report a larger effect on a smaller design.
- Linear predictor only, as specified. A richer predictor is a different paper.

**Verdict.** The pre-registered re-test is complete and the answer is negative
on all three hypotheses. H1: no. H2: no — point estimate has the wrong sign.
H3: moot, since there is no gross advantage to survive costs. Phase 1b closed.

<promise>PORTFOLIO_LAB_ITER_7_COMPLETE</promise>

## Iteration 8 — 2026-08-10 — IC × breadth × cost × stability: the design map

**What I did.**
- Turned the project's question around. Every empirical result so far has been a
  null because on one real path SE(ΔSharpe) ≈ 0.2–0.3. Simulation inverts that:
  **we set the effect size**, so power exists by construction, and the question
  becomes the one institutions actually face — *given a signal of quality IC over
  breadth N at cost c, how far should the decision move?*
- `polab/simulate.py` — one-factor monthly returns calibrated to the sector-ETF
  panel, and `signal_with_ic` producing a prescribed cross-sectional IC.
- `polab/stability.py` — decision rules indexed by stability: `blend` (partial
  rebalancing, closed form) and `no_trade_band` (ℓ₁, a genuine no-trade region).
  λ is set from the signal's own spread so target concentration stays fixed as IC
  varies — otherwise IC and concentration confound, the exact defect that made
  the SPO+ comparison uninterpretable.
- `scripts/run_ic_breadth_stability.py` — 5 IC × 4 breadth × 9 α × 4 cost,
  40 seeds × 240 months, common random numbers so all comparisons are paired.
  Hypotheses S1–S3 and validations V1–V2 declared in the module docstring before
  running.
- 8 new tests (36/36): simulator moments, achieved IC, α=0 ⇒ zero turnover,
  monotone turnover in α, and target concentration invariance across IC.

**Two design errors I made and fixed — both changed the conclusion.**
1. *Cost was fixed at 50 bps.* Wrong: the blend rule never consults cost, so
   net at any level is `gross − rate·traded` on the *same* path. Cost is free to
   sweep, and it turned out to be the axis that decides everything.
2. *The benchmark was rebalanced 1/N, which pays turnover.* That made a
   **zero-signal** rule look significantly profitable at 50 bps (t up to +6.6)
   purely by not churning — a benchmark artifact, not an effect. The null must be
   the same investor *without a signal*: buy-and-hold 1/N (α=0, zero cost). With
   that fixed, IC=0 gives α\*=0 and excess exactly 0, and trading on noise is
   significantly negative (largest |t| = −34).

**What I learned.**
- **V1 holds and is informative.** Gross excess IR is exactly linear in IC (the
  ratio to Grinold's IC·√N is constant to 2 decimals across every IC), confirming
  the signal construction. But the ratio *falls* with breadth (0.53 → 0.39 → 0.29
  → 0.21 for N = 10 → 100), i.e. IR grows far slower than √N. Long-only plus a
  fixed concentration budget is what caps it — you cannot spend breadth you are
  not allowed to take positions in.
- **S1 holds, but only in a band.** α\* is interior exactly where the signal is
  worth acting on but not free to act on: at 25 bps, IC = 0.10 gives α\* = 0.35–0.50.
  Outside that band the optimum is on a boundary — α\* = 1 when cost is 0, α\* = 0
  when the signal cannot pay for itself.
- **S2 holds strongly.** α\* is monotone increasing in IC at every cost level, and
  monotone *decreasing* in cost at every IC. The weaker the signal or the dearer
  the trade, the more stability should be imposed. This is the project's two
  nulls stated as a design rule instead of a complaint.
- **The break-even IC table is the practically useful output**, and it explains
  every null this project has produced:

  | cost | N=10 | N=25 | N=50 | N=100 |
  |---|---|---|---|---|
  | 0 bps  | 0.02 | 0.02 | 0.02 | 0.02 |
  | 10 bps | 0.10 | 0.05 | 0.05 | 0.05 |
  | 25 bps | 0.20 | 0.10 | 0.10 | 0.10 |
  | 50 bps | >0.20 | 0.20 | 0.20 | 0.20 |

  Real monthly cross-sectional ICs are ~0.02–0.05; 0.10 is very good, 0.20 is
  exceptional. **At 50 bps on 9 assets you need an IC no equity signal reliably
  has.** The SPO+ re-test ran at exactly that corner — 9 assets, 50 bps — so its
  null was structural, not a property of SPO+. Breadth buys back roughly one
  grid step of break-even IC, and only up to ~25 names, after which it flattens.

**What is still open.**
- Equal betas, one factor, Gaussian, proportional costs only. No market impact,
  no regime shifts, no signal autocorrelation — a persistent signal would need
  less trading for the same IC and should shift α\* up. That is the first
  robustness axis to add.
- `no_trade_band` (ℓ₁) is implemented and tested but not yet swept against
  `blend` at matched turnover. That comparison is the one that would say whether
  a *no-trade region* beats *partial rebalancing* — i.e. whether the mechanism,
  not just the amount, matters.
- Nothing here is yet a method. It is the map that tells us where a method could
  possibly help: the interior band, where α\* is neither 0 nor 1.

**Verdict.** The question is now well-posed and the design has power. S1–S3 all
hold, both validations pass, and the break-even table retro-explains the whole
project's null record. Next: the ℓ₁-vs-blend mechanism comparison, then a
cost- and IC-aware rule that sets α itself instead of being told it.

<promise>PORTFOLIO_LAB_ITER_8_COMPLETE</promise>
