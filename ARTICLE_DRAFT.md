# Do open-weight agents loop more? What 58,950 public trajectories actually show

Put the exact same model weights inside two different agent harnesses on the exact same coding tasks, and its loop rate often swings more than the gap between open-weight and closed frontier models.[^1] Across Terminal-Bench 2.0 and SWE-rebench, running identical weights across scaffolds shifts loop density by a median of **6.48 loops per 100 steps**—and up to **26.08 loops per 100 steps** on `gpt-5.2`—while the capability- and length-matched gap between open-weight and closed models on SWE-rebench is just **+0.77 loops per 100 steps** (`95% CI [+0.50, +1.02]`).[^1][^2]

![SWE-rebench Open vs Closed Loop Gap Across Controls](assets/hero_raw_vs_controlled.png)

We audited **56,797 executed trajectories** across Terminal-Bench 2.0, SWE-rebench, Longtail-SWE, and Sierra's τ²-bench using a deterministic step labeler, cache-aware billing reconstruction, and counterfactual loop-detector replay.[^3] Here is what holds up when we try to break the result.

## 1. The Raw Gap Looks Huge—Until You Inspect Who Ran What

Naive averages on standard scaffolds make open-weight models look far more prone to looping.[^3] On Terminal-Bench `terminus-2` (`16` open vs `16` closed models across `89` tasks), open models average **6.91** loop steps per 100 steps vs **2.69** for closed models (unmatched gap **+4.22**).[^3] On SWE-rebench `swelike` (`7` open vs `6` closed models across `111` tasks), open models average **5.71** vs **0.98** loops/100 steps (gap **+4.73**), with **89.3%** of open runs flagging at least one loop vs **30.0%** of closed runs.[^3]

Three confounders inflate those raw gaps:[^3]
1. **Capability mismatch**: Leaderboards pair legacy `3%–19%` pass-rate open models (`gpt-oss-20b`, `gpt-oss-120b`) against `60%–72%` closed models; failing runs repeat broken commands until the turn cap.
2. **Longer open-weight runs**: On SWE-rebench `swelike`, open models average **94.7** steps/run vs **45.4** for closed models, giving them more chances to cross any binary per-run loop threshold.[^2]
3. **Hidden harness and tool differences**: Proprietary entries (`claude-code`, `codex`, `gemini-cli`, `cursor`, `junie`) bundle custom loop guards, while Sierra's `banking_knowledge` folder mixes four retrieval toolsets (`AllTools`, `terminal_use`, `openai_embeddings`, `qwen_embeddings`).[^3]

## 2. Controlling for Skill, Run Length, Labeler Strictness, and Model Sampling

Matching open and closed models inside identical pass-rate bands (`<25%`, `25–40%`, `40–55%`, `55–75%`, `>=75%`) on the same scaffold and tasks cuts the SWE-rebench gap nearly in half, from **+4.73** to **+2.54** loops/100 steps (`1-stage task CI [+2.30, +2.78]`).[^3] Matching simultaneously on pass-rate band and step-count decile trims it to **+0.77** (`[+0.50, +1.02]`).[^2]

Trajectory length explains almost the entire binary gap in *"how many runs have at least one loop."* In a run-level logistic regression with task fixed effects and two-way `(model, task)` cluster-robust SEs, open-weight status has an odds ratio of **8.43** (`p=0.0005`) on SWE-rebench and **2.25** (`p=0.024`) pooled across all seven cohorts.[^2][^3] Adding `log(steps)` collapses the SWE-rebench odds ratio to **1.34** (`p=0.58`) and the pooled odds ratio to **1.47** (`p=0.25`)—both indistinguishable from `1.0`.[^2][^3] Open models flag more runs mainly because their trajectories run longer.

![Pre-Specified Cohort Verdicts](assets/cohort_verdicts.png)

To avoid cherry-picking across seven uniform-scaffold cohorts, we applied one pre-specified verdict rule: a cohort is **"higher for open"** only if its matched-gap 95% CI excludes zero *and* its adjusted Negative Binomial `IRR` is reliable with `p < 0.05`.[^4] Under 1-stage task bootstrapping, only **swe_rebench_swelike** (matched gap `+2.54`, NB `IRR=2.69, p=0.0001`) and **tb_openhands** clear both bars.[^4] Terminal-Bench `terminus-2` (`+4.20` matched gap, NB `IRR=1.41, p=0.33`), `tb_mini_swe_agent`, and Sierra `tau2_banking_knowledge_alltools` (`-1.13` unmatched, `+0.82` matched, NB `IRR=0.92, p=0.84`) land in **"mixed"** or **"null"** (`Longtail-SWE` has zero overlapping skill bands because closed models used `1M` context vs `262K` for open models).[^3][^4] Upgrading to a **two-stage `(model, task)` bootstrap** (`[+2.04, +3.05]` on SWE-rebench; `[+1.44, +6.88]` on TB) widens CIs `1.5x–3x`, reflecting `2–16` models per side.[^4]

We also audited labeler precision across 40 random flagged runs (`audit/flagged_samples.md`) and tested a **strict labeler** keeping only exact commands repeated `>=3` times consecutively, A-B-A thrashing (`>=3` cycles), and period-`k` rotations—dropping single repeats, near-duplicates, redundant re-reads, and error streaks.[^5][^6] Under strict rules, SWE-rebench loop density falls `~70%` in both groups—to **0.13** vs **0.02** loops/100 (`Open` vs `Closed`), shrinking the matched gap to **+0.04** (`1.12` vs `0.67` unmatched on TB).[^6] Furthermore, **47.6%** of open-weight `exact_loop` steps on SWE-rebench (`48.7%` for closed) are harmless read-only checks (`git diff`, `git status`, `grep`, `cat`, `ls`, or unchanged `pytest`).[^7]

![Frontier Showdown on SWE-rebench](assets/glm52_vs_opus5.png)

In dollar terms (`6,381` cost-validated SWE-rebench runs), pooled loop spend is **2.38% ($117.38 / $4,927.87)** because expensive low-looping closed runs dominate total spend, though within individual models loop dollar share exceeds step share.[^8] Among SWE-rebench frontier peers, open-weight `glm-5.2` hits a **62.9%** pass rate at **$0.38/run** (`1.56` loops/100; `2.18%` loop dollars), competing with `opus-5` (**63.6%** pass, **$1.64/run**, `0.67` loops/100), `grok-4.5` (**64.1%** pass, **$2.25/run**, `0.65` loops/100), and `gpt-5.6-sol` (**72.1%** pass, **$0.55/run**, `0.31` loops/100).[^3][^6]

## 3. Production Loop Detectors Beat Random Stopping on Precision and Savings

![Production Loop Guards vs Matched Random Stoppers](assets/detectors_vs_random.png)

Replaying `gemini_cli`, `openclaw_10` (30-step sliding window), `cline_style` (5 identical consecutive actions), and `openhands_stuck` against 200-seed step-matched random stoppers shows:[^9]
- **Terminal-Bench 2.0**: `gemini_cli` halves the false-stop rate (`FSR`, stopped runs that would have passed)—**10.77%** vs **23.18%** (`95% CI [20.51%, 25.90%]`) random—saving **5.71% ($190.06)** of validated spend (`0.24 pp` pass-rate loss). `openclaw_10` saves **5.05% ($168.28)** at **13.49% FSR** (`0.19 pp` loss), and `cline_style` saves **5.00% ($166.68)** at **18.38% FSR** vs **23.05%** random.[^9]
- **SWE-rebench**: `gemini_cli` saves **2.05% ($100.87)** vs **1.62%** (`[1.42%, 1.82%]`) random, with **22.20% FSR** vs **28.60%** (`[25.04%, 31.99%]`) random. Adjacent-step guards (`cline_style`) miss **82.9% (`277/334`)** of `period_k_rotation` cycles on TB and **98.3% (`118/120`)** on SWE-rebench—making sliding-window memory (`openclaw_10`) essential.[^8][^9]

## 4. Same Model, Different Harness: Scaffolds Move Loop Rates as Much as Weights

![Same Model Across Different Harnesses](assets/harness_dumbbell.png)

Models evaluated across multiple scaffolds on shared tasks (`results/same_model_across_harness.csv`) show that swapping the harness while holding weights and tasks fixed shifts looping dramatically:[^1]
- **`gpt-5.2` (Closed, TB)**: Jumps from **0.71** loops/100 in `codex` to **1.50** in `terminus-2` and **12.56** in `mini-swe-agent` (**11.85**/100 spread).[^1]
- **`claude-opus-4-5` & `claude-sonnet-4-5` (Closed, TB & SWE)**: Shift from **0.72–1.27** in `terminus-2` to **4.15–4.75** in `claude-code`, **3.79–4.59** in `mini-swe-agent`, and **3.66–4.84** in `openhands` (**3.69–4.03**/100 spread).[^1]
- **`gpt-oss-120b` & `qwen3-coder-480b` (Open, TB)**: Drop from **10.75** and **4.14** in `terminus-2` to **3.22** (`mini-swe-agent`) and **2.26** (`openhands`), cutting open-model looping by `45%–70%`.[^1]

Part of this spread reflects step-logging granularity: `terminus-2` batches multi-command shell turns, whereas `claude-code` and `mini-swe-agent` log per-tool steps (and raw `openhands` pairs action and observation events collapsed before scoring).[^1]

## 5. Inside the Google Gemini Family

Binary "open vs closed" labels also hide wide spread inside a single vendor. Across Google Gemini checkpoints:[^3][^10]
- On Terminal-Bench `terminus-2`, **`gemini-2.5-flash`** (`16.9%` pass) hits **12.22 loops/100** (`47.0%` of runs looping) and **`gemini-2.5-pro`** (`29.2%` pass) hits **9.24**, whereas **`gemini-3-pro`** (`44.2%` pass) drops to **4.57** and **`gemini-3-flash`** (`48.3%` pass) reaches **3.25**.[^3] Moving `gemini-3-pro` and `gemini-3-flash` into `gemini-cli` cuts loop rates to **2.15** and **1.52** loops/100.[^1]
- On Sierra τ²-bench, `gemini-3-pro` (`0.93`/100) and `gemini-3-flash` (`1.92`/100) are disciplined on `tau2_classic`, and on `banking_knowledge` (`AllTools`) `gemini-2.5-pro` (`0.81`/100) and `gemini-3.1-pro-preview` (`1.91`/100) rank near the top.[^3] Placing `gemini-3-flash` and `gemini-3-pro` in Sierra's shell-only `terminal_use` scaffold raises loop rates to **2.98** and **2.35** loops/100.[^10]

## 6. Limitations and Audit Diagnostics

1. **Finite model sample size**: Cohorts contain `2–16` open and `4–17` closed models; two-stage `(model, task)` bootstraps capture model-sampling variance.[^4]
2. **Regression collinearity in `tau2_classic`**: Phase 2 reported an adjusted NB `IRR = 0.03` (`p = 0.017`) on `tau2_classic` despite raw rates of **0.42** (Open) vs **0.76** (Closed) loops/100 (**0.56x** raw ratio).[^4][^11] Both open models released in Feb 2026 (`month=13`), while all six closed models released in Sep–Dec 2025 (`months=8..11`), where Dec closed models looped more than Sep Claude models (`IRR=3.11/month`). Dropping `release_months` restores NB `IRR` to **0.56** (`p=0.008`), matching the raw ratio.[^11]
3. **Terminal-Bench cost rebuild exclusions (`n<100` suppression)**: Our `[0.50x, 2.00x]` filter kept `5,011` TB runs (**800** closed runs) and dropped `24,543` mismatches (median RSC ratio `0.12x`).[^12] Over `85%` of dropped runs contain unresolved Next.js RSC `$<hex>` pointers (`$3b`, `$4a`) in `tbench.ai` exports replacing multi-KB outputs with 3-byte strings—not a `cost_cents / 100` or `cache_tokens` bug.[^12] Cells with `<100` cost-valid runs are suppressed as `n<100`.[^3]
4. **Missing Terminal-Bench step traces**: `17,642` of `52,104` TB rows have `steps = null`, including `claude-opus-4-6` (`70.8%` pass), `gpt-5.3-codex` (`68.5%`), and `gemini-3.1-pro-preview` (`65.2%`).[^13]

> ### Method Box: Labels, Matching & Regression Specifications
> - **Default vs Strict Labeler**: Normalization strips ANSI codes, timestamps, UUIDs, and temp paths. The default 6-class labeler flags `exact_loop`, `near_duplicate`, `aba_thrash`, `period_k_rotation (k<=5)`, `error_streak (>=4)`, and `redundant_reread` while excluding `productive_repeat`. The strict labeler keeps only `>=3` consecutive `exact_loop` steps, `>=3`-cycle `aba_thrash`, and `period_k_rotation`.[^6]
> - **Matching & Uncertainty**: Executed runs (`n_steps >= 1`) on shared tasks are stratified into five pass-rate bands and ten step-count deciles (`D0..D9`), weighted by `min(N_open, N_closed)`.[^2][^3] We report 1-stage task and 2-stage `(model, task)` bootstraps (`1,000` replicates), plus Negative Binomial and Logistic GLMs (`has_any_loop` with/without `log1p(steps)`) with task fixed effects and two-way `(model, task)` cluster-robust SEs.[^4]

---

### Footnotes & Data Provenance
[^1]: [`results/same_model_across_harness.csv`](results/same_model_across_harness.csv), [`assets/harness_dumbbell.png`](assets/harness_dumbbell.png).
[^2]: [`results/length_matched_comparison.csv`](results/length_matched_comparison.csv), [`results/length_decile_breakdown.csv`](results/length_decile_breakdown.csv), [`assets/hero_raw_vs_controlled.png`](assets/hero_raw_vs_controlled.png).
[^3]: [`results/loops_by_model_scaffold.csv`](results/loops_by_model_scaffold.csv), [`results/open_vs_closed_matched.csv`](results/open_vs_closed_matched.csv), [`results/open_vs_closed_model.md`](results/open_vs_closed_model.md).
[^4]: [`results/cohort_verdicts.csv`](results/cohort_verdicts.csv), [`results/two_stage_bootstrap_comparison.csv`](results/two_stage_bootstrap_comparison.csv), [`assets/cohort_verdicts.png`](assets/cohort_verdicts.png).
[^5]: [`audit/flagged_samples.md`](audit/flagged_samples.md).
[^6]: [`results/strict_vs_default_labeler.csv`](results/strict_vs_default_labeler.csv), [`results/strict_vs_default_by_model.csv`](results/strict_vs_default_by_model.csv), [`assets/glm52_vs_opus5.png`](assets/glm52_vs_opus5.png).
[^7]: [`results/readonly_exact_loops.csv`](results/readonly_exact_loops.csv).
[^8]: [`results/study_metrics.json`](results/study_metrics.json), [`results/summary.md`](results/summary.md).
[^9]: [`results/detector_simulation.csv`](results/detector_simulation.csv), [`assets/detectors_vs_random.png`](assets/detectors_vs_random.png).
[^10]: [`results/tau2_non_alltools_banking_knowledge.csv`](results/tau2_non_alltools_banking_knowledge.csv), [`results/loops_by_family.csv`](results/loops_by_family.csv).
[^11]: [`results/tau2_classic_irr_diagnosis.json`](results/tau2_classic_irr_diagnosis.json).
[^12]: [`results/tb_cost_rebuild_audit.csv`](results/tb_cost_rebuild_audit.csv), [`audit/tb_cost_rebuild_audit.md`](audit/tb_cost_rebuild_audit.md).
[^13]: [`results/tb_null_step_models.csv`](results/tb_null_step_models.csv).
