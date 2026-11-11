# Do open-weight agents loop more? What four public trajectory sets say

*Offline audit across Terminal-Bench 2.0, SWE-rebench July 2026, Longtail-SWE, and Sierra's public τ²-bench archive.*

## The short answer

Across **56,797 executed agent trajectories** spanning **7 uniform-scaffold cohorts** (and `104` model-scaffold evaluations across four public benchmarks), comparing raw averages without controls makes open-weight models look worse than they often are — because weak legacy open checkpoints pull raw averages up while frontier closed agent harnesses conceal bespoke loop guards inside proprietary scaffolds.

Once we hold both **the agent scaffold and the exact task set constant** and match models inside identical pass-rate capability bands (`B1 <25%` through `B5 >=75%`), the headline picture fractures rather than confirming a universal open-weight defect: out of cohorts with overlapping pass-rate bands, **4** show a positive Open-vs-Closed loop-rate gap, **2** show no statistically distinguishable difference (95% task-clustered CI crosses zero), and **1** reverse (Open-Weights loop *less* than Closed models at the same capability level).

In a pooled within-scaffold Negative Binomial count regression with `log(steps)` exposure offset and two-way `(model, task)` cluster-robust standard errors — adjusting simultaneously for model pass rate, release month, serving route, task difficulty, and cohort scaffold — open-weight status carries an Incidence Rate Ratio (**IRR**) of **1.31** (`95% CI [0.81, 2.13]`, `p=0.2765`), while a run-level Logistic regression for exhibiting at least one loop gives Odds Ratio (**OR**) **2.25** (`[1.11, 4.53]`, `p=0.0236`; hierarchical random-task `BinomialBayesMixedGLM` OR **2.61**).
 By contrast, **+10 percentage points of model pass rate** shifts loop density with IRR **0.86** (`p=0.0122`), and every **+100 steps in scaffold step cap** moves loop rate with IRR **1.22** (`p=0.0000`).

## Why naive open-vs-closed benchmarks mislead

Three confounders routinely masquerade as an intrinsic weight property:
1. **Scaffold mixing**: Custom proprietary harnesses (`claude-code`, `codex`, `gemini-cli`, `cursor`, `junie`) only ran closed models and already ship bespoke retry/turn guards. Pooling them against open models on bare `terminus-2` or `swelike` measures harness engineering, not weights.
2. **Retrieval tool asymmetry (`tau2 banking_knowledge`)**: Inspecting Sierra's S3 `results.json` metadata revealed four different retrieval harnesses inside `banking_knowledge` alone — `AllTools` (`BM25 + dense + shell`), `terminal_use` (`shell` only, e.g. `gemini-3-flash` and `gemini-3-pro`), `openai_embeddings`, and `qwen_embeddings`. Our fair comparison strictly isolates the **20 uniform `AllTools` models** (`3` Open vs `17` Closed).
3. **Capability imbalance & context truncation**: Comparing a `10%`-pass model that slams into brick walls against an `80%`-pass frontier model inflates failure looping. Meanwhile on Longtail-SWE (`swelike`), closed models ran with `1,000K–1,050K` context while 4 of 5 open models ran at `262K` (and `131K` on `qwen3.6-27b` reruns), and `deepseek-v4-flash` suffered `49` zero-step API timeouts.

## What each of the four datasets shows

| Cohort (Identical Scaffold & Shared Tasks) | Unmatched Loops/100 (Open vs Closed) | Unmatched Gap [95% CI] | Matched Gap [95% CI] | Adjusted NB IRR (p_2way) | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Terminal-Bench 2.0 (terminus-2) | 6.91 vs 2.69 | +4.22 [+2.83, +5.58] | +4.20 [+2.86, +5.49] | 1.56 (0.3266) | higher for Open-Weights |
| Terminal-Bench 2.0 (openhands) | 14.76 vs 7.66 | +7.11 [+4.04, +10.23] | +6.86 [+3.79, +10.08] | 1.77 (0.0272) | higher for Open-Weights |
| Terminal-Bench 2.0 (mini-swe-agent) | 19.99 vs 11.91 | +8.08 [-1.90, +20.06] | +0.50 [-10.21, +12.85] | 1.01 (0.9503) | statistically indistinguishable from zero (null) |
| SWE-rebench Jul 2026 (swelike) | 5.71 vs 0.98 | +4.73 [+4.46, +5.00] | +2.54 [+2.30, +2.78] | 2.69 (0.0014) | higher for Open-Weights |
| Longtail-SWE (swelike) | 3.42 vs 0.96 | +2.45 [+2.13, +2.79] | +0.08 [-0.22, +0.36] | 1.08 (0.7974) | statistically indistinguishable from zero (null) |
| Sierra tau2 Classic (airline+retail+telecom) | 0.42 vs 0.76 | -0.34 [-0.61, -0.04] | -0.32 [-0.59, -0.04] | 0.03 (0.0167) | reversed (lower for Open-Weights) |
| Sierra tau2 Banking Knowledge (AllTools) | 2.70 vs 3.83 | -1.13 [-2.65, +0.62] | +0.82 [+0.38, +1.42] | 0.90 (0.8368) | higher for Open-Weights |

### Cohort-by-cohort highlights

- **Terminal-Bench 2.0 (terminus-2)**: Before matching, Open averaged **6.91** vs Closed **2.69** loops/100 steps (gap **+4.22**; any-loop run share gap **+16.25 pp**). After capability-band matching (4 overlapping band(s)), the gap is **+4.20** `[+2.86, +5.49]` loops/100 steps (**higher for Open-Weights**).
- **Terminal-Bench 2.0 (openhands)**: Before matching, Open averaged **14.76** vs Closed **7.66** loops/100 steps (gap **+7.11**; any-loop run share gap **+25.34 pp**). After capability-band matching (1 overlapping band(s)), the gap is **+6.86** `[+3.79, +10.08]` loops/100 steps (**higher for Open-Weights**).
- **Terminal-Bench 2.0 (mini-swe-agent)**: Before matching, Open averaged **19.99** vs Closed **11.91** loops/100 steps (gap **+8.08**; any-loop run share gap **-2.74 pp**). After capability-band matching (1 overlapping band(s)), the gap is **+0.50** `[-10.21, +12.85]` loops/100 steps (**statistically indistinguishable from zero (null)**).
- **SWE-rebench Jul 2026 (swelike)**: Before matching, Open averaged **5.71** vs Closed **0.98** loops/100 steps (gap **+4.73**; any-loop run share gap **+59.29 pp**). After capability-band matching (2 overlapping band(s)), the gap is **+2.54** `[+2.30, +2.78]` loops/100 steps (**higher for Open-Weights**).
- **Longtail-SWE (swelike)**: Before matching, Open averaged **3.42** vs Closed **0.96** loops/100 steps (gap **+2.45**; any-loop run share gap **+56.41 pp**). After capability-band matching (1 overlapping band(s)), the gap is **+0.08** `[-0.22, +0.36]` loops/100 steps (**statistically indistinguishable from zero (null)**).
- **Sierra tau2 Classic (airline+retail+telecom)**: Before matching, Open averaged **0.42** vs Closed **0.76** loops/100 steps (gap **-0.34**; any-loop run share gap **-3.41 pp**). After capability-band matching (1 overlapping band(s)), the gap is **-0.32** `[-0.59, -0.04]` loops/100 steps (**reversed (lower for Open-Weights)**).
- **Sierra tau2 Banking Knowledge (AllTools)**: Before matching, Open averaged **2.70** vs Closed **3.83** loops/100 steps (gap **-1.13**; any-loop run share gap **+9.84 pp**). After capability-band matching (1 overlapping band(s)), the gap is **+0.82** `[+0.38, +1.42]` loops/100 steps (**higher for Open-Weights**).

### Longtail-SWE context & adjacent-frontier check

- Closest capability neighbors in Longtail-SWE (`swelike`): `kimi-k3` (Open, `262144` ctx, 63.70% pass -> 1.27 loops/100); `deepseek-v4-flash` (Open, `1048576` ctx, 61.04% pass -> 2.06 loops/100); `gpt-5.6-sol` (Closed, `1050000` ctx, 75.74% pass -> 1.00 loops/100); `gpt-5.6-luna` (Closed, `1050000` ctx, 71.45% pass -> 1.58 loops/100).

## Model families, including Google Gemini

Aggregation into binary `Open` vs `Closed` buckets hides huge variance inside *both* camps. Here is how individual families behaved across every primary uniform scaffold:

### Google Gemini across Terminal-Bench and Sierra τ²

| Benchmark & Uniform Scaffold | Model | Executed Pass Rate | Loops / 100 Steps [95% CI] | Any-Loop Run Rate |
| --- | --- | ---: | ---: | ---: |
| `tau2_banking_knowledge_alltools` | `gemini-2.5-pro` | 12.76% | 2.91 [1.58, 5.03] | 27.08% |
| `tau2_banking_knowledge_alltools` | `gemini-3.1-pro` | 22.54% | 7.80 [6.05, 9.63] | 48.45% |
| `tau2_classic` | `gemini-3-flash` | 80.94% | 1.92 [1.59, 2.27] | 17.90% |
| `tau2_classic` | `gemini-3-pro` | 84.82% | 0.15 [0.05, 0.33] | 1.47% |
| `tb_mini_swe_agent` | `gemini-2.5-flash` | 17.59% | 30.64 [20.74, 39.09] | 49.31% |
| `tb_mini_swe_agent` | `gemini-2.5-pro` | 26.67% | 17.81 [10.54, 26.12] | 41.38% |
| `tb_openhands` | `gemini-2.5-flash` | 16.74% | 9.98 [8.25, 12.00] | 59.86% |
| `tb_openhands` | `gemini-2.5-pro` | 16.74% | 17.19 [11.59, 23.46] | 57.34% |
| `tb_terminus_2` | `gemini-2.5-flash` | 17.86% | 4.68 [1.75, 8.65] | 14.76% |
| `tb_terminus_2` | `gemini-2.5-pro` | 33.41% | 2.49 [1.34, 4.18] | 11.37% |
| `tb_terminus_2` | `gemini-3-flash` | 52.59% | 4.53 [1.03, 9.94] | 12.03% |
| `tb_terminus_2` | `gemini-3-pro` | 58.82% | 0.94 [0.42, 1.57] | 6.82% |
| `tau2_bk_terminal_use` (`terminal_use` scaffold) | `gemini-3-flash` | 20.62% | 10.03 | 76.55% |
| `tau2_bk_terminal_use` (`terminal_use` scaffold) | `gemini-3-pro` | 15.72% | 16.16 | 57.22% |

### Highest and lowest looping families per cohort

- **Terminal-Bench 2.0 (terminus-2)**: lowest loop density `deepseek` (Open, 0.82/100 @ 41.76% pass); highest `qwen (open)` (Open, 13.33/100 @ 25.23% pass).
- **Terminal-Bench 2.0 (openhands)**: lowest loop density `gpt` (Closed, 5.40/100 @ 28.48% pass); highest `qwen (open)` (Open, 16.03/100 @ 25.98% pass).
- **Terminal-Bench 2.0 (mini-swe-agent)**: lowest loop density `gpt` (Closed, 8.51/100 @ 27.00% pass); highest `gemini` (Closed, 24.22/100 @ 22.13% pass).
- **SWE-rebench Jul 2026 (swelike)**: lowest loop density `grok` (Closed, 0.65/100 @ 63.78% pass); highest `qwen (open)` (Open, 7.76/100 @ 24.32% pass).
- **Longtail-SWE (swelike)**: lowest loop density `claude` (Closed, 0.45/100 @ 81.52% pass); highest `qwen (open)` (Open, 5.27/100 @ 29.70% pass).
- **Sierra tau2 Classic (airline+retail+telecom)**: lowest loop density `claude` (Closed, 0.16/100 @ 81.52% pass); highest `gpt` (Closed, 1.09/100 @ 77.92% pass).
- **Sierra tau2 Banking Knowledge (AllTools)**: lowest loop density `claude` (Closed, 1.78/100 @ 36.54% pass); highest `grok` (Closed, 7.97/100 @ 23.02% pass).

## Sensitivity checks: passing runs, failing runs, legacy checkpoints & hosting

- **Passing runs only vs failing runs only**: Restricting strictly to solved tasks (`pass_only`) vs failed attempts (`fail_only`) tests whether any gap comes from verification habits on winning trajectories or flailing on unsolved ones — full cohort CIs sit in [`results/open_vs_closed_model.md`](results/open_vs_closed_model.md).
- **Dropping legacy `gpt-oss-20b/120b`**: Exposes how much of Terminal-Bench's raw open-weight loopiness comes from weak 3–19%-pass legacy models rather than modern open releases.
- **First-party routes only**: Only `tb_terminus_2` includes native first-party Open endpoints (`deepseek-v3.2`, `glm-5`, `minimax-m2.5`); every Open model in Sierra `AllTools` (`glm-5-2`, `inkling`, `kimi-k3`) ran via OpenRouter proxies (`third_party`), underscoring that third-party inference routing and quantization remain intertwined with open-weight evaluations.

## Dollar cost and loop detector implications

From our earlier cache-aware billing audit (unchanged), loop steps account for **17.71%** of validated spend on Terminal-Bench 2.0 and **2.87%** on SWE-rebench overall (with late-context bloat doubling dollar share vs step share within models), while rotating/A-B-A cycles slip past naive 5-adjacent-step guards (`277/334` missed on TB; `118/120` on SWE-rebench).

The fair open-vs-closed evidence points to three practical engineering conclusions:
1. **Scaffold & budget discipline beats weight stereotypes**: Harness step caps, rolling-window cycle detection (`openclaw_10`-style multi-period memory), and adequate context windows cut loop spend across *both* open and closed models far more reliably than choosing closed weights alone.
2. **Watch model pass rate and hosting route before blaming weights**: On equal scaffolds and capability bands, strong open models frequently match or beat closed peers in loop discipline, while weaker models in *either* camp (e.g., `gemini-2.5-flash`, `gpt-5-nano`, `gpt-oss`, `qwen3.6`) loop aggressively when stuck.
3. **Audit retrieval/proxy metadata in public leaderboards**: Silent differences in toolsets (`AllTools` vs `terminal_use`), context truncation (`262K` vs `1M`), and OpenRouter/FP8 hosting can swing loop rates by multiples inside the same benchmark folder.

## Visuals

- `assets/loops_per_100_by_model_band.png` — Loops per 100 steps (95% CI) by model and capability band.
- `assets/open_closed_gap_before_after_matching.png` — Open-minus-Closed gap before vs after pass-rate band matching.
- `assets/open_vs_closed_across_four_datasets.png` — Pass rate vs loop rate across all four datasets.
- Full per-model table: [`results/loops_by_model_scaffold.csv`](results/loops_by_model_scaffold.csv) and detailed stats: [`results/open_vs_closed_model.md`](results/open_vs_closed_model.md).

