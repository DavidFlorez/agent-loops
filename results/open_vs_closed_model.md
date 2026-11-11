# Open-Weights vs Closed Models: Fair Within-Scaffold Loop Analysis

Generated programmatically (CPU-only, fixed seed `42`, `1,000` task-clustered bootstrap replicates, unchanged step labeler). Negative differences (`Open - Closed < 0`) indicate **reversed** gaps where Open-Weights looped *less* than Closed models; confidence intervals crossing `0` are reported neutrally as **null**.

## 1. Fair Uniform-Scaffold Cohorts & Denominators

| Cohort | Dataset & Scaffold | Shared Tasks | Open Models (Exec Runs) | Closed Models (Exec Runs) | Zero-Step Crashes | Step Cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tb_terminus_2` | Terminal-Bench 2.0 (terminus-2) | 85 | 11 (7,654) | 18 (7,563) | 43 | 200 |
| `tb_openhands` | Terminal-Bench 2.0 (openhands) | 89 | 2 (1,730) | 10 (4,318) | 7 | none observed |
| `tb_mini_swe_agent` | Terminal-Bench 2.0 (mini-swe-agent) | 89 | 2 (1,770) | 11 (4,722) | 3 | none observed |
| `swe_rebench_swelike` | SWE-rebench Jul 2026 (swelike) | 111 | 7 (3,885) | 6 (3,330) | 0 | 500 |
| `longtail_swe_swelike` | Longtail-SWE (swelike) | 202 | 5 (2,981) | 4 (2,424) | 49 | 500 |
| `tau2_classic` | Sierra tau2 Classic (airline+retail+telecom) | 278 | 2 (2,168) | 6 (6,596) | 132 | 200 |
| `tau2_banking_knowledge_alltools` | Sierra tau2 Banking Knowledge (AllTools) | 97 | 3 (1,164) | 17 (6,492) | 104 | 200 |

- Excluded from Open-vs-Closed pooling: custom closed TB harnesses (`claude-code`, `codex`, `gemini-cli`, etc.), TB null-step rows (`glm-4.7@z-ai`, `gpt-5.3-codex@openai`, `kimi-k2.5@kimi`), SWE-rebench agentic harnesses (`claude-code`, `codex`, `cursor`, `junie`), Sierra `gpt-5_sierra_2025-08-09` (`gpt-4.1` user simulator), and the 6 non-`AllTools` Sierra `banking_knowledge` submissions (`terminal_use` / `embeddings`).

## 2. Before vs After Capability-Band Matching (Equal-Model Cohort Means, 95% Task-Clustered CI)

### 2A. Loops per 100 Steps & Any-Loop Run Rate (`Open - Closed`)

| Cohort | Unmatched Loops/100 (Open vs Closed) | Unmatched Gap [95% CI] | Overlap Bands | Matched Gap Loops/100 [95% CI] | Unmatched Any-Loop Gap (pp) | Matched Any-Loop Gap (pp) [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tb_terminus_2` | 6.91 vs 2.69 | +4.22 [+2.83, +5.58] | 4 | +4.20 [+2.86, +5.49] | +16.25 | +16.93 [+13.98, +20.08] |
| `tb_openhands` | 14.76 vs 7.66 | +7.11 [+4.04, +10.23] | 1 | +6.86 [+3.79, +10.08] | +25.34 | +28.13 [+23.46, +33.40] |
| `tb_mini_swe_agent` | 19.99 vs 11.91 | +8.08 [-1.90, +20.06] | 1 | +0.50 [-10.21, +12.85] | -2.74 | +2.00 [-2.03, +6.24] |
| `swe_rebench_swelike` | 5.71 vs 0.98 | +4.73 [+4.46, +5.00] | 2 | +2.54 [+2.30, +2.78] | +59.29 | +54.71 [+51.89, +57.89] |
| `longtail_swe_swelike` | 3.42 vs 0.96 | +2.45 [+2.13, +2.79] | 1 | +0.08 [-0.22, +0.36] | +56.41 | +24.29 [+19.32, +29.04] |
| `tau2_classic` | 0.42 vs 0.76 | -0.34 [-0.61, -0.04] | 1 | -0.32 [-0.59, -0.04] | -3.41 | -3.36 [-4.31, -2.37] |
| `tau2_banking_knowledge_alltools` | 2.70 vs 3.83 | -1.13 [-2.65, +0.62] | 1 | +0.82 [+0.38, +1.42] | +9.84 | +9.58 [+4.68, +13.66] |

### 2B. Capability-Band Breakdown (Overlapping Pass-Rate Strata)

| Cohort | Pass-Rate Band | Open Models | Closed Models | Loops/100 (Open vs Closed) | Gap [95% CI] | Any-Loop Gap (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tb_terminus_2` | B1 (<25%) | 2 | 4 | 4.08 vs 5.41 | -1.33 [-3.57, +1.18] | -0.30 |
| `tb_terminus_2` | B2 (25-40%) | 6 | 6 | 9.36 vs 2.26 | +7.10 [+5.14, +9.26] | +27.51 |
| `tb_terminus_2` | B3 (40-55%) | 2 | 4 | 2.85 vs 2.12 | +0.73 [-0.44, +1.76] | +7.53 |
| `tb_terminus_2` | B4 (55-75%) | 1 | 4 | 6.02 vs 1.21 | +4.81 [+1.14, +9.84] | +6.70 |
| `tb_openhands` | B2 (25-40%) | 2 | 3 | 14.76 vs 7.90 | +6.86 [+3.79, +10.08] | +28.13 |
| `tb_mini_swe_agent` | B1 (<25%) | 2 | 3 | 19.99 vs 19.48 | +0.50 [-10.21, +12.85] | +2.00 |
| `swe_rebench_swelike` | B3 (40-55%) | 3 | 1 | 4.98 vs 0.57 | +4.41 [+4.05, +4.81] | +82.58 |
| `swe_rebench_swelike` | B4 (55-75%) | 1 | 5 | 1.72 vs 1.06 | +0.66 [+0.45, +0.90] | +26.85 |
| `longtail_swe_swelike` | B4 (55-75%) | 2 | 1 | 1.66 vs 1.58 | +0.08 [-0.22, +0.36] | +24.29 |
| `tau2_classic` | B5 (>=75%) | 2 | 5 | 0.42 vs 0.75 | -0.32 [-0.59, -0.04] | -3.36 |
| `tau2_banking_knowledge_alltools` | B2 (25-40%) | 3 | 2 | 2.70 vs 1.87 | +0.82 [+0.38, +1.42] | +9.58 |

## 3. Statistical Regression Models (Adjusted for Task, Capability, Vintage & Serving Route)

Count models fit **Negative Binomial GLM** with `log(n_steps)` offset (`IRR > 1` means higher loop rate per step for Open); binary models fit **Logit** for `any_loop` (`OR > 1` means higher probability of >=1 loop), with task fixed effects and **two-way Cameron-Gelbach-Miller cluster-robust SEs** on `(model, task)`. Hierarchical checks add task-clustered Poisson GEE and sparse random-intercept `BinomialBayesMixedGLM` (`1 | task_id`) over 100% of executed trajectories.

| Cohort | NB Offset `is_open` IRR [95% CI] | p (2-way) | Logit FE `is_open` OR [95% CI] | p (2-way) | Logit+Steps OR (p) | PassRate (+10pp) IRR (p) | Omitted Covariates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tb_terminus_2` | 1.56 [0.64, 3.81] | 0.3266 | 3.20 [0.82, 12.50] | 0.0946 | 1.45 (0.5113) | 0.89 (0.4782) | none |
| `tb_openhands` | 1.77 [1.07, 2.93] | 0.0272 | 4.35 [1.95, 9.70] | 0.0003 | 3.26 (0.0543) | 1.19 (0.1222) | is_third_party (collinear R2=1.000) |
| `tb_mini_swe_agent` | 1.01 [0.67, 1.53] | 0.9503 | 1.14 [0.58, 2.25] | 0.6976 | 2.04 (0.0942) | 0.77 (0.0002) | is_third_party (collinear R2=1.000) |
| `swe_rebench_swelike` | 2.69 [1.46, 4.95] | 0.0014 | 8.43 [1.36, 52.12] | 0.0218 | 1.34 (0.4919) | 0.78 (0.0069) | is_third_party (uniform gateway) |
| `longtail_swe_swelike` | 1.08 [0.59, 2.00] | 0.7974 | 4.21 [0.56, 31.82] | 0.1635 | 0.39 (0.1353) | 0.81 (0.0473) | is_third_party (collinear R2=1.000) |
| `tau2_classic` | 0.03 [0.00, 0.54] | 0.0167 | 0.02 [0.00, 1.07] | 0.0536 | 0.05 (0.0025) | 0.94 (0.9089) | is_third_party (collinear R2=1.000) |
| `tau2_banking_knowledge_alltools` | 0.90 [0.34, 2.37] | 0.8368 | 1.34 [0.46, 3.84] | 0.5910 | 1.18 (0.7360) | 0.79 (0.3747) | none |
| `pooled_within_scaffold` | 1.31 [0.81, 2.13] | 0.2765 | 2.25 [1.11, 4.53] | 0.0236 | 1.47 (0.2480) | 0.86 (0.0122) | step_cap_100_c (absorbed by cohort FE; tested separately in step_cap spec) |

### 3B. Additional Covariates & Mixed-Effects / GEE Task-Cluster Robustness

| Cohort | NB Release (+1mo) IRR (p) | NB Third-Party IRR (p) | Poisson GEE `is_open` IRR (p_task) | Binomial GLMM Random-Task OR (p_vb) | Task RE SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tb_terminus_2` | 0.958 (0.6065) | 1.39 (0.4411) | failed | 3.17 (0.0e+00) | 0.87 |
| `tb_openhands` | 0.797 (0.0234) | omitted | failed | 4.24 (1.7e-115) | 1.13 |
| `tb_mini_swe_agent` | 0.865 (0.0001) | omitted | failed | 1.14 (0.0146) | 0.78 |
| `swe_rebench_swelike` | 0.978 (0.5378) | omitted | failed | 8.03 (0.0e+00) | 0.56 |
| `longtail_swe_swelike` | 0.779 (0.0112) | omitted | failed | 3.83 (1.4e-147) | 0.73 |
| `tau2_classic` | 3.107 (0.0405) | omitted | failed | 0.02 (3.4e-156) | 1.32 |
| `tau2_banking_knowledge_alltools` | 1.015 (0.8668) | 1.32 (0.2963) | failed | 1.33 (6.5e-05) | 1.53 |
| `pooled_within_scaffold` | 0.960 (0.2065) | 1.44 (0.1170) | failed | 2.61 (0.0e+00) | 0.93 |

- **Pooled step-cap covariate (separate non-cohort-FE specification)**: `+100` step cap IRR = **1.22** (`[1.13, 1.32]`, `p=3.8e-07`), while `is_open_weight` IRR = **1.23** (`p=0.4978`).

## 4. Pre-Registered Sensitivity Checks

### 4A. Pass-Only, Fail-Only, Drop `gpt-oss`, and First-Party-Only (Open - Closed Gap, Loops/100 Steps [95% CI])

| Cohort | Baseline Matched | `pass_only` (Matched) | `fail_only` (Matched) | `drop_gpt_oss` (Matched) | `first_party_only` (Matched / Unmatched) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tb_terminus_2` | +4.20 [+2.86, +5.49] | +2.78 [+1.62, +4.13] | +4.66 [+3.16, +6.34] | +5.43 [+3.78, +6.92] | +2.09 [+0.42, +3.86] |
| `tb_openhands` | +6.86 [+3.79, +10.08] | +4.76 [+0.62, +9.59] | +7.51 [+4.16, +10.86] | +6.86 [+3.81, +10.10] | n/a (single weights) |
| `tb_mini_swe_agent` | +0.50 [-10.21, +12.85] | +3.54 [-4.61, +13.47] | +0.29 [-10.96, +12.08] | n/a (single weights) | n/a (single weights) |
| `swe_rebench_swelike` | +2.54 [+2.30, +2.78] | +2.78 [+2.46, +3.10] | +2.35 [+1.99, +2.71] | +2.54 [+2.29, +2.80] | n/a (single weights) |
| `longtail_swe_swelike` | +0.08 [-0.22, +0.36] | +0.11 [-0.16, +0.37] | +0.01 [-0.81, +0.64] | +0.08 [-0.22, +0.37] | n/a (single weights) |
| `tau2_classic` | -0.32 [-0.59, -0.04] | -0.34 [-0.64, -0.03] | -0.19 [-0.83, +0.66] | -0.32 [-0.59, -0.04] | n/a (single weights) |
| `tau2_banking_knowledge_alltools` | +0.82 [+0.38, +1.42] | +0.59 [+0.12, +1.07] | +0.86 [+0.19, +1.82] | +0.82 [+0.34, +1.46] | n/a (single weights) |

### 4B. Per-Family Breakdown Across All Cohorts (Every Vendor Family Including Google Gemini)

| Cohort | Family | Weights | Models | Exec Runs | Mean Pass | Mean Loops/100 | Mean Any-Loop | Mean Loop $ Share |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `longtail_swe_swelike` | `qwen (open)` | Open | 2 | 1,212 | 29.70% | 5.27 | 87.87% | 5.08% |
| `longtail_swe_swelike` | `glm` | Open | 1 | 606 | 48.18% | 3.22 | 86.80% | 1.34% |
| `longtail_swe_swelike` | `deepseek` | Open | 1 | 557 | 61.04% | 2.06 | 90.66% | 1.37% |
| `longtail_swe_swelike` | `kimi` | Open | 1 | 606 | 63.70% | 1.27 | 61.88% | 1.52% |
| `longtail_swe_swelike` | `gpt` | Closed | 3 | 1,818 | 75.37% | 1.13 | 30.53% | 1.11% |
| `longtail_swe_swelike` | `claude` | Closed | 1 | 606 | 81.52% | 0.45 | 14.85% | 0.35% |
| `swe_rebench_swelike` | `qwen (open)` | Open | 3 | 1,665 | 24.32% | 7.76 | 96.46% | 6.28% |
| `swe_rebench_swelike` | `minimax` | Open | 1 | 555 | 47.21% | 5.95 | 99.28% | 7.90% |
| `swe_rebench_swelike` | `mimo` | Open | 1 | 555 | 46.49% | 4.52 | 90.09% | 3.94% |
| `swe_rebench_swelike` | `deepseek` | Open | 1 | 555 | 40.18% | 4.49 | 85.41% | 4.30% |
| `swe_rebench_swelike` | `glm` | Open | 1 | 555 | 62.88% | 1.72 | 61.08% | 1.40% |
| `swe_rebench_swelike` | `claude` | Closed | 3 | 1,665 | 61.56% | 1.26 | 44.68% | 1.57% |
| `swe_rebench_swelike` | `gpt` | Closed | 2 | 1,110 | 52.97% | 0.71 | 11.44% | 0.64% |
| `swe_rebench_swelike` | `grok` | Closed | 1 | 555 | 63.78% | 0.65 | 23.24% | 0.82% |
| `tau2_banking_knowledge_alltools` | `grok` | Closed | 4 | 1,550 | 23.02% | 7.97 | 26.12% | 1.89% |
| `tau2_banking_knowledge_alltools` | `gemini` | Closed | 2 | 770 | 17.65% | 5.35 | 37.76% | 5.35% |
| `tau2_banking_knowledge_alltools` | `kimi` | Open | 1 | 388 | 37.11% | 2.80 | 35.82% | 3.62% |
| `tau2_banking_knowledge_alltools` | `inkling` | Open | 1 | 388 | 25.00% | 2.69 | 53.35% | 3.39% |
| `tau2_banking_knowledge_alltools` | `qwen (closed Max)` | Closed | 1 | 388 | 55.15% | 2.68 | 38.66% | 3.37% |
| `tau2_banking_knowledge_alltools` | `glm` | Open | 1 | 388 | 37.11% | 2.59 | 31.19% | 3.96% |
| `tau2_banking_knowledge_alltools` | `gpt` | Closed | 4 | 1,509 | 36.88% | 2.28 | 34.78% | 1.77% |
| `tau2_banking_knowledge_alltools` | `claude` | Closed | 6 | 2,275 | 36.54% | 1.78 | 26.17% | 1.76% |
| `tau2_classic` | `gpt` | Closed | 2 | 2,173 | 77.92% | 1.09 | 6.26% | 0.84% |
| `tau2_classic` | `gemini` | Closed | 2 | 2,199 | 82.88% | 1.04 | 9.68% | 1.20% |
| `tau2_classic` | `glm` | Open | 1 | 1,056 | 84.94% | 0.46 | 1.70% | 0.67% |
| `tau2_classic` | `qwen (open)` | Open | 1 | 1,112 | 89.39% | 0.39 | 3.24% | 0.56% |
| `tau2_classic` | `claude` | Closed | 2 | 2,224 | 81.52% | 0.16 | 1.71% | 0.20% |
| `tb_mini_swe_agent` | `gemini` | Closed | 2 | 867 | 22.13% | 24.22 | 45.34% | n<100 |
| `tb_mini_swe_agent` | `gpt_oss` | Open | 2 | 1,770 | 8.82% | 19.99 | 30.68% | 47.52% |
| `tb_mini_swe_agent` | `grok` | Closed | 2 | 846 | 26.97% | 11.12 | 46.71% | 7.14% |
| `tb_mini_swe_agent` | `claude` | Closed | 3 | 1,276 | 37.36% | 8.76 | 35.48% | 20.47% |
| `tb_mini_swe_agent` | `gpt` | Closed | 4 | 1,733 | 27.00% | 8.51 | 19.25% | n/a |
| `tb_openhands` | `qwen (open)` | Open | 1 | 870 | 25.98% | 16.03 | 71.15% | 15.45% |
| `tb_openhands` | `gemini` | Closed | 2 | 872 | 16.74% | 13.59 | 58.60% | n<100 |
| `tb_openhands` | `kimi` | Open | 1 | 860 | 27.67% | 13.50 | 81.05% | 15.28% |
| `tb_openhands` | `grok` | Closed | 1 | 421 | 28.74% | 8.82 | 41.57% | n<100 |
| `tb_openhands` | `claude` | Closed | 4 | 1,734 | 36.94% | 6.09 | 54.03% | n<100 |
| `tb_openhands` | `gpt` | Closed | 3 | 1,291 | 28.48% | 5.40 | 44.25% | n<100 |
| `tb_terminus_2` | `qwen (open)` | Open | 1 | 888 | 25.23% | 13.33 | 55.07% | 20.24% |
| `tb_terminus_2` | `glm` | Open | 2 | 1,226 | 41.42% | 10.68 | 30.92% | 29.06% |
| `tb_terminus_2` | `kimi` | Open | 2 | 1,735 | 33.39% | 7.03 | 38.06% | 11.42% |
| `tb_terminus_2` | `minimax` | Open | 3 | 1,667 | 36.41% | 6.10 | 31.75% | 17.11% |
| `tb_terminus_2` | `gpt_oss` | Open | 2 | 1,774 | 11.64% | 4.08 | 11.17% | 7.51% |
| `tb_terminus_2` | `gpt` | Closed | 7 | 2,927 | 37.27% | 3.21 | 8.64% | n<100 |
| `tb_terminus_2` | `gemini` | Closed | 4 | 1,691 | 40.67% | 3.16 | 11.25% | n<100 |
| `tb_terminus_2` | `claude` | Closed | 5 | 2,110 | 47.23% | 2.20 | 22.79% | 6.35% |
| `tb_terminus_2` | `grok` | Closed | 2 | 835 | 19.59% | 1.17 | 4.89% | 0.90% |
| `tb_terminus_2` | `deepseek` | Open | 1 | 364 | 41.76% | 0.82 | 8.24% | n<100 |

### 4C. Google Gemini Detailed Profile Across Uniform Scaffolds & Sierra Variants

| Cohort / Scaffold | Model | Pass Rate | Loops/100 [95% CI] | Any-Loop Rate [95% CI] | Loop $ Share |
| --- | --- | ---: | ---: | ---: | ---: |
| `tau2_banking_knowledge_alltools` | `gemini-2-5-pro` | 12.76% | 2.91 [1.58, 5.03] | 27.08% [21.99%, 32.81%] | 2.79% |
| `tau2_banking_knowledge_alltools` | `gemini-3-1-pro-preview` | 22.54% | 7.80 [6.05, 9.63] | 48.45% [41.86%, 55.06%] | 7.91% |
| `tau2_classic` | `gemini-3-flash` | 80.94% | 1.92 [1.59, 2.27] | 17.90% [14.84%, 21.22%] | 2.17% |
| `tau2_classic` | `gemini-3-pro` | 84.82% | 0.15 [0.05, 0.33] | 1.47% [0.64%, 2.66%] | 0.23% |
| `tb_mini_swe_agent` | `gemini-2.5-flash@gemini` | 17.59% | 30.64 [20.74, 39.09] | 49.31% [42.05%, 55.99%] | n<100 |
| `tb_mini_swe_agent` | `gemini-2.5-pro@gemini` | 26.67% | 17.81 [10.54, 26.12] | 41.38% [35.25%, 48.01%] | n/a |
| `tb_openhands` | `gemini-2.5-flash@gemini` | 16.74% | 9.98 [8.25, 12.00] | 59.86% [52.41%, 68.04%] | n<100 |
| `tb_openhands` | `gemini-2.5-pro@gemini` | 16.74% | 17.19 [11.59, 23.46] | 57.34% [49.89%, 64.68%] | n<100 |
| `tb_terminus_2` | `gemini-2.5-flash@gemini` | 17.86% | 4.68 [1.75, 8.65] | 14.76% [10.26%, 19.76%] | n<100 |
| `tb_terminus_2` | `gemini-2.5-pro@gemini` | 33.41% | 2.49 [1.34, 4.18] | 11.37% [7.55%, 15.40%] | n/a |
| `tb_terminus_2` | `gemini-3-flash-preview@gemini` | 52.59% | 4.53 [1.03, 9.94] | 12.03% [8.04%, 17.02%] | n/a |
| `tb_terminus_2` | `gemini-3-pro-preview@gemini` | 58.82% | 0.94 [0.42, 1.57] | 6.82% [3.76%, 10.35%] | n/a |
| `tau2_bk_terminal_use` (`terminal_use`, excluded from AllTools) | `gemini-3-flash` | 20.62% | 10.03 | 76.55% | n/a |
| `tau2_bk_terminal_use` (`terminal_use`, excluded from AllTools) | `gemini-3-pro` | 15.72% | 16.16 | 57.22% | n/a |

## 5. Open vs Closed Dollar Share & Longtail Context Confounders

*Cells with fewer than 100 cost-validated runs are suppressed as `n<100` (e.g., `tb_openhands` Closed has only 47 cost-valid runs; Terminal-Bench Closed has only 800 cost-validated runs across all harnesses combined).*

| Cohort | Open Cost-Valid Runs | Open Mean Loop $ Share | Closed Cost-Valid Runs | Closed Mean Loop $ Share | Cost Provenance Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `tb_terminus_2` | 2,863 | 13.34% | 290 | 5.52% | `reported_usd_rebuilt` |
| `tb_openhands` | 417 | 15.36% | 47 | n<100 | `reported_usd_rebuilt` |
| `tb_mini_swe_agent` | 826 | 47.52% | 438 | 27.83% | `reported_usd_rebuilt` |
| `swe_rebench_swelike` | 3,652 | 5.20% | 2,440 | 1.14% | `tokens_rebuilt_list_rates` |
| `longtail_swe_swelike` | 2,872 | 2.88% | 1,922 | 0.92% | `tokens_rebuilt_list_rates` |
| `tau2_classic` | 2,168 | 0.61% | 6,596 | 0.75% | `rebuilt_from_turn_tokens_uncached, reported_provider_usd` |
| `tau2_banking_knowledge_alltools` | 1,164 | 3.65% | 6,492 | 2.31% | `rebuilt_from_turn_tokens_uncached, reported_provider_usd` |

### Longtail-SWE Context Limit & Adjacent Frontier Pair Check (`swelike`)

| Model | Weights | Context Limit | Exec Runs (Zero-Step Drops) | Exec Pass | Loops/100 | Any-Loop Rate | Loop $ Share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `claude-opus-5.5` | Closed | `1000000` | 606 (0) | 81.52% | 0.45 | 14.85% | 0.35% |
| `gpt-6.1-sol` | Closed | `1050000` | 606 (0) | 78.90% | 0.82 | 15.68% | 0.82% |
| `gpt-5.6-sol` | Closed | `1050000` | 606 (0) | 75.74% | 1.00 | 23.93% | 0.96% |
| `gpt-5.6-luna` | Closed | `1050000` | 606 (0) | 71.45% | 1.58 | 51.98% | 1.55% |
| `kimi-k3` | Open | `262144` | 606 (0) | 63.70% | 1.27 | 61.88% | 1.52% |
| `deepseek-v4-flash` | Open | `1048576` | 557 (49) | 61.04% | 2.06 | 90.66% | 1.37% |
| `glm-5.3-flash` | Open | `262144` | 606 (0) | 48.18% | 3.22 | 86.80% | 1.34% |
| `qwen3.8-27b` | Open | `262144` | 606 (0) | 42.74% | 1.59 | 79.54% | 1.33% |
| `qwen3.6-27b` | Open | `131072,262144` | 606 (0) | 16.67% | 8.95 | 96.20% | 8.83% |

### Sierra `banking_knowledge` Non-`AllTools` Submissions (Excluded From Uniform `AllTools` Slice)

| Cohort | Retrieval Scaffold | Model | Weights | Exec Runs | Pass Rate | Loops/100 | Any-Loop Rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `tau2_bk_openai_embeddings` | `openai_embeddings` | `glm-5-think` | Open | 307 | 11.07% | 28.64 | 68.73% |
| `tau2_bk_openai_embeddings` | `openai_embeddings` | `qwen3.5-397b-a17b-think` | Open | 331 | 11.48% | 29.24 | 69.18% |
| `tau2_bk_qwen_embeddings` | `qwen_embeddings` | `gpt-5-2-none` | Closed | 388 | 12.37% | 3.11 | 18.30% |
| `tau2_bk_terminal_use` | `terminal_use` | `claude-sonnet-4-5` | Closed | 388 | 22.42% | 1.31 | 35.05% |
| `tau2_bk_terminal_use` | `terminal_use` | `gemini-3-flash` | Closed | 388 | 20.62% | 10.03 | 76.55% |
| `tau2_bk_terminal_use` | `terminal_use` | `gemini-3-pro` | Closed | 388 | 15.72% | 16.16 | 57.22% |



## 6. Phase 3 Robustness, Audit & Pre-Specified Cohort Verdicts

### 6A. Pre-Specified Cohort Verdict Rule & Two-Stage `(Model, Task)` Cluster Bootstrap

**Verdict rule**: A cohort is classified as **`higher for open`** only if the capability-matched gap 95% CI excludes `0` (`> 0`) **and** the adjusted NB IRR is statistically reliable and significant (`IRR > 1, p < 0.05`); **`lower for open`** for the mirror case; **`null`** if both the matched-gap CI includes `0` and `IRR p >= 0.05`; and **`mixed`** otherwise.

| Cohort | Open / Closed Models | Matched Gap Loops/100 | 1-Stage Task CI | 2-Stage (Model, Task) CI | Raw Rate Ratio [95% CI] | Adj NB IRR (p_2way) | IRR Status | Verdict (1-Stage / 2-Stage) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `tb_terminus_2` | 11 / 18 | +4.20 | [+2.86, +5.49] | [+1.44, +6.88] | 2.57 [1.17, 5.64] | 1.56 (0.3266) | `reliable` | **mixed** / **mixed** |
| `tb_openhands` | 2 / 10 | +6.86 | [+3.79, +10.08] | [+1.40, +12.43] | 1.93 [1.28, 3.01] | 1.77 (0.0272) | `reliable` | **higher for open** / **higher for open** |
| `tb_mini_swe_agent` | 2 / 11 | +0.50 | [-10.21, +12.85] | [-17.45, +26.66] | 1.68 [0.65, 4.10] | 1.01 (0.9503) | `reliable` | **null** / **null** |
| `swe_rebench_swelike` | 7 / 6 | +2.54 | [+2.30, +2.78] | [+2.04, +3.05] | 5.84 [3.57, 9.48] | 2.69 (0.0014) | `reliable` | **higher for open** / **higher for open** |
| `longtail_swe_swelike` | 5 / 4 | +0.08 | [-0.22, +0.36] | [-0.54, +0.69] | 3.55 [1.41, 8.41] | 1.08 (0.7974) | `reliable` | **null** / **null** |
| `tau2_classic` | 2 / 6 | -0.32 | [-0.59, -0.04] | [-1.15, +0.35] | 0.56 [0.22, 2.16] | 0.03 (0.0167) | `unreliable` | **mixed** / **null** |
| `tau2_banking_knowledge_alltools` | 3 / 17 | +0.82 | [+0.38, +1.42] | [-0.14, +1.79] | 0.70 [0.33, 1.55] | 0.90 (0.8368) | `reliable` | **mixed** / **null** |

- **`tau2_classic` NB IRR diagnosis (`0.03` vs raw ratio `0.56`)**: Complete separation on release_months + small N_open=2: both Open models (glm-5-think, qwen3.5-397b) were released in 2026-02 (month=13), whereas all 6 Closed models were released in 2025-09..2025-12 (months 8..11). Within Closed models, Dec-2025 releases (gemini-3-flash=1.92, gpt-5-2=1.20) looped much more than Sep/Nov-2025 Claude releases (0.15-0.17), driving the release_months slope to IRR=3.11/month (3.11^3 = 30x over 3 months) and forcing the collinear is_open_weight indicator down to IRR=0.03 to fit the 2 Open models at month=13. Dropping the collinear release_months_c covariate restores NB IRR to 0.60 (p=0.3920), matching the raw rate ratio (0.56).

### 6B. Default vs Strict Labeler Sensitivity (`swe_rebench_swelike` & `tb_terminus_2`)

The strict labeler counts only `exact_loop` repeated `>= 3` times in a row, `aba_thrash` lasting `>= 3` cycles, and `period_k_rotation`, dropping `near_duplicate`, single 2-step repeats, `redundant_reread`, and `error_streak`.

| Cohort | Metric | Default Open vs Closed (Unmatched) | Default Matched Gap [95% CI] | Strict Open vs Closed (Unmatched) | Strict Matched Gap [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: |
| `tb_terminus_2` | `loops_per_100_steps` | 6.91 vs 2.69 | +4.20 [+2.87, +5.46] | 1.12 vs 0.67 | +0.52 [+0.05, +1.00] |
| `tb_terminus_2` | `any_loop_rate` | 28.99% vs 12.73% | +16.93 [+14.02, +19.92] | 2.17% vs 0.97% | +1.19 [+0.60, +1.83] |
| `swe_rebench_swelike` | `loops_per_100_steps` | 5.71 vs 0.98 | +2.54 [+2.30, +2.78] | 0.13 vs 0.02 | +0.04 [+0.00, +0.09] |
| `swe_rebench_swelike` | `any_loop_rate` | 89.32% vs 30.03% | +54.71 [+51.85, +57.89] | 3.42% vs 0.57% | +0.62 [+0.05, +1.33] |

### 6C. Read-Only Command Share of `exact_loop` Flags (`swe_rebench_swelike`)

| Cohort | Weights | Exact-Loop Steps | Read-Only Steps (%) | Read/Search Tool | Shell Inspect (`ls/cat/grep`) | `git status/diff/log` | `pytest`/Test Re-run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `swe_rebench_swelike` | Open | 12,603 | 6,003 (47.6%) | 2,047 | 2,069 | 1,827 | 60 |
| `swe_rebench_swelike` | Closed | 501 | 244 (48.7%) | 7 | 14 | 223 | 0 |
| `tb_terminus_2` | Open | 17,149 | 1,884 (11.0%) | 535 | 609 | 30 | 710 |
| `tb_terminus_2` | Closed | 2,789 | 512 (18.4%) | 85 | 369 | 1 | 57 |
| `longtail_swe_swelike` | Open | 7,989 | 2,821 (35.3%) | 904 | 1,535 | 382 | 0 |
| `longtail_swe_swelike` | Closed | 155 | 1 (0.6%) | 0 | 0 | 1 | 0 |

### 6D. Run-Length Control: Raw vs Capability-Matched vs Capability+Length-Decile Matched

| Cohort | Open vs Closed Mean Steps | Raw Gap Loops/100 | Capability-Matched Gap | Length-Decile Matched Gap [95% CI] | Capability + Length Matched Gap [95% CI] | Plain Logit OR (p) -> Logit+Steps OR (p) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tb_terminus_2` | 33.1 vs 17.1 (1.94x) | +4.22 | +4.20 | +1.08 [+0.68, +1.47] | +0.99 [+0.62, +1.29] | 3.20 (0.095) -> 1.45 (0.511) |
| `tb_openhands` | 54.1 vs 48.6 (1.11x) | +7.11 | +6.86 | +3.63 [+2.49, +4.94] | +0.49 [-0.69, +1.78] | 4.35 (0.000) -> 3.26 (0.054) |
| `tb_mini_swe_agent` | 9.5 vs 25.7 (0.37x) | +8.08 | +0.50 | +6.28 [+4.94, +7.56] | +4.37 [+2.91, +5.80] | 1.14 (0.698) -> 2.04 (0.094) |
| `swe_rebench_swelike` | 94.7 vs 45.4 (2.09x) | +4.73 | +2.54 | +4.27 [+3.95, +4.58] | +0.77 [+0.50, +1.02] | 8.43 (0.022) -> 1.34 (0.492) |
| `longtail_swe_swelike` | 200.2 vs 47.4 (4.23x) | +2.45 | +0.08 | +2.88 [+2.48, +3.30] | -0.50 [-0.86, -0.25] | 4.21 (0.164) -> 0.39 (0.135) |
| `tau2_classic` | 13.6 vs 13.9 (0.98x) | -0.34 | -0.32 | -0.21 [-0.34, -0.04] | -0.17 [-0.30, -0.02] | 0.02 (0.054) -> 0.05 (0.002) |
| `tau2_banking_knowledge_alltools` | 38.1 vs 35.0 (1.09x) | -1.13 | +0.82 | -0.66 [-2.37, +0.36] | +0.62 [+0.22, +1.01] | 1.34 (0.591) -> 1.18 (0.736) |

### 6E. Same Model Across Different Harnesses (Shared Tasks Within Benchmark)

| Dataset | Canonical Model | Harness | Shared Tasks | Runs | Pass Rate | Any-Loop Rate [95% CI] | Loops/100 Steps [95% CI] | Strict Loops/100 | Model Spread (Max-Min Loops/100) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `terminalbench` | `gemini-2.5-flash` | `terminus-2` | 88 | 433 | 16.86% | 15.01% [10.34%, 19.82%] | 4.64 [1.79, 8.45] | 1.43 | 26.08 (6.6x) |
| `terminalbench` | `gemini-2.5-flash` | `gemini-cli` | 88 | 405 | 16.79% | 16.79% [11.75%, 22.14%] | 7.51 [4.09, 11.02] | 0.48 | 26.08 (6.6x) |
| `terminalbench` | `gemini-2.5-flash` | `openhands` | 88 | 431 | 16.94% | 59.86% [52.09%, 67.69%] | 10.01 [8.24, 11.97] | 0.42 | 26.08 (6.6x) |
| `terminalbench` | `gemini-2.5-flash` | `mini-swe-agent` | 88 | 429 | 17.72% | 49.18% [42.29%, 56.44%] | 30.72 [20.54, 39.28] | 10.33 | 26.08 (6.6x) |
| `terminalbench` | `gpt-oss-20b` | `terminus-2` | 89 | 930 | 3.01% | 8.71% [5.71%, 12.00%] | 3.61 [1.51, 6.33] | 3.31 | 24.31 (7.7x) |
| `terminalbench` | `gpt-oss-20b` | `mini-swe-agent` | 89 | 886 | 3.39% | 31.94% [26.58%, 37.36%] | 27.92 [10.00, 50.82] | 21.37 | 24.31 (7.7x) |
| `terminalbench` | `gpt-5-nano` | `codex` | 89 | 421 | 12.11% | 9.74% [6.50%, 13.08%] | 1.33 [0.83, 1.84] | 0.00 | 18.98 (15.3x) |
| `terminalbench` | `gpt-5-nano` | `openhands` | 89 | 427 | 10.30% | 57.61% [50.93%, 64.24%] | 5.51 [4.26, 6.88] | 0.52 | 18.98 (15.3x) |
| `terminalbench` | `gpt-5-nano` | `terminus-2` | 89 | 434 | 8.06% | 21.89% [16.31%, 27.61%] | 14.58 [8.78, 21.08] | 6.41 | 18.98 (15.3x) |
| `terminalbench` | `gpt-5-nano` | `mini-swe-agent` | 89 | 441 | 7.03% | 22.90% [18.51%, 27.73%] | 20.31 [11.22, 28.47] | 8.71 | 18.98 (15.3x) |
| `terminalbench` | `gemini-2.5-pro` | `terminus-2` | 88 | 436 | 32.57% | 11.70% [8.03%, 15.75%] | 2.51 [1.36, 4.14] | 0.03 | 15.27 (7.1x) |
| `terminalbench` | `gemini-2.5-pro` | `gemini-cli` | 88 | 381 | 22.31% | 28.61% [22.48%, 35.72%] | 5.98 [4.25, 8.00] | 0.58 | 15.27 (7.1x) |
| `terminalbench` | `gemini-2.5-pro` | `openhands` | 88 | 431 | 16.71% | 56.84% [50.46%, 63.36%] | 17.08 [11.47, 23.99] | 2.70 | 15.27 (7.1x) |
| `terminalbench` | `gemini-2.5-pro` | `mini-swe-agent` | 88 | 432 | 26.85% | 40.97% [33.94%, 47.36%] | 17.78 [10.22, 27.46] | 0.14 | 15.27 (7.1x) |
| `terminalbench` | `grok-code-fast-1` | `terminus-2` | 89 | 443 | 14.22% | 6.55% [4.04%, 9.46%] | 1.85 [0.70, 3.45] | 0.00 | 11.27 (7.1x) |
| `terminalbench` | `grok-code-fast-1` | `mini-swe-agent` | 89 | 436 | 26.38% | 46.10% [39.44%, 52.88%] | 13.12 [6.72, 22.75] | 0.06 | 11.27 (7.1x) |
| `terminalbench` | `claude-opus-4.1` | `terminus-2` | 88 | 436 | 38.30% | 8.72% [5.27%, 12.36%] | 1.03 [0.55, 1.71] | 0.00 | 10.23 (10.9x) |
| `terminalbench` | `claude-opus-4.1` | `claude-code` | 88 | 404 | 37.62% | 32.67% [25.31%, 39.55%] | 2.07 [1.47, 2.69] | 0.08 | 10.23 (10.9x) |
| `terminalbench` | `claude-opus-4.1` | `mini-swe-agent` | 88 | 424 | 36.79% | 29.95% [22.25%, 37.77%] | 8.15 [4.07, 12.82] | 0.03 | 10.23 (10.9x) |
| `terminalbench` | `claude-opus-4.1` | `openhands` | 88 | 436 | 37.61% | 72.02% [64.45%, 79.18%] | 11.27 [8.23, 15.47] | 0.00 | 10.23 (10.9x) |
| `terminalbench` | `grok-4` | `terminus-2` | 89 | 428 | 24.07% | 3.27% [1.17%, 5.99%] | 0.49 [0.20, 0.89] | 0.00 | 8.62 (18.5x) |
| `terminalbench` | `grok-4` | `openhands` | 89 | 421 | 28.74% | 41.57% [33.80%, 49.30%] | 8.82 [7.21, 10.54] | 0.04 | 8.62 (18.5x) |
| `terminalbench` | `grok-4` | `mini-swe-agent` | 89 | 410 | 27.56% | 47.32% [40.75%, 53.81%] | 9.11 [6.79, 11.53] | 0.10 | 8.62 (18.5x) |
| `terminalbench` | `gpt-oss-120b` | `terminus-2` | 89 | 924 | 19.26% | 14.39% [10.81%, 18.43%] | 4.57 [3.34, 5.78] | 0.87 | 7.48 (2.6x) |
| `terminalbench` | `gpt-oss-120b` | `mini-swe-agent` | 89 | 884 | 14.25% | 29.41% [23.58%, 35.49%] | 12.05 [7.38, 17.41] | 0.27 | 7.48 (2.6x) |
| `terminalbench` | `claude-haiku-4.5` | `openhands` | 87 | 410 | 14.39% | 29.51% [23.89%, 35.45%] | 1.65 [0.91, 2.81] | 0.12 | 6.65 (5.0x) |
| `terminalbench` | `claude-haiku-4.5` | `claude-code` | 87 | 403 | 30.02% | 30.27% [23.40%, 38.11%] | 2.27 [1.37, 3.34] | 0.06 | 6.65 (5.0x) |
| `terminalbench` | `claude-haiku-4.5` | `terminus-2` | 87 | 428 | 28.50% | 25.70% [19.86%, 32.00%] | 3.83 [2.52, 5.35] | 0.04 | 6.65 (5.0x) |
| `terminalbench` | `claude-haiku-4.5` | `mini-swe-agent` | 87 | 421 | 30.88% | 33.49% [26.60%, 40.92%] | 8.30 [4.52, 12.88] | 0.12 | 6.65 (5.0x) |
| `terminalbench` | `gpt-5` | `terminus-2` | 88 | 430 | 36.05% | 2.09% [0.69%, 3.99%] | 0.81 [0.20, 1.64] | 0.12 | 6.31 (8.8x) |
| `terminalbench` | `gpt-5` | `mini-swe-agent` | 88 | 429 | 35.20% | 6.06% [3.50%, 9.01%] | 1.10 [0.59, 1.76] | 0.00 | 6.31 (8.8x) |
| `terminalbench` | `gpt-5` | `codex` | 88 | 409 | 53.06% | 15.16% [9.87%, 21.55%] | 2.20 [1.27, 3.23] | 0.00 | 6.31 (8.8x) |
| `terminalbench` | `gpt-5` | `openhands` | 88 | 430 | 44.42% | 44.88% [38.48%, 51.73%] | 7.12 [5.83, 8.65] | 0.07 | 6.31 (8.8x) |
| `swe_rebench` | `claude-fable-5` | `swelike` | 111 | 555 | 64.50% | 25.41% [20.36%, 31.35%] | 0.71 [0.54, 0.91] | 0.00 | 5.69 (9.0x) |
| `swe_rebench` | `claude-fable-5` | `claude-code` | 111 | 555 | 60.36% | 31.89% [25.77%, 38.20%] | 6.41 [1.89, 12.12] | 4.70 | 5.69 (9.0x) |
| `terminalbench` | `gpt-5-mini` | `codex` | 89 | 410 | 33.66% | 15.37% [10.10%, 21.27%] | 1.88 [1.04, 2.78] | 0.00 | 5.62 (4.0x) |
| `terminalbench` | `gpt-5-mini` | `terminus-2` | 89 | 436 | 24.54% | 5.05% [2.96%, 7.31%] | 3.49 [1.17, 6.36] | 0.52 | 5.62 (4.0x) |
| `terminalbench` | `gpt-5-mini` | `openhands` | 89 | 429 | 30.30% | 30.07% [24.23%, 36.62%] | 3.60 [2.75, 4.63] | 0.00 | 5.62 (4.0x) |
| `terminalbench` | `gpt-5-mini` | `mini-swe-agent` | 89 | 434 | 22.81% | 13.82% [9.34%, 18.62%] | 7.50 [3.60, 12.41] | 1.74 | 5.62 (4.0x) |
| `terminalbench` | `gpt-5-codex` | `terminus-2` | 53 | 256 | 37.50% | 13.67% [7.90%, 20.31%] | 0.95 [0.57, 1.42] | 0.04 | 5.34 (6.6x) |
| `terminalbench` | `gpt-5-codex` | `codex` | 53 | 168 | 42.26% | 23.21% [13.77%, 32.96%] | 2.96 [1.29, 5.58] | 0.06 | 5.34 (6.6x) |
| `terminalbench` | `gpt-5-codex` | `mini-swe-agent` | 53 | 250 | 40.40% | 32.40% [23.99%, 41.20%] | 6.29 [2.75, 11.32] | 2.60 | 5.34 (6.6x) |
| `terminalbench` | `claude-sonnet-4.5` | `claude-code` | 88 | 404 | 42.82% | 28.22% [21.10%, 35.92%] | 2.21 [1.34, 3.24] | 0.25 | 4.69 (3.1x) |
| `terminalbench` | `claude-sonnet-4.5` | `terminus-2` | 88 | 435 | 43.22% | 25.75% [20.19%, 32.10%] | 2.34 [1.39, 3.52] | 0.06 | 4.69 (3.1x) |
| `terminalbench` | `claude-sonnet-4.5` | `openhands` | 88 | 434 | 42.86% | 61.52% [52.99%, 69.89%] | 5.23 [4.17, 6.41] | 0.00 | 4.69 (3.1x) |
| `terminalbench` | `claude-sonnet-4.5` | `mini-swe-agent` | 88 | 416 | 44.47% | 42.55% [35.21%, 49.76%] | 6.90 [4.44, 9.67] | 0.24 | 4.69 (3.1x) |
| `terminalbench` | `claude-opus-4.5` | `terminus-2` | 88 | 439 | 57.40% | 41.91% [35.45%, 48.17%] | 2.63 [1.91, 3.51] | 0.00 | 3.63 (2.4x) |
| `terminalbench` | `claude-opus-4.5` | `claude-code` | 88 | 436 | 53.21% | 30.50% [23.17%, 38.41%] | 2.90 [1.92, 3.87] | 0.17 | 3.63 (2.4x) |
| `terminalbench` | `claude-opus-4.5` | `openhands` | 88 | 433 | 52.66% | 53.12% [44.21%, 62.15%] | 6.26 [4.69, 7.93] | 0.05 | 3.63 (2.4x) |
| `terminalbench` | `kimi-k2-0905` | `terminus-2` | 89 | 923 | 27.95% | 48.00% [41.25%, 54.99%] | 9.94 [6.89, 13.31] | 0.05 | 3.55 (1.4x) |
| `terminalbench` | `kimi-k2-0905` | `openhands` | 89 | 860 | 27.67% | 81.05% [74.65%, 87.10%] | 13.50 [10.66, 17.37] | 0.07 | 3.55 (1.4x) |
| `terminalbench` | `qwen3-coder-480b` | `terminus-2` | 89 | 928 | 24.14% | 56.25% [49.03%, 62.82%] | 13.35 [9.71, 16.79] | 0.22 | 2.68 (1.2x) |
| `terminalbench` | `qwen3-coder-480b` | `openhands` | 89 | 870 | 25.98% | 71.15% [64.54%, 77.24%] | 16.03 [11.10, 21.06] | 4.38 | 2.68 (1.2x) |
| `terminalbench` | `minimax-m2.1` | `claude-code` | 88 | 440 | 37.05% | 34.77% [28.41%, 41.60%] | 2.00 [1.44, 2.62] | 0.08 | 1.84 (1.9x) |
| `terminalbench` | `minimax-m2.1` | `terminus-2` | 88 | 433 | 30.02% | 23.09% [18.01%, 28.34%] | 3.84 [2.57, 5.20] | 0.35 | 1.84 (1.9x) |
| `swe_rebench` | `gpt-5.6-sol` | `swelike` | 111 | 555 | 62.34% | 13.87% [10.27%, 17.48%] | 0.85 [0.61, 1.09] | 0.00 | 1.39 (2.6x) |
| `swe_rebench` | `gpt-5.6-sol` | `codex` | 111 | 555 | 58.02% | 20.54% [16.03%, 25.77%] | 2.24 [1.48, 3.11] | 0.08 | 1.39 (2.6x) |
| `terminalbench` | `claude-opus-4.6` | `claude-code` | 80 | 400 | 60.25% | 20.00% [13.75%, 27.01%] | 1.13 [0.74, 1.63] | 0.17 | 0.61 (1.5x) |
| `terminalbench` | `claude-opus-4.6` | `terminus-2` | 80 | 400 | 64.00% | 13.50% [9.00%, 18.50%] | 1.26 [0.79, 1.85] | 0.00 | 0.61 (1.5x) |
| `terminalbench` | `claude-opus-4.6` | `terminus-3-3` | 80 | 233 | 75.54% | 18.88% [12.85%, 26.86%] | 1.28 [0.76, 2.00] | 0.00 | 0.61 (1.5x) |
| `terminalbench` | `claude-opus-4.6` | `judy` | 80 | 150 | 78.00% | 38.67% [28.47%, 49.32%] | 1.75 [1.32, 2.17] | 0.00 | 0.61 (1.5x) |
| `swe_rebench` | `grok-4.5` | `junie` | 111 | 555 | 61.80% | 15.68% [12.07%, 19.64%] | 0.38 [0.28, 0.47] | 0.00 | 0.28 (1.7x) |
| `swe_rebench` | `grok-4.5` | `swelike` | 111 | 555 | 63.78% | 23.24% [18.92%, 28.11%] | 0.65 [0.51, 0.83] | 0.00 | 0.28 (1.7x) |

