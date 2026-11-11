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
| `tb_mini_swe_agent` | `gemini` | Closed | 2 | 867 | 22.13% | 24.22 | 45.34% | 91.25% |
| `tb_mini_swe_agent` | `gpt_oss` | Open | 2 | 1,770 | 8.82% | 19.99 | 30.68% | 47.52% |
| `tb_mini_swe_agent` | `grok` | Closed | 2 | 846 | 26.97% | 11.12 | 46.71% | 7.14% |
| `tb_mini_swe_agent` | `claude` | Closed | 3 | 1,276 | 37.36% | 8.76 | 35.48% | 20.47% |
| `tb_mini_swe_agent` | `gpt` | Closed | 4 | 1,733 | 27.00% | 8.51 | 19.25% | n/a |
| `tb_openhands` | `qwen (open)` | Open | 1 | 870 | 25.98% | 16.03 | 71.15% | 15.45% |
| `tb_openhands` | `gemini` | Closed | 2 | 872 | 16.74% | 13.59 | 58.60% | 47.31% |
| `tb_openhands` | `kimi` | Open | 1 | 860 | 27.67% | 13.50 | 81.05% | 15.28% |
| `tb_openhands` | `grok` | Closed | 1 | 421 | 28.74% | 8.82 | 41.57% | 9.16% |
| `tb_openhands` | `claude` | Closed | 4 | 1,734 | 36.94% | 6.09 | 54.03% | 93.27% |
| `tb_openhands` | `gpt` | Closed | 3 | 1,291 | 28.48% | 5.40 | 44.25% | 42.66% |
| `tb_terminus_2` | `qwen (open)` | Open | 1 | 888 | 25.23% | 13.33 | 55.07% | 20.24% |
| `tb_terminus_2` | `glm` | Open | 2 | 1,226 | 41.42% | 10.68 | 30.92% | 29.06% |
| `tb_terminus_2` | `kimi` | Open | 2 | 1,735 | 33.39% | 7.03 | 38.06% | 11.42% |
| `tb_terminus_2` | `minimax` | Open | 3 | 1,667 | 36.41% | 6.10 | 31.75% | 17.11% |
| `tb_terminus_2` | `gpt_oss` | Open | 2 | 1,774 | 11.64% | 4.08 | 11.17% | 7.51% |
| `tb_terminus_2` | `gpt` | Closed | 7 | 2,927 | 37.27% | 3.21 | 8.64% | 0.00% |
| `tb_terminus_2` | `gemini` | Closed | 4 | 1,691 | 40.67% | 3.16 | 11.25% | 16.11% |
| `tb_terminus_2` | `claude` | Closed | 5 | 2,110 | 47.23% | 2.20 | 22.79% | 6.35% |
| `tb_terminus_2` | `grok` | Closed | 2 | 835 | 19.59% | 1.17 | 4.89% | 0.90% |
| `tb_terminus_2` | `deepseek` | Open | 1 | 364 | 41.76% | 0.82 | 8.24% | 2.49% |

### 4C. Google Gemini Detailed Profile Across Uniform Scaffolds & Sierra Variants

| Cohort / Scaffold | Model | Pass Rate | Loops/100 [95% CI] | Any-Loop Rate [95% CI] | Loop $ Share |
| --- | --- | ---: | ---: | ---: | ---: |
| `tau2_banking_knowledge_alltools` | `gemini-2-5-pro` | 12.76% | 2.91 [1.58, 5.03] | 27.08% [21.99%, 32.81%] | 2.79% |
| `tau2_banking_knowledge_alltools` | `gemini-3-1-pro-preview` | 22.54% | 7.80 [6.05, 9.63] | 48.45% [41.86%, 55.06%] | 7.91% |
| `tau2_classic` | `gemini-3-flash` | 80.94% | 1.92 [1.59, 2.27] | 17.90% [14.84%, 21.22%] | 2.17% |
| `tau2_classic` | `gemini-3-pro` | 84.82% | 0.15 [0.05, 0.33] | 1.47% [0.64%, 2.66%] | 0.23% |
| `tb_mini_swe_agent` | `gemini-2.5-flash@gemini` | 17.59% | 30.64 [20.74, 39.09] | 49.31% [42.05%, 55.99%] | 91.25% |
| `tb_mini_swe_agent` | `gemini-2.5-pro@gemini` | 26.67% | 17.81 [10.54, 26.12] | 41.38% [35.25%, 48.01%] | n/a |
| `tb_openhands` | `gemini-2.5-flash@gemini` | 16.74% | 9.98 [8.25, 12.00] | 59.86% [52.41%, 68.04%] | 2.92% |
| `tb_openhands` | `gemini-2.5-pro@gemini` | 16.74% | 17.19 [11.59, 23.46] | 57.34% [49.89%, 64.68%] | 91.69% |
| `tb_terminus_2` | `gemini-2.5-flash@gemini` | 17.86% | 4.68 [1.75, 8.65] | 14.76% [10.26%, 19.76%] | 16.11% |
| `tb_terminus_2` | `gemini-2.5-pro@gemini` | 33.41% | 2.49 [1.34, 4.18] | 11.37% [7.55%, 15.40%] | n/a |
| `tb_terminus_2` | `gemini-3-flash-preview@gemini` | 52.59% | 4.53 [1.03, 9.94] | 12.03% [8.04%, 17.02%] | n/a |
| `tb_terminus_2` | `gemini-3-pro-preview@gemini` | 58.82% | 0.94 [0.42, 1.57] | 6.82% [3.76%, 10.35%] | n/a |
| `tau2_bk_terminal_use` (`terminal_use`, excluded from AllTools) | `gemini-3-flash` | 20.62% | 10.03 | 76.55% | n/a |
| `tau2_bk_terminal_use` (`terminal_use`, excluded from AllTools) | `gemini-3-pro` | 15.72% | 16.16 | 57.22% | n/a |

## 5. Open vs Closed Dollar Share & Longtail Context Confounders

| Cohort | Open Mean Loop $ Share (Cost-Valid) | Closed Mean Loop $ Share (Cost-Valid) | Cost Provenance Note |
| --- | ---: | ---: | --- |
| `tb_terminus_2` | 13.34% | 5.52% | `reported_usd_rebuilt` |
| `tb_openhands` | 15.36% | 47.94% | `reported_usd_rebuilt` |
| `tb_mini_swe_agent` | 47.52% | 27.83% | `reported_usd_rebuilt` |
| `swe_rebench_swelike` | 5.20% | 1.14% | `tokens_rebuilt_list_rates` |
| `longtail_swe_swelike` | 2.88% | 0.92% | `tokens_rebuilt_list_rates` |
| `tau2_classic` | 0.61% | 0.75% | `rebuilt_from_turn_tokens_uncached, reported_provider_usd` |
| `tau2_banking_knowledge_alltools` | 3.65% | 2.31% | `rebuilt_from_turn_tokens_uncached, reported_provider_usd` |

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

