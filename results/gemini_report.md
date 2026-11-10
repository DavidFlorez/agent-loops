# Gemini Models Performance & Loop Report

Generated programmatically from cached public trajectories (`yoonholee/terminalbench-trajectories`, `ibragim-bad/swe_rebench_07_2026_trajectories`, and `sierra-research/tau2-bench`).

## 1. Coverage Across the Three Datasets / Frameworks

| Dataset / Benchmark | Total Runs in Corpus | Gemini Runs | Gemini Runs with Steps | Notes |
| --- | ---: | ---: | ---: | --- |
| Terminal-Bench 2.0 (`yoonholee/terminalbench-trajectories`) | 52,104 | 9,320 | 4,726 | 13 harnesses total; 5 harnesses include step traces |
| SWE-rebench July 2026 (`ibragim-bad/swe_rebench_07_2026_trajectories`) | 9,435 | 0 | 0 | No Google/Gemini participants in the July 2026 release |
| tau2-bench (`sierra-research/tau2-bench`) | 10,832 | 0 | 0 | Bundled final runs cover Claude 3.7 Sonnet, GPT-4.1, GPT-4.1-mini, and o4-mini only |

## 2. Gemini Model Generation Summary (Terminal-Bench 2.0)

| Generation | All Trials | Overall Pass Rate | Step-Logged Runs | Loop Runs (%) | Loop Step Share | Pass (Loop vs Clean) | Mean / P90 Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **gemini-2.5** | 3,536 | 20.14% | 3,406 | 1,198 (35.17%) | 14.58% | 14.02% vs 24.59% | 28.2 / 61 |
| **gemini-3.0** | 4,894 | 58.40% | 889 | 86 (9.67%) | 3.14% | 30.23% vs 56.91% | 16.5 / 34 |
| **gemini-3.1** | 890 | 76.63% | 431 | 31 (7.19%) | 0.64% | 48.39% vs 76.25% | 26.5 / 54 |
| **All Gemini (combined)** | 9,320 | 45.62% | 4,726 | 1,315 (27.82%) | 11.91% | 15.89% vs 38.26% | 25.9 / 55 |

- **Step-length split across all 4,726 step-logged Gemini runs**: passing runs finish at **P50=11, P90=31** steps vs **P50=16, P90=66** steps for failing runs.
- **Zero-RSC clean-trace subset (`122` Gemini runs)**: loop prevalence **9.02%**, step share **5.49%**, validated cost share (`n=0` runs, $0.00 total) **n/a**.

## 3. Gemini Performance by Agent Framework / Harness (Terminal-Bench 2.0)

### 3A. Five Frameworks with Step Traces (4,726 Runs — Loop, Step, Cost, and Pass Analysis)

| Harness | Step-Logged Models | Step Runs | Step-Run Pass | All-Trial Pass | Loop Runs (%) | Loop Step Share | Cost-Valid Runs | Loop $ Share | `gemini_cli` Fires | `openclaw_10` Fires |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `openhands` | gemini-2.5-flash, gemini-2.5-pro | 872 | 16.74% | 16.40% (890) | 511 (58.60%) | 13.04% | 3 | 76.50% | 8 | 50 |
| `mini-swe-agent` | gemini-2.5-flash, gemini-2.5-pro | 867 | 22.15% | 21.60% (889) | 393 (45.33%) | 25.66% | 32 | 91.25% | 14 | 30 |
| `gemini-cli` | gemini-2.5-flash, gemini-2.5-pro | 787 | 19.44% | 33.52% (1,766) | 177 (22.49%) | 6.66% | 0 | n/a | 3 | 16 |
| `terminus-2` | gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview | 1,769 | 39.74% | 39.70% (1,771) | 203 (11.48%) | 3.44% | 16 | 16.11% | 13 | 8 |
| `terminus-3-3` | gemini-3.1-pro-preview | 431 | 74.25% | 74.83% (445) | 31 (7.19%) | 0.64% | 0 | n/a | 2 | 1 |

### 3B. Full Harness × Model Matrix (All 9,320 Gemini Trials)

| Harness | Model | Trials | All-Trial Pass Rate | Mean Wall Time (s) | Step Runs | Loop Runs (%) | Loop Step Share | Loop $ Share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `terminus-2` | `gemini-2.5-pro@gemini` | 442 | 32.81% | 697.8 | 442 | 51 (11.54%) | 2.49% | n/a |
| `mini-swe-agent` | `gemini-2.5-pro@gemini` | 444 | 26.13% | 650.7 | 435 | 180 (41.38%) | 17.81% | n/a |
| `gemini-cli` | `gemini-2.5-pro@gemini` | 437 | 19.68% | 865.6 | 381 | 109 (28.61%) | 5.98% | n/a |
| `terminus-2` | `gemini-2.5-flash@gemini` | 439 | 17.08% | 719.4 | 438 | 66 (15.07%) | 4.61% | 16.11% |
| `mini-swe-agent` | `gemini-2.5-flash@gemini` | 445 | 17.08% | 676.6 | 432 | 213 (49.31%) | 30.64% | 91.25% |
| `openhands` | `gemini-2.5-flash@gemini` | 445 | 16.40% | 718.5 | 436 | 261 (59.86%) | 9.98% | 2.92% |
| `openhands` | `gemini-2.5-pro@gemini` | 445 | 16.40% | 552.4 | 436 | 250 (57.34%) | 17.19% | 91.69% |
| `gemini-cli` | `gemini-2.5-flash@gemini` | 439 | 15.49% | 576.0 | 406 | 68 (16.75%) | 7.51% | n/a |
| `sage` | `gemini-3-pro-preview@Google` | 445 | 65.17% | 481.6 | 0 | n/a (no steps) | n/a | n/a |
| `ante` | `gemini-3-pro-preview@Google` | 444 | 64.86% | 504.5 | 0 | n/a (no steps) | n/a | n/a |
| `Junie` | `gemini-3-flash-preview@gemini` | 445 | 64.27% | 461.0 | 0 | n/a (no steps) | n/a | n/a |
| `ruley` | `gemini-3-pro-preview@Google` | 445 | 62.25% | 571.3 | 0 | n/a (no steps) | n/a | n/a |
| `ii-agent-simple` | `gemini-3-pro-preview@gemini` | 445 | 61.80% | 635.9 | 0 | n/a (no steps) | n/a | n/a |
| `Factory Droid` | `gemini-3-pro-preview@google` | 445 | 61.12% | 418.6 | 0 | n/a (no steps) | n/a | n/a |
| `terminus-2` | `gemini-3-pro-preview@gemini` | 445 | 56.85% | 548.6 | 445 | 32 (7.19%) | 0.91% | n/a |
| `letta-code` | `gemini-3-pro-preview@google` | 445 | 55.96% | 568.6 | 0 | n/a (no steps) | n/a | n/a |
| `terminus-2` | `gemini-3-flash-preview@gemini` | 445 | 51.69% | 547.2 | 444 | 54 (12.16%) | 4.64% | n/a |
| `gemini-cli` | `gemini-3-flash-preview@google` | 445 | 51.01% | 623.8 | 0 | n/a (no steps) | n/a | n/a |
| `gemini-cli` | `gemini-3-flash-preview@Google` | 445 | 47.42% | 835.9 | 0 | n/a (no steps) | n/a | n/a |
| `forge` | `gemini-3.1-pro-preview@Google` | 445 | 78.43% | 561.0 | 0 | n/a (no steps) | n/a | n/a |
| `terminus-3-3` | `gemini-3.1-pro-preview@Google` | 445 | 74.83% | 626.2 | 431 | 31 (7.19%) | 0.64% | n/a |

## 4. Key Framework & Detector Takeaways for Gemini

- **Controlled model upgrade (`terminus-2` harness held constant)**: Moving `gemini-2.5` (`n=880`) → `gemini-3.0` (`n=889`) lowers loop prevalence from **13.30%** to **9.67%** while more than doubling pass rate from **25.00%** to **54.33%**.
- **Harness sensitivity (`gemini-2.5` held constant)**: Even with the exact same models, loop rate varies **4.4×** across frameworks (`openhands` **58.60%**, `mini-swe-agent` **45.33%** / **25.66%** loop step share, `gemini-cli` **22.49%**, `terminus-2` **13.30%**).
- **Detector gap**: Across all 1,315 looping Gemini trajectories, `gemini_cli`'s 5-consecutive / intra-turn rule fires on only **40 (3.04%)** of loopers, whereas rolling-window `openclaw_10` catches **105** (2.62× as many). See [`results/loop_prone_tasks.csv`](loop_prone_tasks.csv) for the top 20 Gemini loop-prone tasks.
