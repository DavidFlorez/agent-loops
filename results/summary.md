# Agent Loops Study — Summary of Computed Findings

All numbers below are generated programmatically from cached public trajectory datasets (CPU-only offline pipeline, fixed seed `42`, 200 matched-random bootstrap seeds).

## 1. Headline Prevalence, Cost Share, and Pass Rates

| Metric | Terminal-Bench 2.0 | TB Clean-Trace Subset (0 RSC refs) | SWE-rebench July 2026 |
| --- | ---: | ---: | ---: |
| Trajectories analyzed (with steps) | 34,462 | 5,371 | 9,435 |
| Runs with any loop | 10,684 (31.00%) | 2,253 (41.95%) | 5,188 (54.99%) |
| Loop share of steps | 7.38% (79,344/1,075,547) | 12.31% | 4.11% (25,975/631,803) |
| Cost-validated runs | 5,011 | 3,787 | 6,381 |
| Loop share of cache-aware dollars | 17.71% ($589.76 / $3,330.95) | 14.94% | 2.38% ($117.38 / $4,927.87) |
| Pass rate: looping runs | 27.42% | 19.22% | 43.14% |
| Pass rate: loop-free runs | 34.27% | 24.47% | 59.17% |
| Productive repeat step share (excluded from loops) | 5.54% | 9.82% | 1.90% |
| Period-k rotation runs missed by Cline-5 | 277/334 (82.93%) | 67/79 (84.81%) | 118/120 (98.33%) |

## 2. Loop Taxonomy Breakdown (Runs Exhibiting Each Pattern)

| Pattern | Terminal-Bench 2.0 Runs | SWE-rebench Runs |
| --- | ---: | ---: |
| Exact loop (same call + same obs) | 8,225 (23.87%) | 3,639 (38.57%) |
| Near-duplicate | 3,421 (9.93%) | 3,147 (33.35%) |
| A-B-A thrash | 2,578 (7.48%) | 841 (8.91%) |
| Period-k rotation (k <= 5) | 334 (0.97%) | 120 (1.27%) |
| Error streak | 2,645 (7.68%) | 1,238 (13.12%) |
| Redundant re-read (no write in between) | 582 (1.69%) | 1,173 (12.43%) |
| Productive repeat (same call, new obs — non-loop) | 15,292 (44.37%) | 4,059 (43.02%) |

## 3. Cost Rebuild Drop Audit (Pre-Registered Margin `[0.50x, 2.00x]`)

### Terminal-Bench 2.0
- `cost_dropped_empty_steps`: 65
- `cost_dropped_missing_reported_cost_and_tokens`: 4,408
- `cost_dropped_rebuild_mismatch`: 24,543
- `cost_dropped_subagents`: 435
- `cost_included`: 5,011
- `dropped_null_or_empty_steps`: 17,642
- `rows_with_steps`: 34,462
- `total_rows`: 52,104

### SWE-rebench July 2026
- `cost_dropped_missing_reported_cost_and_tokens`: 6
- `cost_dropped_rebuild_mismatch`: 2,494
- `cost_dropped_subagents`: 554
- `cost_included`: 6,381
- `total_rows`: 9,435

## 4. Detector Replay vs Matched-Random Baseline (200 Seeds)

### Terminal-Bench 2.0
| Detector | Firing Rate | Dollars Saved | False-Stop Rate | Matched-Rand FSR (95% CI) | Matched-Rand Saved (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cline_style` | 1.18% (408) | 5.00% ($166.68) | 18.38% | 23.05% [19.60%, 26.72%] | 3.72% [2.42%, 5.23%] |
| `openhands_stuck` | 1.01% (347) | 1.82% ($60.59) | 11.53% | 19.35% [16.14%, 23.06%] | 3.08% [1.63%, 5.74%] |
| `gemini_cli` | 2.26% (780) | 5.71% ($190.06) | 10.77% | 23.18% [20.51%, 25.90%] | 4.33% [2.99%, 6.43%] |
| `openclaw_10` | 1.40% (482) | 5.05% ($168.28) | 13.49% | 14.03% [10.58%, 17.23%] | 4.77% [2.72%, 7.49%] |
| `openclaw_20` | 0.25% (86) | 0.98% ($32.68) | 3.49% | 10.70% [4.65%, 16.28%] | 1.18% [0.40%, 2.65%] |
| `step_cap_30` | 30.70% (10580) | 73.00% ($2,431.75) | 23.49% | 23.95% [23.82%, 24.08%] | 70.41% [68.71%, 71.60%] |
| `step_cap_50` | 16.64% (5734) | 53.92% ($1,796.15) | 17.51% | 17.83% [17.67%, 17.98%] | 52.53% [50.91%, 53.37%] |
| `step_cap_75` | 8.89% (3065) | 35.98% ($1,198.52) | 13.83% | 14.08% [13.87%, 14.26%] | 34.80% [33.32%, 35.61%] |
| `step_cap_100` | 5.45% (1878) | 24.07% ($801.76) | 10.70% | 10.81% [10.65%, 10.97%] | 23.69% [22.73%, 24.04%] |
| `step_cap_150` | 2.50% (862) | 12.04% ($401.12) | 7.66% | 7.81% [7.66%, 7.89%] | 11.93% [11.30%, 12.04%] |
| `turn_budget_1_25x_p90` | 14.52% (5004) | 55.36% ($1,844.12) | 13.17% | 13.31% [13.13%, 13.47%] | 53.83% [52.50%, 54.69%] |

### SWE-rebench July 2026
| Detector | Firing Rate | Dollars Saved | False-Stop Rate | Matched-Rand FSR (95% CI) | Matched-Rand Saved (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cline_style` | 0.37% (35) | 0.29% ($14.38) | 37.14% | 46.19% [34.21%, 60.00%] | 0.18% [0.10%, 0.31%] |
| `openhands_stuck` | 0.90% (85) | 0.92% ($45.52) | 41.18% | 42.68% [34.09%, 51.76%] | 0.54% [0.29%, 0.84%] |
| `gemini_cli` | 4.44% (419) | 2.05% ($100.87) | 22.20% | 28.60% [25.04%, 31.99%] | 1.62% [1.42%, 1.82%] |
| `openclaw_10` | 0.18% (17) | 0.04% ($2.14) | 29.41% | 41.06% [23.53%, 64.71%] | 0.03% [0.01%, 0.08%] |
| `openclaw_20` | 0.07% (7) | 0.00% ($0.00) | 14.29% | 50.57% [14.29%, 85.71%] | 0.00% [0.00%, 0.00%] |
| `step_cap_30` | 75.41% (7115) | 63.46% ($3,127.08) | 47.65% | 47.76% [47.60%, 47.90%] | 62.39% [61.79%, 62.82%] |
| `step_cap_50` | 56.01% (5285) | 41.70% ($2,054.76) | 43.88% | 44.08% [43.92%, 44.24%] | 40.88% [40.38%, 41.22%] |
| `step_cap_75` | 31.90% (3010) | 23.17% ($1,141.56) | 40.13% | 40.21% [39.97%, 40.40%] | 22.79% [22.42%, 22.97%] |
| `step_cap_100` | 17.23% (1626) | 12.95% ($638.10) | 37.76% | 38.06% [37.70%, 38.44%] | 12.57% [12.19%, 12.76%] |
| `step_cap_150` | 6.10% (576) | 4.81% ($236.99) | 38.54% | 38.65% [38.19%, 39.24%] | 4.69% [4.59%, 4.75%] |
| `turn_budget_1_25x_p90` | 4.98% (470) | 4.06% ($200.09) | 30.00% | 29.94% [29.15%, 30.64%] | 3.91% [3.56%, 4.02%] |

## 5. Per-Harness Summary

### Terminal-Bench 2.0
| Harness | Runs | Loop Runs (%) | Loop Step Share | Loop Dollar Share | Pass (Loop vs Clean) | 1.25x P90 Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openhands` | 6,055 | 58.05% | 8.89% | 22.59% | 29.10% vs 28.43% | 78 |
| `deepagent-harbor` | 417 | 46.04% | 8.55% | n/a | 79.17% vs 58.67% | 86 |
| `judy` | 157 | 38.85% | 1.66% | n/a | 81.97% vs 72.92% | 64 |
| `mini-swe-agent` | 6,495 | 32.58% | 13.72% | 27.76% | 19.94% vs 25.01% | 45 |
| `claude-code` | 2,522 | 29.94% | 2.21% | 0.00% | 40.79% vs 44.77% | 120 |
| `gemini-cli` | 787 | 22.49% | 6.66% | n/a | 15.82% vs 20.49% | 38 |
| `terminus-2` | 15,941 | 22.48% | 6.73% | 15.03% | 23.07% vs 35.21% | 40 |
| `codex` | 1,410 | 14.54% | 2.09% | n/a | 33.66% vs 33.86% | 47 |
| `terminus-3-3` | 678 | 11.65% | 0.87% | 0.88% | 64.56% vs 75.96% | 54 |

### SWE-rebench July 2026
| Participant | Runs | Loop Runs (%) | Loop Step Share | Loop Dollar Share | Pass (Loop vs Clean) | 1.25x P90 Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `minimax-m3` | 555 | 99.28% | 5.95% | 7.60% | 47.19% vs 50.00% | 364 |
| `qwen3.5-35b-a3b` | 555 | 97.12% | 7.98% | 6.24% | 16.51% vs 37.50% | 164 |
| `qwen3.6-35b-a3b` | 555 | 96.58% | 8.46% | 7.72% | 24.63% vs 26.32% | 133 |
| `qwen3.6-27b` | 555 | 95.68% | 6.84% | 4.88% | 31.45% vs 25.00% | 127 |
| `mimo-v2.5-pro` | 555 | 90.09% | 4.52% | 5.02% | 46.60% vs 45.45% | 135 |
| `deepseek-v4-pro` | 555 | 85.41% | 4.49% | 4.30% | 40.72% vs 37.04% | 125 |
| `sonnet-5` | 555 | 70.81% | 1.96% | 2.79% | 55.98% vs 58.64% | 133 |
| `cursor` | 555 | 61.26% | 2.75% | 3.90% | 46.76% vs 59.53% | 113 |
| `glm-5.2` | 555 | 61.08% | 1.72% | 1.40% | 65.78% vs 58.33% | 198 |
| `opus-5` | 555 | 37.84% | 1.12% | 1.06% | 62.38% vs 64.06% | 126 |
| `claude-code` | 555 | 31.89% | 6.41% | n/a | 46.89% vs 66.67% | 120 |
| `fable-5` | 555 | 25.41% | 0.71% | 0.87% | 57.45% vs 66.91% | 96 |
| `grok-4.5` | 555 | 23.24% | 0.65% | 0.82% | 62.79% vs 64.08% | 89 |
| `codex` | 555 | 20.54% | 2.24% | 2.27% | 60.53% vs 57.37% | 39 |
| `junie` | 555 | 15.68% | 0.38% | n/a | 59.77% vs 62.18% | 109 |
| `gpt-5.6-sol` | 555 | 13.87% | 0.85% | 0.73% | 63.64% vs 62.13% | 33 |
| `gpt-5.6-luna` | 555 | 9.01% | 0.57% | 0.57% | 32.00% vs 44.75% | 25 |

## 6. Top 10 Loop-Prone Tasks for Gemini-Family Models (full 20 in `results/loop_prone_tasks.csv`)

| Rank | Task | Gemini Runs | Loop Rate | Loop Step Share | Gemini Pass Rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `winning-avg-corewars` | 55 | 60.00% | 11.29% | 20.00% |
| 2 | `crack-7z-hash` | 54 | 59.26% | 36.58% | 27.78% |
| 3 | `db-wal-recovery` | 54 | 59.26% | 17.82% | 3.70% |
| 4 | `largest-eigenval` | 55 | 58.18% | 4.20% | 30.91% |
| 5 | `break-filter-js-from-html` | 54 | 55.56% | 11.64% | 20.37% |
| 6 | `mailman` | 51 | 54.90% | 8.00% | 31.37% |
| 7 | `password-recovery` | 53 | 54.72% | 16.29% | 26.42% |
| 8 | `tune-mjcf` | 53 | 54.72% | 9.94% | 47.17% |
| 9 | `make-mips-interpreter` | 49 | 53.06% | 7.60% | 4.08% |
| 10 | `schemelike-metacircular-eval` | 53 | 52.83% | 32.24% | 16.98% |

## 7. τ²-bench Optional Sanity Check

- Simulations analyzed: **10,832** across **26** result files (99,236 tool calls).
- Simulations with any pathological loop: **2,769 (25.56%)**; loop step share: **10.96%**.
- Breakdown: exact_loop=2689, near_duplicate=19, aba_thrash=679, period_k_rotation=175, error_streak=160, redundant_reread=0; productive_repeat=2387.
- Pass rate: looping 48.36% vs loop-free 72.45%.
