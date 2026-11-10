# Open-Weights vs Closed Models: Loop Behavior, Cost & Performance Report

Generated programmatically from cached public trajectories (`yoonholee/terminalbench-trajectories`, `ibragim-bad/swe_rebench_07_2026_trajectories`, and `sierra-research/tau2-bench`).

## 1. Model Classification

- **Terminal-Bench Open-Weights (16 model entries, 13,899 trials)**: `GLM-4.7@z-ai`, `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@modelscope`, `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@together_ai`, `accounts/fireworks/models/glm-4p6@fireworks_ai`, `accounts/fireworks/models/minimax-m2@fireworks_ai`, `deepseek-v3.2@deepseek`, `glm-4.7@z-ai`, `glm-5@z-ai`, `kimi-k2.5@kimi`, `minimax-m2.1@minimax`, `minimax-m2.1@openai`, `minimax-m2.5@Minimax`, `moonshotai/Kimi-K2-Instruct-0905@together_ai`, `moonshotai/Kimi-K2-Thinking@together_ai`, `openai/gpt-oss-120b@together_ai`, `openai/gpt-oss-20b@together_ai`
- **Terminal-Bench Closed (33 model entries, 38,205 trials)**: Anthropic `claude-*`, Google `gemini-*`, OpenAI proprietary `gpt-5*` (excluding open-weight `openai/gpt-oss-*`), xAI `grok-*`.
- **SWE-rebench Open-Weights (7 `swelike` participants, 3,885 runs)**: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`, `deepseek/deepseek-v4-pro`, `minimax/minimax-m3`, `xiaomi/mimo-v2.5-pro`, `z-ai/glm-5.2`
- **SWE-rebench Closed (10 participants — 6 on `swelike` + 4 agentic systems, 5,550 runs)**: `claude-code (fable-5)`, `codex (gpt-5.6-sol)`, `cursor (composer-2.5)`, `fable-5 (claude-fable-5)`, `gpt-5.6-luna (gpt-5.6-luna)`, `gpt-5.6-sol (gpt-5.6-sol)`, `grok-4.5 (x-ai/grok-4.5)`, `junie (grok-4.5)`, `opus-5 (claude-opus-5)`, `sonnet-5 (claude-sonnet-5)`. *(Note: Cursor's `composer-2.5` is a proprietary commercial checkpoint built on Moonshot's Kimi K2.5 base and is grouped with Closed agentic systems.)*
- **tau2-bench**: All **10,832** simulations use Closed models (`claude-3-7-sonnet`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`); 0 open-weight runs.

## 2. Headline Comparison Across Benchmarks

| Cohort | Runs | Pass Rate | Loop Runs (%) | Loop Step Share | Mean / P90 Steps | Cost-Valid Runs | Avg Cost/Run | Loop $ Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TB All Trials (incl. null steps) — **Open** | 13,899 | 26.05% | n/a | n/a | n/a | n/a | n/a | n/a |
| TB All Trials (incl. null steps) — **Closed** | 38,205 | 44.57% | n/a | n/a | n/a | n/a | n/a | n/a |
| TB With Steps — **Open-Weights** | 11,943 | 24.83% | 4,569 (38.26%) | 9.78% | 35.0 / 81 | 4,211 | $0.70 | 15.61% |
| TB With Steps — **Closed** | 22,519 | 36.03% | 6,115 (27.15%) | 5.85% | 29.2 / 64 | 800 | $0.47 | 34.11% |
| SWE-rebench (`swelike` controlled) — **Open-Weights (7)** | 3,885 | 38.53% | 3,470 (89.32%) | 5.66% | 94.7 / 167 | 3,652 | $0.49 | 4.44% |
| SWE-rebench (`swelike` controlled) — **Closed (6)** | 3,330 | 59.07% | 1,000 (30.03%) | 1.14% | 45.4 / 89 | 2,440 | $1.24 | 1.15% |
| SWE-rebench (`agentic_system` 4 harnesses) — **Closed** | 2,220 | 57.97% | 718 (32.34%) | 3.05% | 50.7 / 92 | 289 | $0.43 | 2.80% |
| SWE-rebench (All 17) — **Closed (10)** | 5,550 | 58.63% | 1,718 (30.95%) | 1.95% | 47.5 / 90 | 2,729 | $1.15 | 1.22% |

> **Methodological note on Terminal-Bench Closed cost share (`34.11%`)**: Only `800 / 22,519` Closed TB runs survived the `[0.50x, 2.00x]` cost-rebuild validation (newer Gemini-3, GPT-5.2, and RSC-pointer runs had `cost_cents=0` or missing tokens), leaving an older subset dominated by `claude-opus-4-1` and `gemini-2.5-flash`. On SWE-rebench (`swelike` controlled), where both cohorts have high cost coverage (`3,652` Open vs `2,440` Closed), Open models spend **4.44%** of dollars on loops vs **1.15%** for Closed.

## 3. Controlled Same-Harness Comparisons (Eliminating Scaffold Confounding)

### 3A. Terminal-Bench Harnesses Where Both Open and Closed Models Ran

| Harness | Cohort | Runs | Pass Rate | Loop Runs (%) | Loop Step Share | Mean / Med Steps | Cost-Valid Runs | Loop $ Share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `terminus-2` | Open-Weights | 8,003 | 27.28% | 2,557 (31.95%) | 8.59% | 33.9 / 20 | 2,968 | 15.54% |
| `terminus-2` | Closed | 7,938 | 37.73% | 1,027 (12.94%) | 3.05% | 17.3 / 10 | 299 | 4.42% |
| `openhands` | Open-Weights | 1,730 | 26.82% | 1,316 (76.07%) | 14.76% | 54.1 / 41 | 417 | 15.39% |
| `openhands` | Closed | 4,325 | 29.62% | 2,199 (50.84%) | 6.27% | 48.5 / 28 | 47 | 87.96% |
| `mini-swe-agent` | Open-Weights | 1,770 | 8.81% | 543 (30.68%) | 18.02% | 9.5 / 5 | 826 | 41.48% |
| `mini-swe-agent` | Closed | 4,725 | 28.80% | 1,573 (33.29%) | 13.12% | 25.7 / 15 | 438 | 26.72% |

### 3B. SWE-rebench `swelike` Controlled Taxonomy Rates (Exact Same Harness, 111 Tasks × 5 Runs Each)

| Pattern | Open-Weights (`swelike`, n=3,885) | Closed (`swelike`, n=3,330) | Closed (`agentic_system`, n=2,220) |
| --- | ---: | ---: | ---: |
| Any pathological loop | 89.32% | 30.03% | 32.34% |
| Exact loop | 74.95% | 10.84% | 16.49% |
| Near-duplicate | 60.88% | 15.02% | 12.70% |
| A-B-A thrash | 19.10% | 0.87% | 3.15% |
| Period-k rotation (k <= 5) | 3.06% | 0.00% | 0.05% |
| Error streak | 13.64% | 13.87% | 11.08% |
| Redundant re-read | 26.10% | 0.03% | 7.12% |
| Productive repeat (non-loop) | 76.63% | 16.31% | 24.28% |

### 3C. Detector Behavior: Rule Detectors vs Chanting Open-Weight Models

| Cohort | Loop Runs | `gemini_cli` Det. Fires | `cline_style` Fires | `openhands_stuck` Fires | `openclaw_10` Fires |
| --- | ---: | ---: | ---: | ---: | ---: |
| TB Open-Weights | 4,569 | 575 | 223 | 202 | 273 |
| TB Closed | 6,115 | 205 | 185 | 145 | 209 |
| SWE Open-Weights (`swelike`) | 3,470 | 402 | 19 | 32 | 6 |
| SWE Closed (`swelike`) | 1,000 | 0 | 0 | 17 | 0 |
| SWE Closed (`agentic_system`) | 718 | 17 | 16 | 36 | 11 |

## 4. Per-Participant Ranking on SWE-rebench (Pass Rate, Loop Rate, Cost)

| Participant | Model | Weights | Scaffold | Pass Rate | Loop Runs (%) | Loop Step Share | Cost/Run | Loop $ Share | Rotations |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fable-5` | `claude-fable-5` | Closed | `swelike` | 64.50% | 25.41% | 0.71% | $1.00 | 0.87% | 0 |
| `grok-4.5` | `x-ai/grok-4.5` | Closed | `swelike` | 63.78% | 23.24% | 0.65% | $2.28 | 0.82% | 0 |
| `opus-5` | `claude-opus-5` | Closed | `swelike` | 63.42% | 37.84% | 1.12% | $2.81 | 1.06% | 0 |
| `glm-5.2` | `z-ai/glm-5.2` | Open | `swelike` | 62.88% | 61.08% | 1.72% | $1.19 | 1.40% | 47 |
| `gpt-5.6-sol` | `gpt-5.6-sol` | Closed | `swelike` | 62.34% | 13.87% | 0.85% | $0.31 | 0.73% | 0 |
| `junie` | `grok-4.5` | Closed | `agentic_system` | 61.80% | 15.68% | 0.38% | n/a | n/a | 0 |
| `claude-code` | `fable-5` | Closed | `agentic_system` | 60.36% | 31.89% | 6.41% | n/a | n/a | 0 |
| `codex` | `gpt-5.6-sol` | Closed | `agentic_system` | 58.02% | 20.54% | 2.24% | $0.64 | 2.27% | 1 |
| `sonnet-5` | `claude-sonnet-5` | Closed | `swelike` | 56.76% | 70.81% | 1.96% | $1.51 | 2.79% | 0 |
| `cursor` | `composer-2.5` | Closed | `agentic_system` | 51.71% | 61.26% | 2.75% | $0.25 | 3.90% | 0 |
| `minimax-m3` | `minimax/minimax-m3` | Open | `swelike` | 47.21% | 99.28% | 5.95% | $0.54 | 7.60% | 5 |
| `mimo-v2.5-pro` | `xiaomi/mimo-v2.5-pro` | Open | `swelike` | 46.49% | 90.09% | 4.52% | $0.31 | 5.02% | 8 |
| `gpt-5.6-luna` | `gpt-5.6-luna` | Closed | `swelike` | 43.60% | 9.01% | 0.57% | $0.05 | 0.57% | 0 |
| `deepseek-v4-pro` | `deepseek/deepseek-v4-pro` | Open | `swelike` | 40.18% | 85.41% | 4.49% | $0.40 | 4.30% | 41 |
| `qwen3.6-27b` | `Qwen/Qwen3.6-27B` | Open | `swelike` | 31.17% | 95.68% | 6.84% | $0.24 | 4.88% | 3 |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | Open | `swelike` | 24.68% | 96.58% | 8.46% | $0.23 | 7.72% | 9 |
| `qwen3.5-35b-a3b` | `Qwen/Qwen3.5-35B-A3B` | Open | `swelike` | 17.12% | 97.12% | 7.98% | $0.51 | 6.24% | 6 |

## 5. Terminal-Bench Frontier Standings (Top 10 Open vs Top 10 Closed Across All 52,104 Trials)

### Top 10 Open Models (Terminal-Bench 2.0 by All-Trial Pass Rate)
| Model | Trials | All-Trial Pass Rate | Step Runs | Loop Runs (%) | Loop Step Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `glm-5@z-ai` | 442 | 52.26% | 376 | 23.94% | 6.00% |
| `kimi-k2.5@kimi` | 435 | 43.45% | 0 | n/a | n/a |
| `minimax-m2.5@Minimax` | 445 | 42.25% | 411 | 33.82% | 4.87% |
| `deepseek-v3.2@deepseek` | 445 | 39.55% | 373 | 8.58% | 0.82% |
| `minimax-m2.1@minimax` | 445 | 36.63% | 440 | 34.77% | 2.00% |
| `moonshotai/Kimi-K2-Thinking@together_ai` | 892 | 35.87% | 892 | 31.28% | 4.64% |
| `glm-4.7@z-ai` | 427 | 34.43% | 0 | n/a | n/a |
| `GLM-4.7@z-ai` | 445 | 33.26% | 0 | n/a | n/a |
| `accounts/fireworks/models/minimax-m2@fireworks_ai` | 902 | 30.60% | 902 | 40.13% | 10.09% |
| `minimax-m2.1@openai` | 438 | 29.68% | 438 | 23.52% | 3.82% |

### Top 10 Closed Models (Terminal-Bench 2.0 by All-Trial Pass Rate)
| Model | Trials | All-Trial Pass Rate | Step Runs | Loop Runs (%) | Loop Step Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemini-3.1-pro-preview@Google` | 890 | 76.63% | 431 | 7.19% | 0.64% |
| `claude-opus-4.6@anthropic` | 445 | 71.91% | 157 | 38.85% | 1.66% |
| `gpt-5.3-codex@openai` | 1,778 | 71.37% | 0 | n/a | n/a |
| `gpt-5.3-codex@OpenAI` | 445 | 70.34% | 0 | n/a | n/a |
| `gpt-5.2-codex@openai` | 433 | 67.67% | 417 | 46.04% | 8.55% |
| `claude-opus-4-6@anthropic` | 2,620 | 66.37% | 1,127 | 18.46% | 1.49% |
| `gemini-3-pro-preview@Google` | 1,334 | 64.09% | 0 | n/a | n/a |
| `gpt-5.2@openai` | 1,422 | 60.69% | 443 | 0.45% | 0.05% |
| `gpt-5.1-codex-max@openai` | 445 | 60.45% | 0 | n/a | n/a |
| `gemini-3-pro-preview@gemini` | 890 | 59.33% | 445 | 7.19% | 0.91% |

## 6. Key Takeaways

- **Same-scaffold loop gap is massive (`swelike` controlled)**: Holding the harness constant on SWE-rebench, Open-Weights models loop **3.0×** as often by run (**89.32%** vs **30.03%**), spend **5.0×** the step share on loops (**5.66%** vs **1.14%**), and run nearly **2×** longer (mean **94.7** vs **45.4** steps).
- **Repeatedly confirmed on Terminal-Bench `terminus-2`**: On the largest shared TB harness, Open-Weights loop at **31.95%** vs **12.94%** for Closed (**8.59%** vs **3.05%** loop steps; pass rate **27.28%** vs **37.73%**).
- **Frontier pass parity, residual loopiness**: Top open-weight SWE participant `glm-5.2` passes **62.88%** at **$1.19/run** — essentially matching `opus-5` (**63.42%**, **$2.81/run**) and `grok-4.5` (**63.78%**, **$2.28/run**), yet still loops in **61.08%** of runs (including **47** `period_k_rotation` runs vs **0** for both `opus-5` and `grok-4.5`).
- **Intra-turn chanting & redundant re-reads skew Open**: On `swelike`, Open-Weights hit **402** `gemini_cli` triggers (mostly intra-turn 50-char chanting in `minimax-m3`/`qwen3.6`) vs **0** for Closed `swelike`, and exhibit **26.10%** redundant re-reads vs **0.03%** for Closed `swelike`.
