# Terminal-Bench 2.0 Cost Rebuild & Null-Step Coverage Audit

## 1. Accounting for All 52,104 Terminal-Bench Rows

- **Total rows in `yoonholee/terminalbench-trajectories`**: `52,104`
- **Dropped null/empty `steps` (`steps == 'null'`)**: `17,642` (across 16 models with 0 step traces and 17 harnesses without step traces)
- **Rows with step traces analyzed**: `34,462`
  - `cost_dropped_empty_steps` (0 tool/message chars): `65`
  - `cost_dropped_subagents` (`Task` / subagent calls): `435`
  - `cost_dropped_missing_reported_cost_and_tokens` (`cost_cents == 0` or `null`): `4,408`
  - `cost_dropped_rebuild_mismatch` (`raw_estimated / reported` outside `[0.50x, 2.00x]`): `24,543`
  - **`cost_included` (validated within `[0.50x, 2.00x]`)**: **`5,011`** (`4,211` Open-Weights vs **`800` Closed**)

## 2. Why 24,543 Runs Failed the `[0.50x, 2.00x]` Character-to-Dollar Rebuild Check

We checked for unit bugs (`cost_cents / 100.0` vs dollars) and `cache_tokens` double-counting (`input_tokens - cache_tokens` vs `input_tokens + cache_tokens`):

1. **No `cost_cents` unit bug or `cache_tokens` double-count bug**: When pricing trajectory-level `(input_tokens, cache_tokens, output_tokens)` directly against `cost_cents / 100.0` across the `29,554` runs with both token counts and `cost_cents > 0`, the median ratio under `uncached = input_tokens - cache_tokens` is **`1.249x`** (`P25=1.000x`, `P75=2.781x`), confirming `cost_cents` is in cents (`$ = cost_cents / 100`) and `input_tokens` in TB includes `cache_tokens`.
2. **Root Cause #1 — Next.js RSC `$<hex>` pointer truncation in `steps` (`24,300` of `29,554` priced runs)**: In the scraped `tbench.ai` Parquet export, `23,076` of the `24,543` mismatch runs (`94.0%`) contain unresolved React Server Component pointers (`$3b`, `$4a`) replacing large tool observations and prompts with 3-byte strings. Because `rebuild_trajectory_costs` reconstructs per-step cumulative tokens from the visible step text (`chars / 4`), replacing multi-kilobyte terminal outputs with `'$3b'` collapses the character-rebuilt cost to a median of **`0.120x`** of `cost_cents / 100.0`.
3. **Root Cause #2 — Unlogged internal reasoning tokens & terminal state on clean traces**: On the `5,254` priced runs without RSC pointers, **`3,787` (`72.1%`) pass** the `[0.50x, 2.00x]` check (median ratio **`0.620x`**), while `1,467` fall outside `[0.50x, 2.00x]` (primarily reasoning models whose hidden thinking tokens are billed in `output_tokens` but omitted from `steps[].msg`, or multi-turn tmux buffers truncated at 5,000 chars).

### Histogram of `raw_estimated_usd / reported_usd` (`n = 29,554` runs with `cost_cents > 0`)

| Ratio Bucket (`raw_estimated / reported`) | Total Runs | Share (%) | With RSC `$<hex>` Truncation | Zero-RSC Clean Traces | Open-Weights | Closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<0.05x (severe truncation)` | 5,597 | 18.94% | 5,471 | 126 | 338 | 5,259 |
| `0.05x-0.10x` | 4,991 | 16.89% | 4,894 | 97 | 703 | 4,288 |
| `0.10x-0.25x` | 8,684 | 29.38% | 8,481 | 203 | 2,290 | 6,394 |
| `0.25x-0.50x (moderate under-rebuild)` | 5,219 | 17.66% | 4,186 | 1,033 | 2,695 | 2,524 |
| `0.50x-1.00x (KEPT in [0.5, 2.0])` | 4,691 | 15.87% | 1,153 | 3,538 | 3,956 | 735 |
| `1.00x-2.00x (KEPT in [0.5, 2.0])` | 320 | 1.08% | 71 | 249 | 255 | 65 |
| `2.00x-5.00x (over-rebuild)` | 39 | 0.13% | 31 | 8 | 9 | 30 |
| `>5.00x` | 13 | 0.04% | 13 | 0 | 0 | 13 |

- **Quantiles (all 29,554 priced runs)**: `p01=0.005x`, `p05=0.015x`, `p10=0.028x`, `p25=0.067x`, `p50=0.153x`, `p75=0.360x`, `p90=0.644x`, `p95=0.773x`, `p99=1.041x`
- **Quantiles (zero-RSC clean traces, n=5,254)**: `p01=0.020x`, `p05=0.133x`, `p10=0.297x`, `p25=0.481x`, `p50=0.620x`, `p75=0.759x`, `p90=0.906x`, `p95=0.994x`, `p99=1.212x`
- **Quantiles (RSC-truncated traces, n=24,300)**: `p01=0.004x`, `p05=0.013x`, `p10=0.024x`, `p25=0.056x`, `p50=0.120x`, `p75=0.232x`, `p90=0.386x`, `p95=0.506x`, `p99=0.787x`

## 3. Top-10 Terminal-Bench Models with `0` Step Traces (`steps = 'null'`)

| Rank | Model | Weights | Pass Rate | Total Trials | Step-Logged Runs | Harnesses |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `gpt-5.3-codex@openai` | Closed | 71.37% | 1,778 | 0 | `Factory Droid, mux, simple_codex, terminus-2` |
| 2 | `gpt-5.3-codex@OpenAI` | Closed | 70.34% | 445 | 0 | `ruley` |
| 3 | `gemini-3-pro-preview@Google` | Closed | 64.09% | 1,334 | 0 | `ante, ruley, sage` |
| 4 | `gpt-5.1-codex-max@openai` | Closed | 60.45% | 445 | 0 | `codex` |
| 5 | `gemini-3-pro-preview@google` | Closed | 58.54% | 890 | 0 | `Factory Droid, letta-code` |
| 6 | `claude-opus-4-5@anthropic` | Closed | 56.34% | 1,065 | 0 | `goose, letta-code, mux, opencode` |
| 7 | `gemini-3-flash-preview@google` | Closed | 51.01% | 445 | 0 | `gemini-cli` |
| 8 | `gemini-3-flash-preview@Google` | Closed | 47.42% | 445 | 0 | `gemini-cli` |
| 9 | `claude-sonnet-4-5@anthropic` | Closed | 44.83% | 890 | 0 | `TerminalBenchAgent, goose` |
| 10 | `kimi-k2.5@kimi` | Open | 43.45% | 435 | 0 | `terminus-2` |

