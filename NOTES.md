# Agent Loops Study — Engineering Notes

## Goal
Measure offline on an Apple Silicon laptop (CPU-only, no LLM API calls, no GPU) how often AI coding and terminal agents get stuck in loops, what share of their real cache-aware dollar bill those loop steps consume, and how well production-harness loop detectors would have caught them relative to matched-random stopping baselines.

## Setup
- **Hardware**: Apple MacBook Air (Apple M3, 8 CPU cores, 16 GB RAM; CPU-only offline pipeline).
- **Environment**: Python 3.13.7, local virtualenv `.venv/` managed with `requirements.txt` (`huggingface_hub`, `pyarrow`, `pandas`, `numpy`, `matplotlib`, `orjson`, `pytest`, `requests`, `tqdm`).
- **Pre-registered parameters** (locked a priori before inspecting cost discrepancy distributions):
  - Rebuild validation bounds: `REBUILD_MIN_RATIO = 0.50`, `REBUILD_MAX_RATIO = 2.00` on `raw_estimated / reported`.
  - Fixed seed: `42` (`200` matched-random bootstrap seeds per detector, matched per harness for firing count and stop-step distribution).
- **Datasets (downloaded once and cached under gitignored `data/`)**:
  1. **Primary — Terminal-Bench 2.0 Trajectories**: [`yoonholee/terminalbench-trajectories`](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories) — 52,104 trials across 89 tasks, 26 agent harnesses (9 with recorded step traces across 34,462 runs), 49 models (33 with step traces); 2 Parquet shards (~221 MB compressed).
  2. **Secondary — SWE-rebench July 2026 Trajectories**: [`ibragim-bad/swe_rebench_07_2026_trajectories`](https://huggingface.co/datasets/ibragim-bad/swe_rebench_07_2026_trajectories) — 9,435 normalized trajectories (111 tasks × 17 participants × 5 runs, including `claude-code`, `codex`, `cursor`, `junie`, and 13 standalone models); 85 `.jsonl.gz` shards (~481 MB gzipped).
  3. **Sanity check — τ²-bench Final Results**: [`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/results/final) — 26 JSON result files spanning 10,832 simulated conversations across `airline` (800), `retail` (1,824), `telecom` (4,560), and `telecom-workflow` (3,648).

## Log
- **2026-11-10 (setup & schema inspection)**: Initialized project under `agent-loops`, downloaded and cached all three public datasets under `data/`, and printed schemas in `download.py`. Confirmed `yoonholee/terminalbench-trajectories` stores `steps` as JSON strings in 34,462 of 52,104 rows (with 17,642 null/empty rows).
- **2026-11-10 (canonicalization, labeler, cost rebuilder, detectors)**:
  - Built ANSI/whitespace/timestamp/epoch/UUID/temp-path normalization, benign `Error: (none)` filtering for `gemini-cli`, and file read/write tracking in `src/agent_loops/canonicalize.py`.
  - Added Next.js RSC `$<hex>` pointer skipping and coarse path-only edit handling (`Edit`, `Write`, `replace`, `str_replace_editor`) so sequential edits with unlogged diffs are not mislabeled as loops.
  - Added `collapse_openhands_tb_steps` in `canonicalize.py` to merge OpenHands Terminal-Bench action-announcement + result step pairs into single logical actions.
  - Implemented 7-class priority step labeler (`exact_loop`, `near_duplicate`, `aba_thrash`, `period_k_rotation` with `k <= 5`, `error_streak`, `redundant_reread`, `productive_repeat`, with `productive_repeat` excluded from `is_loop`).
  - Implemented cumulative cache-aware cost rebuilder in `src/agent_loops/cost.py` with subagent/compaction filtering and `[0.50x, 2.00x]` rebuild validation.
  - Implemented detector replay (`cline_style`, `openhands_stuck`, `gemini_cli`, `openclaw_10`, `openclaw_20`, step caps `30/50/75/100/150`, and `1.25x P90` harness turn budget) plus `O(M+k)` swap-and-pop 200-seed matched-random evaluator in `src/agent_loops/detectors.py`.
- **2026-11-10 (bug fixes from first audit & final clean run)**:
  - Inspected top detector firing runs and caught two false-positive modes: (1) `junie` on SWE-rebench emits `agent_step` summary events (`Opened file`, `Edited files`) rather than raw `tool_call`s, causing `Opened file` with empty args to trigger `cline_style`; fixed `_parse_swe_steps` in `src/agent_loops/study.py` to extract file path / coarse write markers. (2) Cross-turn XML markup (`<tool_call>`) in `qwen3.5-35b-a3b` caused repeated 50-char chunks across turns to trigger `gemini_cli`; scoped content-chunk detection in `detect_gemini_cli` to intra-turn assistant text chunks.
  - Fixed colliding scatter/Pareto chart label offsets in `src/agent_loops/plots.py` and re-ran `make clean && make all`: all **24 pytest tests passed in 0.09s**, full study finished in **118.07s** (`download/schema=0.12s`, `TB=34.04s`, `SWE=77.61s`, `tau2=5.74s`, `plots/report=0.56s`).

## Key results
All figures below come verbatim from `results/summary.md`, `results/study_metrics.json`, `results/tb_by_model.csv`, `results/swe_by_model.csv`, and `results/loop_prone_tasks.csv`.

1. **Loop prevalence, step share, and cache-aware dollar share**:
   - **Terminal-Bench 2.0 (34,462 runs with steps; 1,075,547 steps)**:
     - **31.00% (10,684)** of runs exhibit at least one pathological loop; **7.38% (79,344 / 1,075,547)** of steps are loop steps (`productive_repeat` adds **5.54%** non-loop steps).
     - Across **5,011** cost-validated runs (`$3,330.95` total), loop steps consume **17.71% ($589.76 / $3,330.95)** of cache-aware spend (more than double their 7.38% step share due to late-trajectory context bloat).
     - Pass rate drops from **34.27%** (loop-free) to **27.42%** (looping).
     - In the zero-RSC clean-trace subset (`5,371` runs; `3,787` cost-validated), loop run prevalence is **41.95% (2,253)**, loop step share **12.31%**, and loop dollar share **14.94%** (`$397.96 / $2,663.99`).
   - **SWE-rebench July 2026 (9,435 runs; 631,803 steps)**:
     - **54.99% (5,188)** of runs exhibit at least one loop; **4.11% (25,975 / 631,803)** of steps are loop steps (`productive_repeat` **1.90%**).
     - Across **6,381** cost-validated runs (`$4,927.87` total), pooled loop dollar share is **2.38% ($117.38 / $4,927.87)** — lower than the pooled step share because of a **Simpson's paradox** across models (expensive, low-looping models like `grok-4.5` at `$1,246.45` and `opus-5` at `$909.26` dominate total spend while cheap open-weight Qwen/Minimax models account for most loop steps; within most individual models/harnesses such as `minimax-m3` `[5.95% steps vs 7.60% $]`, `sonnet-5` `[1.96% vs 2.79%]`, `cursor/composer-2.5` `[2.75% vs 3.90%]`, `mimo-v2.5-pro` `[4.52% vs 5.02%]`, `fable-5` `[0.71% vs 0.87%]`, and `grok-4.5` `[0.65% vs 0.82%]`, dollar share still exceeds step share).
     - Pass rate drops sharply from **59.17%** (loop-free) to **43.14%** (looping).
   - **τ²-bench sanity check (10,832 sims across 26 files; 99,236 tool calls)**:
     - **25.56% (2,769)** of simulations contain a pathological loop; **10.96% (10,872)** of tool steps are loop steps.
     - Pass rate drops from **72.45%** (loop-free) to **48.36%** (looping).

2. **Taxonomy & why Cline-style 5-step detectors miss cycles**:
   - Terminal-Bench run counts: `exact_loop` **8,225 (23.87%)**, `near_duplicate` **3,421 (9.93%)**, `error_streak` **2,645 (7.68%)**, `aba_thrash` **2,578 (7.48%)**, `redundant_reread` **582 (1.69%)**, `period_k_rotation (k<=5)` **334 (0.97%)**; non-loop `productive_repeat` **15,292 (44.37%)**.
   - SWE-rebench run counts: `exact_loop` **3,639 (38.57%)**, `near_duplicate` **3,147 (33.35%)**, `error_streak` **1,238 (13.12%)**, `redundant_reread` **1,173 (12.43%)**, `aba_thrash` **841 (8.91%)**, `period_k_rotation (k<=5)` **120 (1.27%)**; non-loop `productive_repeat` **4,059 (43.02%)**.
   - **`period_k_rotation` runs missed by `cline_style`**: **277/334 (82.93%)** on Terminal-Bench (67/79 = 84.81% on zero-RSC clean traces) and **118/120 (98.33%)** on SWE-rebench.

3. **Detectors vs 200-seed matched-random stopping baselines**:
   - **Rule-based production detectors** achieve lower false-stop rates (FSR) than matched-random stopping at identical firing counts and stop-step positions, but save only a small fraction of total spend because they fire conservatively (`0.25%`–`2.26%` of TB runs; `0.07%`–`4.44%` of SWE runs):
     - **TB `gemini_cli`**: fires `2.26%` (780), saves **5.71% ($190.06)** vs matched-random `4.33% [2.99%, 6.43%]`, FSR **10.77%** vs random `23.18% [20.51%, 25.90%]`.
     - **TB `openclaw_10`**: fires `1.40%` (482), saves **5.05% ($168.28)**, FSR **13.49%** vs random `14.03% [10.58%, 17.23%]`.
     - **TB `cline_style`**: fires `1.18%` (408), saves **5.00% ($166.68)**, FSR **18.38%** vs random `23.05% [19.60%, 26.72%]`.
     - **TB `openhands_stuck`**: fires `1.01%` (347), saves **1.82% ($60.59)**, FSR **11.53%** vs random `19.35% [16.14%, 23.06%]`.
     - **TB `openclaw_20`**: fires `0.25%` (86), saves **0.98% ($32.68)**, FSR **3.49%** vs random `10.70% [4.65%, 16.28%]`.
     - **SWE `gemini_cli`**: fires `4.44%` (419, mostly intra-turn chanting in `minimax-m3` / `qwen3.6`), saves **2.05% ($100.87)** vs random `1.62% [1.42%, 1.82%]`, FSR **22.20%** vs random `28.60% [25.04%, 31.99%]`. Rule detectors on SWE otherwise save `<1%` (`cline_style` 0.29% / FSR 37.14%; `openhands_stuck` 0.92% / FSR 41.18%; `openclaw_10` 0.04% / FSR 29.41%).
   - **Step caps & `1.25x P90` harness turn budgets** dominate total dollar savings (`55.36%` saved at `13.17%` FSR for `turn_budget_1_25x_p90` on TB; `4.06%` at `30.00%` FSR on SWE), closely tracking matched-random curves because hard turn truncation depends mainly on cutoff timing rather than detecting specific loop structure.

4. **Top Gemini loop-prone Terminal-Bench tasks** (see full 20-task table in `results/loop_prone_tasks.csv`):
   - Top 5 by Gemini loop-run rate: `winning-avg-corewars` (60.00%, 55 runs), `crack-7z-hash` (59.26%, 54 runs, 36.58% step share), `db-wal-recovery` (59.26%, 54 runs, 3.70% pass rate), `largest-eigenval` (58.18%, 55 runs), `break-filter-js-from-html` (55.56%, 54 runs).

## Visuals
1. [step_vs_dollar_share_per_model.png](file:///Users/davidflorez/.gemini/jetski/scratch/agent-loops/assets/step_vs_dollar_share_per_model.png) — Loop step share vs cache-aware dollar share per model (`Terminal-Bench 2.0` vs `SWE-rebench July 2026`); models above `y = x` incur disproportionate spend on late-trajectory loop steps (e.g., `gemini-2.5-flash` at 15.77% steps / 70.69% dollars on cost-validated TB runs).
2. [detector_savings_vs_false_stop.png](file:///Users/davidflorez/.gemini/jetski/scratch/agent-loops/assets/detector_savings_vs_false_stop.png) — Pareto comparison of rule-based loop detectors, step caps, and `1.25x P90` turn budgets against 200-seed matched-random baselines (with 95% CIs).
3. [loop_prevalence_by_harness.png](file:///Users/davidflorez/.gemini/jetski/scratch/agent-loops/assets/loop_prevalence_by_harness.png) — Any-loop vs `period_k_rotation (k<=5)` prevalence across all 9 TB harnesses (`11.7%` in `terminus-3-3` to `58.1%` in `openhands`) and all 17 SWE-rebench participants (`9.0%` in `gpt-5.6-luna` to `99.3%` in `minimax-m3`).

## Surprises and limitations
1. **Terminal-Bench missing steps**: Only `34,462` of `52,104` TB rows (9 of 26 agents, 33 of 49 models) include step traces; `17,642` rows have `steps = "null"`.
2. **Next.js RSC `$<hex>` chunk pointers in Terminal-Bench**: Exported `tbench.ai` trajectories left unresolved React Server Components chunk IDs (`$31`, `$4a`, etc.) inside ~29k rows (`gemini-3*`, `gpt-5.2`, `glm-5`, `minimax-m2.5`, plus some `claude`/`codex` harnesses often also paired with `cost_cents = 0`). Our pre-registered `[0.50x, 2.00x]` filter dropped `24,543` rebuild mismatches and `4,408` missing-cost rows instead of inventing unvalidated dollar shares; `results/summary.md` reports both full-TB cost-validated runs (`5,011`) and the zero-RSC clean subset (`5,371` runs / `3,787` cost-validated).
3. **OpenHands TB step-pair doubling**: OpenHands Terminal-Bench trajectories emit action-announcement + execution-observation pairs (`201,836` pairs with duplicated `tools`), requiring `collapse_openhands_tb_steps` so single tool actions do not look like consecutive identical calls.
4. **Coarse path-only edit logging**: TB harnesses logging `Edit`, `Write`, `replace`, or `str_replace_editor` (and `swe_rebench` `junie` `"Edited files"`) omit file diff payloads; treated as `args_known=False` so iterative edits on the same file do not falsely register as identical-arg loops.
5. **SWE-rebench token accounting differences**: `claude-code`, `cursor`, and `junie` exclude `cached_input` / `cache_creation` from reported `input` tokens, whereas the 14 other SWE participants report `total == input + output` with `cached_input` included in `input`. Handled explicitly in `_extract_swe_tokens`.
6. **SWE-rebench prompt visibility & context eviction drops**: `claude-code` omits its built-in system prompt (`unavailable: ["built_in_system_prompt"]`, causing `raw_ratio ~ 0.19` vs `[0.50x, 2.00x]`), `cursor` evicts/truncates older context rather than pure append-only (`ratio ~ 3.11`), and `junie` uses `aux_models` (`554/555` runs flagged as subagent usage). All `2,494` mismatch runs and `554` subagent runs are cleanly excluded by the pre-registered audit rules.
7. **SWE-rebench Simpson's paradox**: Pooled SWE loop dollar share (`2.38%`) sits below step share (`4.11%`) only because expensive low-looping models dominate total USD while cheap models loop more; within most individual participants (`minimax-m3`, `sonnet-5`, `cursor`, `mimo-v2.5-pro`, `fable-5`, `grok-4.5`), loop dollar share exceeds loop step share.
8. **True intra-turn chanting vs XML leakage**: `minimax-m3` and `qwen3.6` exhibit real intra-turn 50-char token chanting caught by `detect_gemini_cli`, whereas naive cross-turn buffering falsely flagged `<tool_call>` XML tags across turns until scoped per turn.
9. **Observation truncation & replay limits**: Terminal-Bench observations are truncated to 5,000 chars, and detector replay is counterfactually offline (earlier interventions might alter downstream recovery rather than deterministically truncating at trigger time).

## Open questions / next steps
- Test hybrid detectors that combine rotation/thrash memory (period `k <= 5` A-B-A / multi-step cycles over canonicalized commands & exit codes) with adaptive step budgets instead of fixed thresholds.
- Quantify cache-write / cache-miss spikes immediately following context compaction or idle TTL expiry where raw vendor logs include per-turn usage.
- Add full system-prompt token lengths for harness wrappers like `claude-code` if upstream telemetry becomes available so `claude-code` passes the `[0.50x, 2.00x]` rebuild check without relaxing validation bounds.
