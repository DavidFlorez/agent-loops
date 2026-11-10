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

---

## Phase 2 Pre-Registration: Do Open-Weight Models Loop More Than Closed Models?

**Pre-registration lock**: The datasets, uniform-scaffold cohort boundaries, exclusions, pass-rate capability bands, uncertainty estimator, statistical regression specifications, and sensitivity checks below are locked and committed **before** computing or inspecting any Open-vs-Closed loop outcomes across the four datasets.

### Research Question & Neutrality Rule
Measure offline (CPU-only, fixed seed `42`, `labeler.py` untouched) whether open-weight models loop more than closed/proprietary models in agent trajectories when compared fairly within identical scaffolds and common task intersections, matched on capability, and adjusted for task difficulty, serving route, release vintage, and scaffold step caps. Every family (including Google's Gemini, Anthropic Claude, OpenAI GPT / `gpt-oss`, xAI Grok, Alibaba Qwen open vs proprietary Max, Z.ai GLM, DeepSeek, MiniMax, Moonshot Kimi, Thinking Machines Inkling, Xiaomi MiMo) is reported verbatim; null and reversed effects are given equal prominence.

### Datasets & Schema / Token-Cost Coverage (Inspected 2026-11-10)
All four public datasets are cached locally under gitignored `data/`:

1. **Terminal-Bench 2.0** ([`yoonholee/terminalbench-trajectories`](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories), `2` Parquet shards, `52,104` rows, `89` tasks):
   - Columns: `task_name, agent, model, reward, duration_seconds, input_tokens, output_tokens, cache_tokens, cost_cents, trial_name, trial_id, started_at, ended_at, steps` (JSON array with `{msg, obs, src, tools: [{fn, cmd}]}`).
   - Custom closed harnesses (`claude-code`, `codex`, `gemini-cli`, etc.) are **never** pooled against generic scaffolds. Fair within-scaffold slices (`89/89` shared tasks each):
     - **`tb_terminus_2` (Primary TB slice)**: `29` step-logged models (`3` null-step models excluded: `glm-4.7@z-ai`, `gpt-5.3-codex@openai`, `kimi-k2.5@kimi`).
     - **`tb_openhands` (Secondary TB slice)**: `12` step-logged models (`claude-opus-4-5` lacks tokens/cost).
     - **`tb_mini_swe_agent` (Secondary TB slice)**: `13` step-logged models.
   - TB token/cost gaps recorded in `terminus-2`: `gemini-3-flash-preview@gemini`, `glm-5@z-ai`, `gpt-5.2@openai`, `minimax-m2.1@openai`, and `openhands/claude-opus-4-5` have `tokens=0, cost=0`; `minimax-m2.5@Minimax` has tokens (`444/445`) with `cost_cents=0`. Models lacking `cost_cents` or failing `[0.50x, 2.00x]` rebuild validation are omitted from `loop_dollar_share` (never backfilled unless token totals pass `[0.50x, 2.00x]`), while keeping full step/loop metrics where `steps` exist.

2. **SWE-rebench July 2026** ([`ibragim-bad/swe_rebench_07_2026_trajectories`](https://huggingface.co/datasets/ibragim-bad/swe_rebench_07_2026_trajectories), `85` gzipped JSONL shards, `9,435` rows):
   - Primary uniform cohort **`swe_rebench_swelike`**: `participant.type == "common_scaffold_model"` (`scaffold == "swelike"` only; `claude-code`, `codex`, `cursor`, `junie` excluded from Open-vs-Closed pooling), yielding **13 models** (`7` Open, `6` Closed) × `5` runs × `111/111` shared tasks = `7,215` runs.
   - Token/cost coverage: `555/555` runs per model have `usage.tokens` (including `cached_input > 0`); `0/555` report `cost_usd` (`cost_usd = null`), so `loop_dollar_share` uses cache-aware token rebuild validated at `[0.50x, 2.00x]` via public API list pricing. Max observed steps = `492` (`500`-step scaffold budget).

3. **Longtail-SWE** ([`nebius/Longtail-SWE-trajectories`](https://huggingface.co/datasets/nebius/Longtail-SWE-trajectories), `arXiv:2611.10304`, `9` Parquet shards, `5,454` rows, `202/202` shared tasks × `3` runs × `2` cohorts):
   - **`longtail_swe_swelike`**: All `9` configurations use `swelike` (`5` Open: `deepseek-v4-flash`, `glm-5.3-flash`, `kimi-k3`, `qwen3.6-27b`, `qwen3.8-27b`; `4` Closed: `claude-opus-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-6.1-sol`). Schema (`1.2.0`) mirrors SWE-rebench `events` (`prompt`, `assistant_message`, `tool_call`, `tool_result`, `final_message`).
   - Important schema/confounder observations documented a priori:
     - `gpt-6.1-sol` has `602/606` graded rows (`4` `resolved = null` skipped/timeout excluded from pass-rate denominator, kept for step traces if `n_steps >= 1`).
     - `deepseek-v4-flash` has `49` zero-step API timeout failures (`steps = 0, tokens = 0, wall_time ~ 300s`).
     - Context-limit asymmetry: Closed models + `deepseek-v4-flash` ran at `1,000K–1,050K` (`effective_context_tokens`), whereas `kimi-k3`, `glm-5.3-flash`, and `qwen3.8-27b` ran at `262,144`, and `qwen3.6-27b` ran at `262,144` (original) / `131,072` (statement_rerun).
     - `cost_usd` is `null` across all 9 models; `usage.tokens` (`total` and `cached_input`) is populated on all executed (`n_steps >= 1`) runs. Observed max steps = `501` (`500` turn cap).
     - Benchmark curation provenance: `Qwen 3.8 27B` was excluded from difficulty-calibration test attempts; `Kimi K3`, `GPT-5.6 Luna`, and `Claude Opus 5.5` participated in benchmark task curation (`NOTES`/limitations flag).

4. **Sierra `tau2-bench` Public S3** (`s3://sierra-tau-bench-public/submissions/`, `27` non-voice `_sierra_` submissions cached):
   - Schema inspection revealed two critical scaffold boundaries enforced a priori:
     - **Uniform user simulator filter**: `gpt-5_sierra_2025-08-09` used `gpt-4.1-2025-04-14` as customer simulator, whereas all 26 2026 `_sierra_` submissions used `gpt-5.2` (`reasoning_effort = low`, `max_steps = 200` message turns, `max_errors = 10`). Therefore `gpt-5_sierra_2025-08-09` is excluded from fair pooling.
     - **Two distinct uniform-scaffold cohorts**:
       1. **`tau2_classic`** (`airline` + `retail` + `telecom`, `278/278` shared tasks × `4` trials = `1,112` runs/model): `8` models (`2` Open: `glm-5-think`, `qwen3.5-397b-a17b-think`; `6` Closed: `claude-opus-4-5`, `claude-sonnet-4-5`, `gemini-3-flash`, `gemini-3-pro`, `gpt-5-2`, `gpt-5-2-none`).
       2. **`tau2_banking_knowledge_alltools`** (`banking_knowledge`, strictly matching `retrieval_config in {"AllTools", "alltools"}`, re-graded under `tau2-bench v1.0.1`, `97/97` shared tasks × `4` trials = `388` runs/model): **`20` models** (`3` Open: `glm-5-2`, `inkling`, `kimi-k3`; `17` Closed: `claude-fable-5`, `claude-opus-4-5`, `4-6`, `4-7`, `4-8`, `5`, `gemini-2-5-pro`, `gemini-3-1-pro-preview`, `gpt-5-2`, `gpt-5-4`, `gpt-5-5`, `gpt-5-6-sol`, `grok-4-fast`, `grok-4-1-fast`, `grok-4-2`, `grok-4-5`, `qwen3-8-max`).
          - The `6` non-`AllTools` `banking_knowledge` submissions used incompatible tool/retrieval scaffolds (`terminal_use` without KB search: `claude-sonnet-4-5`, `gemini-3-flash`, `gemini-3-pro`; `openai_embeddings`: `glm-5-think`, `qwen3.5-397b`; `qwen_embeddings`: `gpt-5-2-none`) and are **excluded from `tau2_banking_knowledge_alltools`** (reported separately in inventory & Gemini notes).
   - Sierra token/cost coverage: every executed simulation records per-turn `usage.prompt_tokens` and `usage.completion_tokens` (`cached_tokens` not broken out). `agent_cost > 0` is reported for 18 direct-provider/Vertex submissions and `0.0` for `claude-opus-4-6`, OpenRouter proxies (`glm-5-2`, `inkling`, `kimi-k3`, `qwen3-8-max`, `grok-4-5`), and self-hosted vLLM (`glm-5-think`, `qwen3.5-397b`). Where `agent_cost > 0`, turn `cost` is summed across executed steps (`reported_provider_usd`); where `agent_cost == 0.0` and tokens exist, per-turn USD is priced from `(prompt_tokens, completion_tokens)` × public API rates (`rebuilt_from_turn_tokens`), validated and tagged with explicit `cost_source` provenance so imputed vs provider-billed USD are never conflated.

### Model Classification & Covariate Definitions (`src/agent_loops/model_catalog.py`)
- **Open-Weights vs Closed edge cases**: `openai/gpt-oss-120b/20b`, `inkling` (`Thinking Machines`, Apache-2.0), `kimi-k3`, `glm-5.2/5.3-flash`, `deepseek-v4-flash/pro`, `qwen3.5/3.6/3.8-27b`, `minimax-m2..m3`, `mimo-v2.5-pro` are **Open-Weights**; **`qwen3-8-max` (`Qwen 3.8 Max`)**, `claude-*`, `gemini-*`, `gpt-5*/6*` (proprietary), `grok-*`, and `composer-2.5` are **Closed**.
- **Serving Route (`first_party` vs `third_party`)**:
  - TB: `@together_ai`, `@fireworks_ai`, `@modelscope`, and `minimax-m2.1@openai` are `third_party`; vendor native `@anthropic`, `@openai` (GPT), `@gemini`/`@google`, `@xai`, `@deepseek`, `@z-ai`, `@minimax`, `@kimi` are `first_party`.
  - Sierra: `openrouter/*` (incl. `glm-5-2`, `inkling`, `kimi-k3`, `grok-4-5`, `qwen3-8-max`) and self-hosted `openai/glm-5-fp8` / `openai/qwen3.5-397b` are `third_party`; `anthropic/*`, `claude-*`, `vertex_ai/*`, `openai/responses/*`, `gpt-*`, `xai/*` are `first_party`.
  - SWE-rebench `swelike`: uniform benchmark API gateway across all 13 models (`serving_route` constant within cohort, noted in regression output).
  - Longtail-SWE `swelike`: official Anthropic/OpenAI APIs (`first_party`) for Closed; third-party/hosted endpoints (`third_party`) for Open.

### Pre-Registered Outcome Metrics, Denominators & Uncertainty
1. **Zero-step execution rule**: Primary behavioral denominators restrict to **executed runs (`n_steps >= 1`)** within each cohort's exact shared task intersection, because immediate `n_steps == 0` API timeouts (e.g., `49` `deepseek-v4-flash` timeouts on Longtail-SWE, or `infrastructure_error` at turn 0 on Sierra) never invoke a step and would otherwise artificially dilute `any_loop_rate`. Dropped zero-step counts are logged transparently in `results/loops_by_model_scaffold.csv`.
2. **Sierra step extraction**: Every assistant turn emits one `NormalizedStep` per tool call (`tool_calls`) or, when an assistant message replies to the simulated user without calling tools, one non-tool dialogue step paired with the user reply; `label_trajectory` (`src/agent_loops/labeler.py`, **unchanged**) evaluates pathological tool loops over tool-bearing steps. Both `loops_per_100_steps` (`100 * loop_steps / n_steps`) and `loops_per_100_tool_steps` (`100 * loop_steps / n_tool_steps`) are reported.
3. **Primary outcomes**:
   - `loops_per_100_steps = 100 * sum(loop_steps) / sum(n_steps)` (exposure-normalized loop density).
   - `any_loop_rate = mean(has_any_loop)` (share of executed runs with `>= 1` pathological loop step).
   - Secondary outcomes: `loop_dollar_share` (cost-validated runs), `pass_rate_looping` vs `pass_rate_clean`, and taxonomy rates (`exact_loop`, `near_duplicate`, `aba_thrash`, `period_k_rotation`, `error_streak`, `redundant_reread`, `productive_repeat`).
4. **Confidence Intervals**: **Task-clustered bootstrap (`n_boot = 1,000`, `seed = 42`, percentile 2.5%–97.5%)** resampling `task_id` clusters with replacement within each `(cohort, model)` or matched stratum so repeated trials on the same task are treated as correlated clusters.
5. **Equal-model vs run-weighted aggregation**: Unmatched cohort Open-vs-Closed means weight each model equally (so TB models with 900+ trials do not receive double weight vs 445-trial models); run-weighted pooled totals are reported alongside for completeness.

### Pre-Registered Capability Matching & Regression Models
- **Uniform Capability Pass-Rate Bands**: `B1 [0.00, 0.25)`, `B2 [0.25, 0.40)`, `B3 [0.40, 0.55)`, `B4 [0.55, 0.75)`, `B5 [0.75, 1.01]`, chosen a priori without inspecting loop outcomes.
  - Within each cohort, any band containing `>= 1` Open model and `>= 1` Closed model is an overlapping stratum.
  - Within each overlapping band `b`, `Delta_b = equal_model_mean_Open,b - equal_model_mean_Closed,b`.
  - Cohort matched difference averages `Delta_b` across overlapping bands weighted by balanced common-support `w_b = min(N_open_b, N_closed_b)` (bootstrap CIs re-computed inside each bootstrap task-cluster replicate).
- **Regression Specifications (per cohort & pooled within-scaffold)**:
  - **Count model**: **Negative Binomial GLM** with `log(n_steps)` offset (`Poisson` fallback checked if dispersion `<= 0`) modeling `loop_steps ~ is_open + pass_rate_10pp + release_months [+ serving_route] [+ task_id fixed effects]`, with **two-way Cameron-Gelbach-Miller cluster-robust standard errors clustered simultaneously on `(model_id, task_id)`**.
  - **Binary model**: **Logistic regression (`Logit`)** for `has_any_loop` (both with and without `log1p(n_steps)` exposure control) with task fixed effects and two-way `(model_id, task_id)` cluster-robust SEs.
  - **Mixed-effects robustness check**: `statsmodels` `PoissonBayesMixedGLM` (with `log(n_steps)` offset) / `BinomialBayesMixedGLM` including random `task_id` intercept alongside model-level covariates.
  - Whenever a covariate (`serving_route`, `step_cap`) is collinear with `is_open` or constant within a cohort, it is automatically omitted from that single-cohort fit and noted explicitly.

### Pre-Registered Sensitivity Checks (All 5 Reported)
1. **`pass_only`**: Only `succeeded == True` executed trajectories (separates intrinsic looping/verification habits from brick-wall task failure).
2. **`fail_only`**: Only `succeeded == False` executed trajectories.
3. **`drop_gpt_oss`**: Drops legacy `openai/gpt-oss-20b` and `openai/gpt-oss-120b` on Terminal-Bench.
4. **`first_party_only`**: Retains `first_party` API routes only (and explicitly marks cohorts where Open models only ran via third-party gateways).
5. **`per_family`**: Model-family breakdown (`claude`, `gpt`, `gemini`, `grok`, `qwen`, `glm`, `minimax`, `deepseek`, `kimi`, `gpt_oss`, `inkling`, `mimo`) across every cohort.

