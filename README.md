# agent-loops

Offline CPU-only study measuring how often AI coding and terminal agents get stuck in loops, what share of their cache-aware dollar bill those loop steps consume, and how well real-harness loop detectors catch them relative to matched-random stopping baselines.

## Datasets (Downloaded Once, Cached Locally under `data/`)

1. **Terminal-Bench 2.0 Trajectories** (`yoonholee/terminalbench-trajectories`): 52,104 trials across 89 tasks, 26 agent harnesses (9 with recorded step traces, 34,462 trajectories), and 49 models (33 with step traces).
2. **SWE-rebench July 2026 Trajectories** (`ibragim-bad/swe_rebench_07_2026_trajectories`): 9,435 normalized trajectories across 111 tasks × 17 participants × 5 runs (`claude-code`, `codex`, `cursor`, `junie`, and 13 standalone models in a common scaffold).
3. **τ²-bench Final Results** (`sierra-research/tau2-bench`, `data/tau2/results/final`): 26 result files (10,832 conversational tool-use simulations) as a sanity check on non-coding conversational tool loops.

## Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make all
```

`make all` runs the unit test suite (`pytest`) and executes `run_all.py`, which caches public datasets locally under `data/` (gitignored), prints schemas, labels every trajectory step, rebuilds cache-aware per-step costs with pre-registered validation bounds (`[0.50x, 2.00x]`), replays loop detectors + 200-seed matched-random baselines, writes CSV/JSON/Markdown outputs to `results/`, and generates PNG figures under `assets/`.

## Repository Layout

- `src/agent_loops/download.py` — Dataset download, local caching, and schema inspection.
- `src/agent_loops/canonicalize.py` — Tool-call canonicalization, noise-normalized observation/error hashing, OpenHands step-pair collapsing, and file read/write tracking.
- `src/agent_loops/labeler.py` — Per-step labeling (`exact_loop`, `near_duplicate`, `aba_thrash`, `period_k_rotation`, `error_streak`, `redundant_reread`, `productive_repeat`).
- `src/agent_loops/cost.py` — Cache-aware cumulative prompt pricing, rescaling, compaction/subagent filtering, and mismatch dropping.
- `src/agent_loops/detectors.py` — Detector replay (`cline_style`, `openhands_stuck`, `gemini_cli`, `openclaw_10`, `openclaw_20`, step caps, `1.25x P90` harness turn budget) and 200-seed matched-random baseline.
- `src/agent_loops/study.py` — Dataset parsers and aggregation runners.
- `src/agent_loops/plots.py` & `src/agent_loops/report.py` — Plot and programmatic report generation (`results/summary.md`, `results/loop_prone_tasks.csv`).
- `tests/` — Unit tests for canonicalization, step labeling, cost rebuild, and detector replay.
- `NOTES.md` — Engineering log, setup, findings, visuals, and limitations.
