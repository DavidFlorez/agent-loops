# agent-loops

Offline CPU-only study analyzing how often AI agents loop, what share of cache-aware spend loops consume, how production-harness loop detectors compare against matched-random stopping baselines, and **whether open-weight models loop more than closed models once compared fairly within identical scaffolds, shared tasks, capability bands, and task/vintage/serving controls**.

## Datasets (Downloaded Once, Cached Locally under `data/`)

1. **Terminal-Bench 2.0 Trajectories** (`yoonholee/terminalbench-trajectories`): 52,104 trials across 89 tasks, 26 agent harnesses (9 with step traces, 34,462 trajectories), and 49 models; uniform-scaffold Open-vs-Closed comparison uses `terminus-2` (29 step-logged models), `openhands` (12), and `mini-swe-agent` (13).
2. **SWE-rebench July 2026 Trajectories** (`ibragim-bad/swe_rebench_07_2026_trajectories`): 9,435 trajectories across 111 tasks × 17 participants × 5 runs; Open-vs-Closed comparison restricts strictly to the 13 `common_scaffold_model` (`swelike`) runs.
3. **Longtail-SWE Trajectories** (`nebius/Longtail-SWE-trajectories`, arXiv:2611.10304): 5,454 `swelike` trajectories across 202 hard tasks × 9 models (`5` Open, `4` Closed) × 3 runs.
4. **Sierra τ²-bench Public S3 Archive** (`s3://sierra-tau-bench-public/submissions/`) + bundled GitHub `data/tau2/results/final`: 27 non-voice `_sierra_` submissions split cleanly into `tau2_classic` (`airline+retail+telecom`, `8` models, `278` shared tasks, `gpt-5.2` user simulator) and `tau2_banking_knowledge_alltools` (`banking_knowledge`, strictly `AllTools` retrieval, `20` models, `97` shared tasks).

## Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make all
```

`make all` runs the full `pytest` suite and executes `run_all.py` (fixed seed `42`), which verifies/caches datasets under gitignored `data/`, prints all 4 schemas and token/cost coverage, labels every step via `src/agent_loops/labeler.py`, runs detector replay + cache-aware cost rebuild, executes Phase 2 within-scaffold task-clustered bootstraps and regressions, writes outputs to `results/` + `ARTICLE_DRAFT.md`, and generates PNG figures under `assets/`.

## Key Outputs & Repository Layout

- `NOTES.md` — Phase 1 engineering log plus committed Phase 2 pre-registration lock.
- `results/summary.md` & `results/study_metrics.json` — Combined Phase 1 + Phase 2 headline metrics.
- `results/loops_by_model_scaffold.csv` — Per-(cohort, model) loop rates, 95% task-clustered CIs, pass rates, dollar share, taxonomy breakdown, and serving/context provenance.
- `results/open_vs_closed_model.md` — Unmatched vs capability-band-matched differences, Negative Binomial / Logit / GEE / Binomial-GLMM regressions, all 5 sensitivity checks (`pass_only`, `fail_only`, `drop_gpt_oss`, `first_party_only`, `per_family`), and Google Gemini breakdown.
- `ARTICLE_DRAFT.md` — Programmatically generated long-form write-up grounded in computed metrics.
- `assets/*.png` — 6 publication-quality PNG figures across Phase 1 and Phase 2.
- `src/agent_loops/` — `download.py`, `canonicalize.py`, `labeler.py` (unchanged), `cost.py`, `detectors.py`, `model_catalog.py`, `study.py`, `fair_study.py`, `plots.py`, `report.py`, `extra_reports.py`, `article.py`.
- `tests/` — Unit tests covering canonicalization, step labeling, cost rebuild, detectors, model catalog, Longtail/Sierra parsers, and fair-comparison matching/regressions.
