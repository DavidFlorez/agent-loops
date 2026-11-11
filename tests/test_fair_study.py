"""Unit tests for Phase 2 fair cohort filtering, task-clustered bootstrapping, matching, and regressions."""

from pathlib import Path

import numpy as np

from agent_loops.fair_study import (
    assign_capability_band,
    compute_before_after_matching,
    compute_model_scaffold_table,
    compute_sensitivity_analyses,
    extract_fair_cohort_datasets,
    fit_cohort_and_pooled_regressions,
    run_fair_open_vs_closed_study,
)
from agent_loops.study import CompactRunRecord


def _rec(
    *,
    dataset: str,
    harness: str,
    cohort: str | None,
    model: str,
    task: str,
    run_id: str,
    succeeded: bool,
    n_steps: int,
    loop_steps: int,
    cost_usd: float = 0.10,
    loop_cost_usd: float = 0.02,
    litellm_llm: str | None = None,
) -> CompactRunRecord:
    has_loop = loop_steps > 0
    return CompactRunRecord(
        dataset=dataset,
        run_id=run_id,
        harness=harness,
        model=model,
        task_name=task,
        succeeded=succeeded,
        n_steps=n_steps,
        n_tool_steps=n_steps,
        has_any_loop=has_loop,
        has_rotation_loop=False,
        has_aba_thrash=False,
        has_exact_loop=has_loop,
        has_near_duplicate=False,
        has_error_streak=False,
        has_redundant_reread=False,
        has_productive_repeat=False,
        loop_steps=loop_steps,
        productive_repeat_steps=0,
        label_step_counts={},
        cost_included=n_steps > 0,
        drop_reason=None if n_steps > 0 else "empty_steps",
        mismatch_ratio=1.0,
        total_cost_usd=cost_usd,
        loop_cost_usd=loop_cost_usd if has_loop else 0.0,
        step_costs=np.zeros(max(0, n_steps), dtype=np.float64),
        stop_steps={},
        has_rsc_ref=False,
        graded=True,
        cost_source="reported_usd_rebuilt",
        litellm_llm=litellm_llm,
        cohort=cohort,
    )


def test_common_task_intersection_and_band_matching_simpson_check():
    # Construct 20 shared tasks + 1 unshared task in tb terminus-2.
    # Open has one low-pass model (10%, loops 15/100) and one high-pass model (60%, loops 3/100).
    # Closed has only a high-pass model (60%, loops 6/100).
    # Unmatched Open mean = (15 + 3)/2 = 9.0 > Closed 6.0 (+3.0).
    # Band-matched in B4 (55-75%) only: Open 3.0 - Closed 6.0 = -3.0 (reversal!).
    tb_recs = []
    for t_idx in range(20):
        tname = f"task_{t_idx:02d}"
        # Weak open model (gpt-oss-20b@together_ai): pass first 2 tasks -> 10%
        tb_recs.append(
            _rec(
                dataset="terminalbench",
                harness="terminus-2",
                cohort="tb_terminus_2",
                model="openai/gpt-oss-20b@together_ai",
                task=tname,
                run_id=f"w_{t_idx}",
                succeeded=(t_idx < 2),
                n_steps=20,
                loop_steps=3,  # 15 / 100 steps
            )
        )
        # Frontier open model (glm-5@z-ai): pass first 12 tasks -> 60%
        tb_recs.append(
            _rec(
                dataset="terminalbench",
                harness="terminus-2",
                cohort="tb_terminus_2",
                model="glm-5@z-ai",
                task=tname,
                run_id=f"fo_{t_idx}",
                succeeded=(t_idx < 12),
                n_steps=100,
                loop_steps=3,  # 3 / 100 steps
            )
        )
        # Frontier closed model (gemini-3-pro-preview@gemini): pass first 12 tasks -> 60%
        tb_recs.append(
            _rec(
                dataset="terminalbench",
                harness="terminus-2",
                cohort="tb_terminus_2",
                model="gemini-3-pro-preview@gemini",
                task=tname,
                run_id=f"fc_{t_idx}",
                succeeded=(t_idx < 12),
                n_steps=100,
                loop_steps=6,  # 6 / 100 steps
            )
        )
    # Extra unshared task on gemini only + custom harness run (must both be excluded!)
    tb_recs.append(
        _rec(
            dataset="terminalbench",
            harness="terminus-2",
            cohort="tb_terminus_2",
            model="gemini-3-pro-preview@gemini",
            task="unshared_extra",
            run_id="extra",
            succeeded=False,
            n_steps=50,
            loop_steps=40,
        )
    )
    tb_recs.append(
        _rec(
            dataset="terminalbench",
            harness="claude-code",
            cohort=None,
            model="claude-opus-4-5-20251101@anthropic",
            task="task_00",
            run_id="cc",
            succeeded=True,
            n_steps=10,
            loop_steps=0,
        )
    )

    cohorts, _ = extract_fair_cohort_datasets(
        tb_records=tb_recs,
        swe_records=[],
        longtail_records=[],
        sierra_records=[],
    )
    assert list(cohorts.keys()) == ["tb_terminus_2"]
    assert len(cohorts["tb_terminus_2"].common_tasks) == 20
    assert "unshared_extra" not in cohorts["tb_terminus_2"].common_tasks

    model_df = compute_model_scaffold_table(cohorts, n_boot=50, seed=42)
    assert len(model_df) == 3
    matching = compute_before_after_matching(cohorts, model_df, n_boot=50, seed=42)
    lp = matching["tb_terminus_2"]["loops_per_100_steps"]
    assert abs(lp["unmatched_diff"] - 3.0) < 1e-6
    assert lp["overlap_bands"] == 1
    assert abs(lp["matched_diff"] - (-3.0)) < 1e-6


def test_regressions_and_sensitivity_run_end_to_end(tmp_path: Path):
    swe_recs = []
    # 2 open models + 2 closed models across 16 tasks x 2 runs with non-zero variance
    models = [
        ("z-ai/glm-5.2", True, 0.50, 2),
        ("deepseek/deepseek-v4-pro", True, 0.25, 4),
        ("claude-opus-5", False, 0.50, 1),
        ("gpt-5.6-sol", False, 0.50, 1),
    ]
    for tidx in range(16):
        tname = f"swe_t_{tidx:02d}"
        for m, is_op, pr, lp in models:
            for r_i in range(2):
                swe_recs.append(
                    _rec(
                        dataset="swe_rebench",
                        harness=m.split("/")[-1],
                        cohort="swe_rebench_swelike",
                        model=m,
                        task=tname,
                        run_id=f"{m}_{tidx}_{r_i}",
                        succeeded=(tidx < int(16 * pr)),
                        n_steps=20,
                        loop_steps=lp if (tidx % 2 == 0 or is_op) else 0,
                    )
                )

    out = run_fair_open_vs_closed_study(
        results_dir=tmp_path,
        tb_records=[],
        swe_records=swe_recs,
        longtail_records=[],
        sierra_records=[],
    )
    assert out["csv_path"].exists()
    assert out["md_path"].exists()
    reg = out["regressions"]["swe_rebench_swelike"]
    assert reg["negbin_offset"]["converged"] is True
    assert reg["logit"]["converged"] is True
    assert reg["glmm_binomial_random_task"]["converged"] is True
    assert "per_family_df" in out["sensitivities"]
    assert assign_capability_band(0.30) == "B2 (25-40%)"
