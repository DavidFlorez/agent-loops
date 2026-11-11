"""Smoke tests for Phase 2 publication plots and programmatic ARTICLE_DRAFT generator."""

from pathlib import Path

import numpy as np

from agent_loops.article import build_article_draft
from agent_loops.fair_study import run_fair_open_vs_closed_study
from agent_loops.plots import (
    plot_loops_per_100_by_model_band,
    plot_open_closed_gap_before_after_matching,
    plot_open_vs_closed_across_four_datasets,
)
from agent_loops.study import CompactRunRecord


def _rec(
    *,
    dataset: str,
    harness: str,
    cohort: str,
    model: str,
    task: str,
    run_id: str,
    succeeded: bool,
    n_steps: int,
    loop_steps: int,
    litellm_llm: str | None = None,
) -> CompactRunRecord:
    return CompactRunRecord(
        dataset=dataset,
        run_id=run_id,
        harness=harness,
        model=model,
        task_name=task,
        succeeded=succeeded,
        n_steps=n_steps,
        n_tool_steps=n_steps,
        has_any_loop=loop_steps > 0,
        has_rotation_loop=False,
        has_aba_thrash=False,
        has_exact_loop=loop_steps > 0,
        has_near_duplicate=False,
        has_error_streak=False,
        has_redundant_reread=False,
        has_productive_repeat=False,
        loop_steps=loop_steps,
        productive_repeat_steps=0,
        label_step_counts={},
        cost_included=True,
        drop_reason=None,
        mismatch_ratio=1.0,
        total_cost_usd=0.25,
        loop_cost_usd=0.05 if loop_steps > 0 else 0.0,
        step_costs=np.zeros(n_steps, dtype=np.float64),
        stop_steps={},
        has_rsc_ref=False,
        graded=True,
        cost_source="reported_usd_rebuilt",
        litellm_llm=litellm_llm,
        cohort=cohort,
    )


def test_phase2_plots_and_article_generation(tmp_path: Path):
    recs_tb = []
    recs_bk = []
    for tidx in range(12):
        tname = f"task_{tidx:02d}"
        for m, lp in [
            ("glm-5@z-ai", 2),
            ("openai/gpt-oss-120b@together_ai", 6),
            ("gemini-3-pro-preview@gemini", 1 if tidx % 2 == 0 else 3),
            ("claude-opus-4-6@anthropic", 1),
        ]:
            recs_tb.append(
                _rec(
                    dataset="terminalbench",
                    harness="terminus-2",
                    cohort="tb_terminus_2",
                    model=m,
                    task=tname,
                    run_id=f"{m}_{tidx}",
                    succeeded=(tidx < 7 if "oss" not in m else tidx < 2),
                    n_steps=25 + tidx,
                    loop_steps=lp,
                )
            )
        for m, lp in [
            ("kimi-k3", 1),
            ("gemini-3-1-pro-preview", 4),
            ("gpt-5-4", 2),
        ]:
            recs_bk.append(
                _rec(
                    dataset="sierra_tau2",
                    harness="tau2_bk_alltools",
                    cohort="tau2_banking_knowledge_alltools",
                    model=m,
                    task=tname,
                    run_id=f"bk_{m}_{tidx}",
                    succeeded=(tidx < 4),
                    n_steps=20 + tidx,
                    loop_steps=lp,
                    litellm_llm="openrouter/moonshotai/kimi-k3" if m == "kimi-k3" else "vertex_ai/gemini-3.1-pro-preview",
                )
            )

    fair_res = run_fair_open_vs_closed_study(
        results_dir=tmp_path / "results",
        tb_records=recs_tb,
        swe_records=[],
        longtail_records=[],
        sierra_records=recs_bk,
    )

    p1 = tmp_path / "assets" / "loops_per_100_by_model_band.png"
    p2 = tmp_path / "assets" / "open_closed_gap_before_after_matching.png"
    p3 = tmp_path / "assets" / "open_vs_closed_across_four_datasets.png"
    plot_loops_per_100_by_model_band(fair_res["model_df"], p1)
    plot_open_closed_gap_before_after_matching(fair_res["matching"], p2)
    plot_open_vs_closed_across_four_datasets(fair_res["model_df"], fair_res["regressions"], p3)

    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 5000

    draft = build_article_draft(
        study_metrics={"terminalbench": {"loop_dollar_share": 0.177, "rotation_missed_by_cline": 10, "rotation_runs": 12}},
        fair_res=fair_res,
    )
    assert "# Do open-weight agents loop more?" in draft
    assert "Google Gemini" in draft
