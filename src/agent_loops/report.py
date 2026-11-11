"""Programmatic report generation from computed metrics (zero hand-typed numbers)."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent_loops.cost import REBUILD_MAX_RATIO, REBUILD_MIN_RATIO
from agent_loops.detectors import DetectorResult
from agent_loops.study import CompactRunRecord


def _corpus_headline(records: list[CompactRunRecord], drop_audit: dict[str, int]) -> dict[str, Any]:
    n = len(records)
    loop_runs = sum(1 for r in records if r.has_any_loop)
    loop_free = n - loop_runs
    succ_loop = sum(1 for r in records if r.has_any_loop and r.succeeded)
    succ_free = sum(1 for r in records if (not r.has_any_loop) and r.succeeded)

    tot_steps = sum(r.n_steps for r in records)
    lp_steps = sum(r.loop_steps for r in records)
    pr_steps = sum(r.productive_repeat_steps for r in records)

    cost_recs = [r for r in records if r.cost_included]
    tot_usd = sum(r.total_cost_usd for r in cost_recs)
    lp_usd = sum(r.loop_cost_usd for r in cost_recs)

    rot_runs = sum(1 for r in records if r.has_rotation_loop)
    aba_runs = sum(1 for r in records if r.has_aba_thrash)
    exact_runs = sum(1 for r in records if r.has_exact_loop)
    nd_runs = sum(1 for r in records if r.has_near_duplicate)
    err_runs = sum(1 for r in records if r.has_error_streak)
    reread_runs = sum(1 for r in records if r.has_redundant_reread)
    prod_runs = sum(1 for r in records if r.has_productive_repeat)

    # Check how many rotation runs Cline-style misses
    rot_missed_by_cline = sum(
        1 for r in records if r.has_rotation_loop and r.stop_steps.get("cline_style") is None
    )

    return {
        "runs_analyzed": n,
        "loop_runs": loop_runs,
        "loop_run_rate": loop_runs / n if n else 0.0,
        "total_steps": tot_steps,
        "loop_steps": lp_steps,
        "loop_step_share": lp_steps / tot_steps if tot_steps else 0.0,
        "productive_repeat_step_share": pr_steps / tot_steps if tot_steps else 0.0,
        "cost_eval_runs": len(cost_recs),
        "total_cost_usd": tot_usd,
        "loop_cost_usd": lp_usd,
        "loop_dollar_share": lp_usd / tot_usd if tot_usd > 0 else 0.0,
        "overall_pass_rate": (succ_loop + succ_free) / n if n else 0.0,
        "looping_pass_rate": succ_loop / loop_runs if loop_runs else 0.0,
        "loop_free_pass_rate": succ_free / loop_free if loop_free else 0.0,
        "exact_loop_runs": exact_runs,
        "near_duplicate_runs": nd_runs,
        "aba_thrash_runs": aba_runs,
        "rotation_runs": rot_runs,
        "rotation_missed_by_cline": rot_missed_by_cline,
        "rotation_missed_by_cline_pct": (rot_missed_by_cline / rot_runs) if rot_runs else 0.0,
        "error_streak_runs": err_runs,
        "redundant_reread_runs": reread_runs,
        "productive_repeat_runs": prod_runs,
        "drop_audit": drop_audit,
    }


def _fmt_pct(val: float) -> str:
    return f"{val * 100.0:.2f}%"


def _detector_table_md(df: pd.DataFrame) -> str:
    lines = [
        "| Detector | Firing Rate | Dollars Saved | False-Stop Rate | Matched-Rand FSR (95% CI) | Matched-Rand Saved (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['name']}` | {_fmt_pct(r['firing_rate'])} ({int(r['fired_runs'])}) | "
            f"{_fmt_pct(r['dollars_saved_pct'])} (${r['dollars_saved_usd']:,.2f}) | "
            f"{_fmt_pct(r['false_stop_rate'])} | "
            f"{_fmt_pct(r['rand_false_stop_mean'])} [{_fmt_pct(r['rand_false_stop_ci_low'])}, {_fmt_pct(r['rand_false_stop_ci_high'])}] | "
            f"{_fmt_pct(r['rand_dollars_saved_pct_mean'])} [{_fmt_pct(r['rand_dollars_saved_pct_ci_low'])}, {_fmt_pct(r['rand_dollars_saved_pct_ci_high'])}] |"
        )
    return "\n".join(lines)


def write_results_and_summary(
    *,
    results_dir: Path,
    tb_records: list[CompactRunRecord],
    tb_drops: dict[str, int],
    tb_by_model: pd.DataFrame,
    tb_by_harness: pd.DataFrame,
    tb_detectors: list[DetectorResult],
    tb_budgets: dict[str, int],
    swe_records: list[CompactRunRecord],
    swe_drops: dict[str, int],
    swe_by_model: pd.DataFrame,
    swe_by_harness: pd.DataFrame,
    swe_detectors: list[DetectorResult],
    swe_budgets: dict[str, int],
    tau2_summary: dict[str, Any],
    loop_prone_tasks: pd.DataFrame,
    timings_sec: dict[str, float],
    longtail_counts: dict[str, int] | None = None,
    sierra_counts: dict[str, int] | None = None,
    fair_study_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write all CSV/JSON artifacts and programmatically generate results/summary.md."""
    results_dir.mkdir(parents=True, exist_ok=True)

    tb_head = _corpus_headline(tb_records, tb_drops)
    swe_head = _corpus_headline(swe_records, swe_drops)

    tb_det_df = pd.DataFrame([asdict(d) for d in tb_detectors])
    swe_det_df = pd.DataFrame([asdict(d) for d in swe_detectors])

    tb_by_model.to_csv(results_dir / "tb_by_model.csv", index=False)
    tb_by_harness.to_csv(results_dir / "tb_by_harness.csv", index=False)
    tb_det_df.to_csv(results_dir / "tb_detectors.csv", index=False)

    swe_by_model.to_csv(results_dir / "swe_by_model.csv", index=False)
    swe_by_harness.to_csv(results_dir / "swe_by_harness.csv", index=False)
    swe_det_df.to_csv(results_dir / "swe_detectors.csv", index=False)

    loop_prone_tasks.to_csv(results_dir / "loop_prone_tasks.csv", index=False)

    # Also compute clean sensitivity slice on Terminal-Bench trajectories without RSC $hex refs
    tb_no_rsc = [r for r in tb_records if not r.has_rsc_ref]
    tb_no_rsc_head = _corpus_headline(tb_no_rsc, {"note": "zero-RSC-pointer subset"})

    phase2_meta: dict[str, Any] = {}
    if fair_study_results is not None:
        mres = fair_study_results.get("matching", {})
        reg = fair_study_results.get("regressions", {})
        phase2_meta = {
            "longtail_counts": longtail_counts or {},
            "sierra_counts": sierra_counts or {},
            "cohorts": {
                c: {
                    "n_open_models": mres[c]["n_open_models"],
                    "n_closed_models": mres[c]["n_closed_models"],
                    "n_open_runs": mres[c]["n_open_runs"],
                    "n_closed_runs": mres[c]["n_closed_runs"],
                    "unmatched_loops_per_100_diff": mres[c]["loops_per_100_steps"]["unmatched_diff"],
                    "matched_loops_per_100_diff": mres[c]["loops_per_100_steps"]["matched_diff"],
                    "matched_loops_per_100_ci": mres[c]["loops_per_100_steps"]["matched_diff_ci"],
                    "overlap_bands": mres[c]["loops_per_100_steps"]["overlap_bands"],
                    "matched_any_loop_diff": mres[c]["any_loop_rate"]["matched_diff"],
                }
                for c in mres
            },
            "pooled_within_scaffold": {
                k: v
                for k, v in (reg.get("pooled_within_scaffold") or {}).items()
                if k in ("negbin_offset", "logit", "logit_with_steps", "gee_poisson_task", "glmm_binomial_random_task")
            },
        }

    metrics = {
        "pre_registered_rebuild_ratio_bounds": [REBUILD_MIN_RATIO, REBUILD_MAX_RATIO],
        "terminalbench": tb_head,
        "terminalbench_no_rsc_subset": tb_no_rsc_head,
        "terminalbench_turn_budgets": tb_budgets,
        "swe_rebench": swe_head,
        "swe_rebench_turn_budgets": swe_budgets,
        "tau2_bench": tau2_summary,
        "phase2_open_vs_closed": phase2_meta,
        "timings_sec": timings_sec,
    }
    (results_dir / "study_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Build summary.md purely from computed numbers
    md: list[str] = []
    md.append("# Agent Loops Study — Summary of Computed Findings\n")
    md.append(
        "All numbers below are generated programmatically from cached public trajectory datasets "
        "(CPU-only offline pipeline, fixed seed `42`, 200 matched-random bootstrap seeds).\n"
    )

    md.append("## 1. Headline Prevalence, Cost Share, and Pass Rates\n")
    md.append("| Metric | Terminal-Bench 2.0 | TB Clean-Trace Subset (0 RSC refs) | SWE-rebench July 2026 |")
    md.append("| --- | ---: | ---: | ---: |")
    md.append(
        f"| Trajectories analyzed (with steps) | {tb_head['runs_analyzed']:,} | "
        f"{tb_no_rsc_head['runs_analyzed']:,} | {swe_head['runs_analyzed']:,} |"
    )
    md.append(
        f"| Runs with any loop | {tb_head['loop_runs']:,} ({_fmt_pct(tb_head['loop_run_rate'])}) | "
        f"{tb_no_rsc_head['loop_runs']:,} ({_fmt_pct(tb_no_rsc_head['loop_run_rate'])}) | "
        f"{swe_head['loop_runs']:,} ({_fmt_pct(swe_head['loop_run_rate'])}) |"
    )
    md.append(
        f"| Loop share of steps | {_fmt_pct(tb_head['loop_step_share'])} ({tb_head['loop_steps']:,}/{tb_head['total_steps']:,}) | "
        f"{_fmt_pct(tb_no_rsc_head['loop_step_share'])} | "
        f"{_fmt_pct(swe_head['loop_step_share'])} ({swe_head['loop_steps']:,}/{swe_head['total_steps']:,}) |"
    )
    md.append(
        f"| Cost-validated runs | {tb_head['cost_eval_runs']:,} | "
        f"{tb_no_rsc_head['cost_eval_runs']:,} | {swe_head['cost_eval_runs']:,} |"
    )
    md.append(
        f"| Loop share of cache-aware dollars | {_fmt_pct(tb_head['loop_dollar_share'])} (${tb_head['loop_cost_usd']:,.2f} / ${tb_head['total_cost_usd']:,.2f}) | "
        f"{_fmt_pct(tb_no_rsc_head['loop_dollar_share'])} | "
        f"{_fmt_pct(swe_head['loop_dollar_share'])} (${swe_head['loop_cost_usd']:,.2f} / ${swe_head['total_cost_usd']:,.2f}) |"
    )
    md.append(
        f"| Pass rate: looping runs | {_fmt_pct(tb_head['looping_pass_rate'])} | "
        f"{_fmt_pct(tb_no_rsc_head['looping_pass_rate'])} | {_fmt_pct(swe_head['looping_pass_rate'])} |"
    )
    md.append(
        f"| Pass rate: loop-free runs | {_fmt_pct(tb_head['loop_free_pass_rate'])} | "
        f"{_fmt_pct(tb_no_rsc_head['loop_free_pass_rate'])} | {_fmt_pct(swe_head['loop_free_pass_rate'])} |"
    )
    md.append(
        f"| Productive repeat step share (excluded from loops) | {_fmt_pct(tb_head['productive_repeat_step_share'])} | "
        f"{_fmt_pct(tb_no_rsc_head['productive_repeat_step_share'])} | {_fmt_pct(swe_head['productive_repeat_step_share'])} |"
    )
    md.append(
        f"| Period-k rotation runs missed by Cline-5 | "
        f"{tb_head['rotation_missed_by_cline']:,}/{tb_head['rotation_runs']:,} ({_fmt_pct(tb_head['rotation_missed_by_cline_pct'])}) | "
        f"{tb_no_rsc_head['rotation_missed_by_cline']:,}/{tb_no_rsc_head['rotation_runs']:,} ({_fmt_pct(tb_no_rsc_head['rotation_missed_by_cline_pct'])}) | "
        f"{swe_head['rotation_missed_by_cline']:,}/{swe_head['rotation_runs']:,} ({_fmt_pct(swe_head['rotation_missed_by_cline_pct'])}) |\n"
    )

    md.append("## 2. Loop Taxonomy Breakdown (Runs Exhibiting Each Pattern)\n")
    md.append("| Pattern | Terminal-Bench 2.0 Runs | SWE-rebench Runs |")
    md.append("| --- | ---: | ---: |")
    for label_key, disp in [
        ("exact_loop_runs", "Exact loop (same call + same obs)"),
        ("near_duplicate_runs", "Near-duplicate"),
        ("aba_thrash_runs", "A-B-A thrash"),
        ("rotation_runs", "Period-k rotation (k <= 5)"),
        ("error_streak_runs", "Error streak"),
        ("redundant_reread_runs", "Redundant re-read (no write in between)"),
        ("productive_repeat_runs", "Productive repeat (same call, new obs — non-loop)"),
    ]:
        md.append(
            f"| {disp} | {tb_head[label_key]:,} ({_fmt_pct(tb_head[label_key] / tb_head['runs_analyzed'])}) | "
            f"{swe_head[label_key]:,} ({_fmt_pct(swe_head[label_key] / swe_head['runs_analyzed'])}) |"
        )
    md.append("")

    md.append("## 3. Cost Rebuild Drop Audit (Pre-Registered Margin `[0.50x, 2.00x]`)\n")
    md.append("### Terminal-Bench 2.0")
    for k, v in sorted(tb_drops.items()):
        md.append(f"- `{k}`: {v:,}")
    md.append("\n### SWE-rebench July 2026")
    for k, v in sorted(swe_drops.items()):
        md.append(f"- `{k}`: {v:,}")
    md.append("")

    md.append("## 4. Detector Replay vs Matched-Random Baseline (200 Seeds)\n")
    md.append("### Terminal-Bench 2.0")
    md.append(_detector_table_md(tb_det_df))
    md.append("\n### SWE-rebench July 2026")
    md.append(_detector_table_md(swe_det_df))
    md.append("")

    md.append("## 5. Per-Harness Summary\n")
    md.append("### Terminal-Bench 2.0")
    md.append("| Harness | Runs | Loop Runs (%) | Loop Step Share | Loop Dollar Share | Pass (Loop vs Clean) | 1.25x P90 Budget |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in tb_by_harness.sort_values("loop_run_rate", ascending=False).iterrows():
        h = r["harness"]
        c_runs = int(r.get("cost_runs", 0))
        if c_runs == 0 or pd.isna(r["loop_dollar_share"]):
            dsh = "n/a"
        elif c_runs < 100:
            dsh = "n<100"
        else:
            dsh = _fmt_pct(r["loop_dollar_share"])
        lp_p = _fmt_pct(r["looping_pass_rate"]) if pd.notna(r["looping_pass_rate"]) else "n/a"
        fr_p = _fmt_pct(r["loop_free_pass_rate"]) if pd.notna(r["loop_free_pass_rate"]) else "n/a"
        md.append(
            f"| `{h}` | {int(r['runs']):,} | {_fmt_pct(r['loop_run_rate'])} | "
            f"{_fmt_pct(r['loop_step_share'])} | {dsh} | {lp_p} vs {fr_p} | {tb_budgets.get(h, 'n/a')} |"
        )

    md.append("\n### SWE-rebench July 2026")
    md.append("| Participant | Runs | Loop Runs (%) | Loop Step Share | Loop Dollar Share | Pass (Loop vs Clean) | 1.25x P90 Budget |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in swe_by_harness.sort_values("loop_run_rate", ascending=False).iterrows():
        h = r["harness"]
        c_runs = int(r.get("cost_runs", 0))
        if c_runs == 0 or pd.isna(r["loop_dollar_share"]):
            dsh = "n/a"
        elif c_runs < 100:
            dsh = "n<100"
        else:
            dsh = _fmt_pct(r["loop_dollar_share"])
        lp_p = _fmt_pct(r["looping_pass_rate"]) if pd.notna(r["looping_pass_rate"]) else "n/a"
        fr_p = _fmt_pct(r["loop_free_pass_rate"]) if pd.notna(r["loop_free_pass_rate"]) else "n/a"
        md.append(
            f"| `{h}` | {int(r['runs']):,} | {_fmt_pct(r['loop_run_rate'])} | "
            f"{_fmt_pct(r['loop_step_share'])} | {dsh} | {lp_p} vs {fr_p} | {swe_budgets.get(h, 'n/a')} |"
        )
    md.append("")

    md.append("## 6. Top 10 Loop-Prone Tasks for Gemini-Family Models (full 20 in `results/loop_prone_tasks.csv`)\n")
    md.append("| Rank | Task | Gemini Runs | Loop Rate | Loop Step Share | Gemini Pass Rate |")
    md.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for _, r in loop_prone_tasks.head(10).iterrows():
        md.append(
            f"| {int(r['rank'])} | `{r['task_name']}` | {int(r['gemini_runs'])} | "
            f"{_fmt_pct(r['gemini_loop_run_rate'])} | {_fmt_pct(r['gemini_loop_step_share'])} | "
            f"{_fmt_pct(r['gemini_pass_rate'])} |"
        )
    md.append("")

    md.append("## 7. τ²-bench Optional Sanity Check\n")
    md.append(
        f"- Simulations analyzed: **{tau2_summary.get('simulations', 0):,}** across **{tau2_summary.get('files', 0)}** result files "
        f"({tau2_summary.get('total_tool_steps', 0):,} tool calls)."
    )
    md.append(
        f"- Simulations with any pathological loop: **{tau2_summary.get('any_loop_sims', 0):,} "
        f"({_fmt_pct(tau2_summary.get('any_loop_rate', 0.0))})**; loop step share: "
        f"**{_fmt_pct(tau2_summary.get('loop_step_share', 0.0))}**."
    )
    md.append(
        f"- Breakdown: exact_loop={tau2_summary.get('exact_loop_sims', 0)}, "
        f"near_duplicate={tau2_summary.get('near_duplicate_sims', 0)}, "
        f"aba_thrash={tau2_summary.get('aba_thrash_sims', 0)}, "
        f"period_k_rotation={tau2_summary.get('period_k_rotation_sims', 0)}, "
        f"error_streak={tau2_summary.get('error_streak_sims', 0)}, "
        f"redundant_reread={tau2_summary.get('redundant_reread_sims', 0)}; "
        f"productive_repeat={tau2_summary.get('productive_repeat_sims', 0)}."
    )
    md.append(
        f"- Pass rate: looping {_fmt_pct(tau2_summary.get('pass_rate_looping', 0.0))} vs "
        f"loop-free {_fmt_pct(tau2_summary.get('pass_rate_loop_free', 0.0))}.\n"
    )

    if fair_study_results is not None:
        mres = fair_study_results.get("matching", {})
        reg = fair_study_results.get("regressions", {})
        md.append("## 8. Phase 2 Fair Open-Weights vs Closed Comparison (Within-Scaffold, Common Tasks)\n")
        md.append(
            "See [`results/open_vs_closed_model.md`](open_vs_closed_model.md), "
            "[`results/loops_by_model_scaffold.csv`](loops_by_model_scaffold.csv), and `../ARTICLE_DRAFT.md`.\n"
        )
        md.append(
            "| Cohort | Unmatched Gap Loops/100 [95% CI] | Overlap Bands | Matched Gap Loops/100 [95% CI] | NB Offset Adj IRR (p_2way) |"
        )
        md.append("| --- | ---: | ---: | ---: | ---: |")
        for cname, entry in mres.items():
            lp = entry["loops_per_100_steps"]
            u_s = f"{lp['unmatched_diff']:+.2f} [{lp['unmatched_diff_ci'][0]:+.2f}, {lp['unmatched_diff_ci'][1]:+.2f}]"
            if lp["overlap_bands"] > 0 and pd.notna(lp["matched_diff"]):
                m_s = f"{lp['matched_diff']:+.2f} [{lp['matched_diff_ci'][0]:+.2f}, {lp['matched_diff_ci'][1]:+.2f}]"
            else:
                m_s = "no band overlap"
            nb_c = ((reg.get(cname) or {}).get("negbin_offset") or {}).get("coefs", {}).get("is_open_weight")
            nb_s = f"{nb_c['irr_or']:.2f} ({nb_c['p']:.4f})" if nb_c else "n/a"
            md.append(f"| `{cname}` | {u_s} | {lp['overlap_bands']} | {m_s} | {nb_s} |")
        pool_nb = ((reg.get("pooled_within_scaffold") or {}).get("negbin_offset") or {}).get("coefs", {}).get("is_open_weight")
        pool_lg = ((reg.get("pooled_within_scaffold") or {}).get("logit") or {}).get("coefs", {}).get("is_open_weight")
        if pool_nb and pool_lg:
            md.append(
                f"\n- **Pooled within-scaffold regression**: `is_open_weight` NB `log(steps)`-offset IRR = "
                f"**{pool_nb['irr_or']:.2f}** (`[{pool_nb['ci_low']:.2f}, {pool_nb['ci_high']:.2f}]`, `p={pool_nb['p']:.4f}`); "
                f"Logit OR = **{pool_lg['irr_or']:.2f}** (`[{pool_lg['ci_low']:.2f}, {pool_lg['ci_high']:.2f}]`, `p={pool_lg['p']:.4f}`).\n"
            )

    summary_text = "\n".join(md)
    (results_dir / "summary.md").write_text(summary_text)
    return metrics
