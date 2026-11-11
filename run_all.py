"""Single-command reproducible entrypoint for the entire agent-loops study (Phases 1, 2, and 3)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import pandas as pd

from agent_loops.article import build_article_draft, update_notes_with_phase3
from agent_loops.download import download_all
from agent_loops.extra_reports import generate_extra_reports
from agent_loops.fair_study import run_fair_open_vs_closed_study
from agent_loops.phase3 import load_or_analyze_dataset, run_phase3
from agent_loops.plots import (
    plot_cohort_verdicts,
    plot_detector_savings_vs_false_stop,
    plot_detectors_vs_random_16x9,
    plot_glm52_vs_opus5,
    plot_harness_dumbbell,
    plot_hero_raw_vs_controlled,
    plot_loop_prevalence_by_harness,
    plot_loops_per_100_by_model_band,
    plot_open_closed_gap_before_after_matching,
    plot_open_vs_closed_across_four_datasets,
    plot_step_vs_dollar_share,
)
from agent_loops.report import write_results_and_summary
from agent_loops.study import (
    aggregate_group_metrics,
    analyze_longtail_swe,
    analyze_sierra_tau2,
    analyze_swe_rebench,
    analyze_tau2,
    analyze_terminalbench,
    evaluate_all_detectors,
    select_gemini_loop_prone_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Agent Loops study (Phases 1, 2, and 3).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AGENT_LOOPS_DATA_DIR", Path(__file__).resolve().parent / "data")),
        help="Directory for downloaded raw datasets (default: ./data or $AGENT_LOOPS_DATA_DIR).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ["AGENT_LOOPS_CACHE_DIR"]) if os.environ.get("AGENT_LOOPS_CACHE_DIR") else None,
        help="Optional directory for caching intermediate CompactRunRecord Parquet files (e.g., /content/cache_parquet).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_dir: Path = args.data_dir.resolve()
    cache_dir: Path | None = args.cache_dir.resolve() if args.cache_dir else None
    results_dir = root / "results"
    assets_dir = root / "assets"
    audit_dir = root / "audit"

    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    # 1. Download (cached locally) and print schemas across all 4 datasets
    t0 = time.perf_counter()
    download_all(data_dir=data_dir, include_tau2=True)
    timings["download_and_schema_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 1] download/schema check done in {timings['download_and_schema_s']:.2f}s (data_dir={data_dir})", flush=True)

    # 2. Terminal-Bench 2.0 analysis (with optional Parquet resume cache)
    t0 = time.perf_counter()
    if cache_dir is not None:
        tb_records, tb_drops = load_or_analyze_dataset("terminalbench", cache_dir, analyze_terminalbench, data_dir)
    else:
        tb_records, tb_drops = analyze_terminalbench(data_dir)
    tb_by_model = aggregate_group_metrics(tb_records, "model")
    tb_by_harness = aggregate_group_metrics(tb_records, "harness")
    tb_detectors, tb_budgets = evaluate_all_detectors(tb_records, n_seeds=200, seed=42)
    loop_prone_tasks = select_gemini_loop_prone_tasks(tb_records, top_n=20)
    timings["terminalbench_analysis_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 2] Terminal-Bench analyzed ({len(tb_records):,} runs) in {timings['terminalbench_analysis_s']:.2f}s", flush=True)

    # 3. SWE-rebench July 2026 analysis (with optional Parquet resume cache)
    t0 = time.perf_counter()
    if cache_dir is not None:
        swe_records, swe_drops = load_or_analyze_dataset("swe_rebench", cache_dir, analyze_swe_rebench, data_dir)
    else:
        swe_records, swe_drops = analyze_swe_rebench(data_dir)
    swe_by_model = aggregate_group_metrics(swe_records, "model")
    swe_by_harness = aggregate_group_metrics(swe_records, "harness")
    swe_detectors, swe_budgets = evaluate_all_detectors(swe_records, n_seeds=200, seed=42)
    timings["swe_rebench_analysis_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 3] SWE-rebench analyzed ({len(swe_records):,} runs) in {timings['swe_rebench_analysis_s']:.2f}s", flush=True)

    # 4. Legacy tau2-bench sanity check + Longtail-SWE + Sierra public S3 archive
    t0 = time.perf_counter()
    tau2_summary = analyze_tau2(data_dir)
    if cache_dir is not None:
        longtail_records, longtail_counts = load_or_analyze_dataset("longtail_swe", cache_dir, analyze_longtail_swe, data_dir)
        sierra_records, sierra_counts = load_or_analyze_dataset("sierra_tau2", cache_dir, analyze_sierra_tau2, data_dir)
    else:
        longtail_records, longtail_counts = analyze_longtail_swe(data_dir)
        sierra_records, sierra_counts = analyze_sierra_tau2(data_dir)
    timings["longtail_and_sierra_s"] = round(time.perf_counter() - t0, 2)
    print(
        f"[stage 4] Longtail ({len(longtail_records):,}) + Sierra S3 ({len(sierra_records):,}) "
        f"+ tau2 sanity ({tau2_summary.get('simulations', 0):,}) in {timings['longtail_and_sierra_s']:.2f}s",
        flush=True,
    )

    # 5. Fair Open-vs-Closed statistical comparison & regressions
    t0 = time.perf_counter()
    fair_res = run_fair_open_vs_closed_study(
        results_dir=results_dir,
        tb_records=tb_records,
        swe_records=swe_records,
        longtail_records=longtail_records,
        sierra_records=sierra_records,
    )
    timings["fair_open_vs_closed_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 5] Fair Open-vs-Closed statistical study done in {timings['fair_open_vs_closed_s']:.2f}s", flush=True)

    # 6. Phase 1 & 2 Reports + Extra Reports
    t0 = time.perf_counter()
    metrics = write_results_and_summary(
        results_dir=results_dir,
        tb_records=tb_records,
        tb_drops=tb_drops,
        tb_by_model=tb_by_model,
        tb_by_harness=tb_by_harness,
        tb_detectors=tb_detectors,
        tb_budgets=tb_budgets,
        swe_records=swe_records,
        swe_drops=swe_drops,
        swe_by_model=swe_by_model,
        swe_by_harness=swe_by_harness,
        swe_detectors=swe_detectors,
        swe_budgets=swe_budgets,
        tau2_summary=tau2_summary,
        loop_prone_tasks=loop_prone_tasks,
        timings_sec=timings,
        longtail_counts=longtail_counts,
        sierra_counts=sierra_counts,
        fair_study_results=fair_res,
    )

    generate_extra_reports(
        data_dir=data_dir,
        results_dir=results_dir,
        tb_records=tb_records,
        swe_records=swe_records,
        tau2_summary=tau2_summary,
    )

    # 7. Phase 3: Labeler Precision Audit, Run-Length Control, Two-Stage Bootstrap, Verdict Rule, Same-Model-Across-Harness & Cost Rebuild Audit
    t_p3 = time.perf_counter()
    phase3_res = run_phase3(
        data_dir=data_dir,
        results_dir=results_dir,
        audit_dir=audit_dir,
        tb_records=tb_records,
        tb_drops=tb_drops,
        swe_records=swe_records,
        fair_res=fair_res,
    )
    timings["phase3_audit_and_bootstraps_s"] = round(time.perf_counter() - t_p3, 2)
    print(f"[stage 6] Phase 3 robustness & precision audit done in {timings['phase3_audit_and_bootstraps_s']:.2f}s", flush=True)

    # 8. All Plots (6 Phase 1+2 plots + 5 new 1600x900 px Phase 3 figures)
    tb_det_df = pd.DataFrame([asdict(d) for d in tb_detectors])
    swe_det_df = pd.DataFrame([asdict(d) for d in swe_detectors])

    plot_step_vs_dollar_share(
        tb_by_model,
        swe_by_model,
        assets_dir / "step_vs_dollar_share_per_model.png",
    )
    plot_detector_savings_vs_false_stop(
        tb_det_df,
        swe_det_df,
        assets_dir / "detector_savings_vs_false_stop.png",
    )
    plot_loop_prevalence_by_harness(
        tb_by_harness,
        swe_by_harness,
        assets_dir / "loop_prevalence_by_harness.png",
    )
    plot_loops_per_100_by_model_band(
        fair_res["model_df"],
        assets_dir / "loops_per_100_by_model_band.png",
    )
    plot_open_closed_gap_before_after_matching(
        fair_res["matching"],
        assets_dir / "open_closed_gap_before_after_matching.png",
    )
    plot_open_vs_closed_across_four_datasets(
        fair_res["model_df"],
        fair_res["regressions"],
        assets_dir / "open_vs_closed_across_four_datasets.png",
    )

    # 5 New 1600x900 px Phase 3 Article Figures
    plot_hero_raw_vs_controlled(
        phase3_res["length_df"],
        phase3_res["boot_comp_df"],
        phase3_res["strict_comp_df"],
        fair_res["regressions"],
        assets_dir / "hero_raw_vs_controlled.png",
    )
    plot_cohort_verdicts(
        phase3_res["verdicts_df"],
        assets_dir / "cohort_verdicts.png",
    )
    plot_detectors_vs_random_16x9(
        tb_det_df,
        swe_det_df,
        assets_dir / "detectors_vs_random.png",
    )
    plot_glm52_vs_opus5(
        fair_res["model_df"],
        phase3_res["strict_by_model_df"],
        swe_by_model,
        assets_dir / "glm52_vs_opus5.png",
    )
    plot_harness_dumbbell(
        phase3_res["same_model_df"],
        assets_dir / "harness_dumbbell.png",
    )

    # 9. Update ARTICLE_DRAFT.md (1,200-1,500 words) and NOTES.md
    article_md = build_article_draft(
        study_metrics=metrics,
        fair_res=fair_res,
        phase3_res=phase3_res,
    )
    (root / "ARTICLE_DRAFT.md").write_text(article_md, encoding="utf-8")
    update_notes_with_phase3(root / "NOTES.md", phase3_res)

    timings["plots_and_report_s"] = round(time.perf_counter() - t0, 2)
    timings["total_s"] = round(time.perf_counter() - t_start, 2)
    metrics["timings_sec"] = timings
    metrics["phase3_summary"] = {
        "tb_cost_audit": phase3_res["tb_cost_audit_summary"],
        "tau2_classic_irr_diagnosis": phase3_res["tau2_diag"],
        "article_word_count": len(article_md.split()),
    }
    (results_dir / "study_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print("\n" + "=" * 72)
    print("HEADLINE SUMMARY")
    print("=" * 72)
    print((results_dir / "summary.md").read_text())
    print(f"\nARTICLE_DRAFT.md word count: {len(article_md.split())} words.")
    print(f"Completed full reproducible study (Phases 1-3) in {timings['total_s']:.2f}s.")


if __name__ == "__main__":
    main()
