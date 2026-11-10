"""Single-command reproducible entrypoint for the entire agent-loops study."""

from __future__ import annotations

from pathlib import Path
import time

from agent_loops.download import download_all
from agent_loops.plots import (
    plot_detector_savings_vs_false_stop,
    plot_loop_prevalence_by_harness,
    plot_step_vs_dollar_share,
)
from agent_loops.report import write_results_and_summary
from agent_loops.study import (
    aggregate_group_metrics,
    analyze_swe_rebench,
    analyze_tau2,
    analyze_terminalbench,
    evaluate_all_detectors,
    select_gemini_loop_prone_tasks,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    results_dir = root / "results"
    assets_dir = root / "assets"

    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    # 1. Download (cached locally) and print schemas
    t0 = time.perf_counter()
    download_all(data_dir=data_dir, include_tau2=True)
    timings["download_and_schema_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 1] download/schema check done in {timings['download_and_schema_s']:.2f}s", flush=True)

    # 2. Terminal-Bench 2.0 analysis
    t0 = time.perf_counter()
    tb_records, tb_drops = analyze_terminalbench(data_dir)
    tb_by_model = aggregate_group_metrics(tb_records, "model")
    tb_by_harness = aggregate_group_metrics(tb_records, "harness")
    tb_detectors, tb_budgets = evaluate_all_detectors(tb_records, n_seeds=200, seed=42)
    loop_prone_tasks = select_gemini_loop_prone_tasks(tb_records, top_n=20)
    timings["terminalbench_analysis_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 2] Terminal-Bench analyzed ({len(tb_records):,} runs) in {timings['terminalbench_analysis_s']:.2f}s", flush=True)

    # 3. SWE-rebench July 2026 analysis
    t0 = time.perf_counter()
    swe_records, swe_drops = analyze_swe_rebench(data_dir)
    swe_by_model = aggregate_group_metrics(swe_records, "model")
    swe_by_harness = aggregate_group_metrics(swe_records, "harness")
    swe_detectors, swe_budgets = evaluate_all_detectors(swe_records, n_seeds=200, seed=42)
    timings["swe_rebench_analysis_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 3] SWE-rebench analyzed ({len(swe_records):,} runs) in {timings['swe_rebench_analysis_s']:.2f}s", flush=True)

    # 4. tau2-bench sanity check
    t0 = time.perf_counter()
    tau2_summary = analyze_tau2(data_dir)
    timings["tau2_analysis_s"] = round(time.perf_counter() - t0, 2)
    print(f"[stage 4] tau2-bench analyzed ({tau2_summary.get('simulations', 0):,} sims) in {timings['tau2_analysis_s']:.2f}s", flush=True)

    # 5. Report and Plots
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
    )
    import pandas as pd
    from dataclasses import asdict

    plot_step_vs_dollar_share(
        tb_by_model,
        swe_by_model,
        assets_dir / "step_vs_dollar_share_per_model.png",
    )
    plot_detector_savings_vs_false_stop(
        pd.DataFrame([asdict(d) for d in tb_detectors]),
        pd.DataFrame([asdict(d) for d in swe_detectors]),
        assets_dir / "detector_savings_vs_false_stop.png",
    )
    plot_loop_prevalence_by_harness(
        tb_by_harness,
        swe_by_harness,
        assets_dir / "loop_prevalence_by_harness.png",
    )
    timings["plots_and_report_s"] = round(time.perf_counter() - t0, 2)
    timings["total_s"] = round(time.perf_counter() - t_start, 2)

    # Re-write study_metrics.json with final timings
    import json
    metrics["timings_sec"] = timings
    (results_dir / "study_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Print headline numbers
    print("\n" + "=" * 72)
    print("HEADLINE SUMMARY")
    print("=" * 72)
    print((results_dir / "summary.md").read_text())
    print(f"\nCompleted full reproducible study in {timings['total_s']:.2f}s.")


if __name__ == "__main__":
    main()
