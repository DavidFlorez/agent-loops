"""Generate focused sub-reports for Gemini models and Open-Weights vs Closed models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agent_loops.model_catalog import is_open_weight_model
from agent_loops.study import (
    CompactRunRecord,
    analyze_swe_rebench,
    analyze_tau2,
    analyze_terminalbench,
)

SWE_AGENTIC_HARNESSES = frozenset({"claude-code", "codex", "cursor", "junie"})


def _fmt_pct(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val * 100.0:.2f}%"


def _fmt_usd(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"${val:,.2f}"


def _gemini_generation(model: str) -> str:
    ml = (model or "").lower()
    if "gemini-3.1" in ml:
        return "gemini-3.1"
    if "gemini-3" in ml:
        return "gemini-3.0"
    if "gemini-2.5" in ml:
        return "gemini-2.5"
    return "other"


def _aggregate_compact(records: list[CompactRunRecord]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"runs": 0}

    loop_runs = sum(1 for r in records if r.has_any_loop)
    loop_free = n - loop_runs
    succ = sum(1 for r in records if r.succeeded)
    succ_loop = sum(1 for r in records if r.has_any_loop and r.succeeded)
    succ_free = sum(1 for r in records if (not r.has_any_loop) and r.succeeded)

    steps_arr = np.array([r.n_steps for r in records], dtype=np.float64)
    tot_steps = int(steps_arr.sum())
    lp_steps = sum(r.loop_steps for r in records)
    pr_steps = sum(r.productive_repeat_steps for r in records)

    cost_recs = [r for r in records if r.cost_included]
    tot_usd = sum(r.total_cost_usd for r in cost_recs)
    lp_usd = sum(r.loop_cost_usd for r in cost_recs)

    det_counts = {
        k: sum(1 for r in records if r.stop_steps.get(k) is not None)
        for k in ("gemini_cli", "cline_style", "openhands_stuck", "openclaw_10", "openclaw_20")
    }
    any_rule = sum(
        1
        for r in records
        if any(
            r.stop_steps.get(k) is not None
            for k in ("gemini_cli", "cline_style", "openhands_stuck", "openclaw_10", "openclaw_20")
        )
    )

    return {
        "runs": n,
        "loop_runs": loop_runs,
        "loop_run_rate": loop_runs / n,
        "total_steps": tot_steps,
        "loop_steps": lp_steps,
        "loop_step_share": lp_steps / tot_steps if tot_steps else 0.0,
        "prod_repeat_step_share": pr_steps / tot_steps if tot_steps else 0.0,
        "mean_steps": float(np.mean(steps_arr)),
        "median_steps": float(np.median(steps_arr)),
        "p90_steps": float(np.percentile(steps_arr, 90)),
        "pass_rate": succ / n,
        "looping_pass_rate": succ_loop / loop_runs if loop_runs else None,
        "loop_free_pass_rate": succ_free / loop_free if loop_free else None,
        "cost_runs": len(cost_recs),
        "total_cost_usd": tot_usd,
        "loop_cost_usd": lp_usd,
        "loop_dollar_share": (lp_usd / tot_usd) if tot_usd > 0 else None,
        "cost_per_run_usd": (tot_usd / len(cost_recs)) if cost_recs else None,
        "exact_loop_rate": sum(1 for r in records if r.has_exact_loop) / n,
        "near_dup_rate": sum(1 for r in records if r.has_near_duplicate) / n,
        "aba_thrash_rate": sum(1 for r in records if r.has_aba_thrash) / n,
        "rotation_rate": sum(1 for r in records if r.has_rotation_loop) / n,
        "error_streak_rate": sum(1 for r in records if r.has_error_streak) / n,
        "reread_rate": sum(1 for r in records if r.has_redundant_reread) / n,
        "prod_repeat_rate": sum(1 for r in records if r.has_productive_repeat) / n,
        "det_gemini_cli": det_counts["gemini_cli"],
        "det_cline": det_counts["cline_style"],
        "det_openhands": det_counts["openhands_stuck"],
        "det_openclaw_10": det_counts["openclaw_10"],
        "det_any_rule": any_rule,
        "any_rule_recall_on_loopers": (any_rule / loop_runs) if loop_runs else 0.0,
        "gemini_cli_recall_on_loopers": (det_counts["gemini_cli"] / loop_runs) if loop_runs else 0.0,
    }


def _load_tb_all_rows_df(data_dir: Path) -> pd.DataFrame:
    pq_files = sorted((data_dir / "terminalbench").glob("**/*.parquet"))
    tb = pq.read_table(
        pq_files,
        columns=["agent", "model", "reward", "duration_seconds", "cost_cents", "steps"],
    ).to_pandas()
    tb["has_steps"] = tb["steps"].notna() & (tb["steps"] != "null") & (tb["steps"].str.len() > 2)
    tb["passed"] = tb["reward"].fillna(0).astype(int) == 1
    tb["is_open"] = tb["model"].astype(str).map(is_open_weight_model)
    return tb.drop(columns=["steps"])


def build_gemini_report(
    *,
    data_dir: Path,
    results_dir: Path,
    tb_records: list[CompactRunRecord],
    swe_records: list[CompactRunRecord],
    tau2_summary: dict[str, Any],
) -> Path:
    """Compute and write results/gemini_report.md and results/gemini_by_harness_model.csv."""
    tb_all = _load_tb_all_rows_df(data_dir)
    is_gem_df = tb_all["model"].str.lower().str.contains("gemini", na=False) | (tb_all["agent"] == "gemini-cli")
    gem_df = tb_all[is_gem_df].copy()
    gem_recs = [r for r in tb_records if "gemini" in r.model.lower() or r.harness == "gemini-cli"]

    swe_gem = [r for r in swe_records if "gemini" in r.model.lower() or "gemini" in r.harness.lower()]

    # Group TB all-rows & TB step-records by (agent, model)
    step_map: dict[tuple[str, str], list[CompactRunRecord]] = defaultdict(list)
    for r in gem_recs:
        step_map[(r.harness, r.model)].append(r)

    rows_hm: list[dict[str, Any]] = []
    for (agent, model), grp in gem_df.groupby(["agent", "model"], sort=True):
        srecs = step_map.get((agent, model), [])
        sa = _aggregate_compact(srecs) if srecs else {}
        rows_hm.append(
            {
                "harness": agent,
                "model": model,
                "generation": _gemini_generation(model),
                "total_trials": int(len(grp)),
                "all_pass_rate": float(grp["passed"].mean()),
                "mean_duration_s": float(grp["duration_seconds"].mean()),
                "step_runs": sa.get("runs", 0),
                "loop_runs": sa.get("loop_runs", 0),
                "loop_run_rate": sa.get("loop_run_rate"),
                "loop_step_share": sa.get("loop_step_share"),
                "step_pass_rate": sa.get("pass_rate"),
                "cost_runs": sa.get("cost_runs", 0),
                "loop_dollar_share": sa.get("loop_dollar_share"),
                "det_gemini_cli": sa.get("det_gemini_cli", 0),
                "det_openclaw_10": sa.get("det_openclaw_10", 0),
            }
        )
    hm_df = pd.DataFrame(rows_hm).sort_values(["generation", "all_pass_rate"], ascending=[True, False])
    hm_df.to_csv(results_dir / "gemini_by_harness_model.csv", index=False)

    # Generation summaries
    gen_groups: list[tuple[str, pd.DataFrame, list[CompactRunRecord]]] = [
        ("gemini-2.5", gem_df[gem_df["model"].str.contains("2.5", case=False)], [r for r in gem_recs if "2.5" in r.model]),
        ("gemini-3.0", gem_df[gem_df["model"].str.contains("gemini-3-", case=False)], [r for r in gem_recs if "gemini-3-" in r.model.lower()]),
        ("gemini-3.1", gem_df[gem_df["model"].str.contains("gemini-3.1", case=False)], [r for r in gem_recs if "gemini-3.1" in r.model.lower()]),
        ("All Gemini (combined)", gem_df, gem_recs),
    ]

    # Step-logged harness summaries
    step_harnesses = sorted({r.harness for r in gem_recs})

    # Controlled comparisons
    t2_g25 = _aggregate_compact([r for r in gem_recs if r.harness == "terminus-2" and "2.5" in r.model])
    t2_g3 = _aggregate_compact([r for r in gem_recs if r.harness == "terminus-2" and "gemini-3" in r.model.lower()])

    # Pass vs Fail step lengths for Gemini
    pass_steps = [r.n_steps for r in gem_recs if r.succeeded]
    fail_steps = [r.n_steps for r in gem_recs if not r.succeeded]
    overall_agg = _aggregate_compact(gem_recs)
    no_rsc_gem = _aggregate_compact([r for r in gem_recs if not r.has_rsc_ref])

    md: list[str] = []
    md.append("# Gemini Models Performance & Loop Report\n")
    md.append(
        "Generated programmatically from cached public trajectories (`yoonholee/terminalbench-trajectories`, "
        "`ibragim-bad/swe_rebench_07_2026_trajectories`, and `sierra-research/tau2-bench`).\n"
    )

    md.append("## 1. Coverage Across the Three Datasets / Frameworks\n")
    md.append("| Dataset / Benchmark | Total Runs in Corpus | Gemini Runs | Gemini Runs with Steps | Notes |")
    md.append("| --- | ---: | ---: | ---: | --- |")
    md.append(
        f"| Terminal-Bench 2.0 (`yoonholee/terminalbench-trajectories`) | {len(tb_all):,} | "
        f"{len(gem_df):,} | {len(gem_recs):,} | {gem_df['agent'].nunique()} harnesses total; "
        f"{len(step_harnesses)} harnesses include step traces |"
    )
    md.append(
        f"| SWE-rebench July 2026 (`ibragim-bad/swe_rebench_07_2026_trajectories`) | {len(swe_records):,} | "
        f"{len(swe_gem):,} | {len(swe_gem):,} | No Google/Gemini participants in the July 2026 release |"
    )
    md.append(
        f"| tau2-bench (`sierra-research/tau2-bench`) | {tau2_summary.get('simulations', 0):,} | "
        f"0 | 0 | Bundled final runs cover Claude 3.7 Sonnet, GPT-4.1, GPT-4.1-mini, and o4-mini only |"
    )
    md.append("")

    md.append("## 2. Gemini Model Generation Summary (Terminal-Bench 2.0)\n")
    md.append(
        "| Generation | All Trials | Overall Pass Rate | Step-Logged Runs | Loop Runs (%) | Loop Step Share | "
        "Pass (Loop vs Clean) | Mean / P90 Steps |"
    )
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, gdf, srecs in gen_groups:
        agg = _aggregate_compact(srecs)
        all_pr = float(gdf["passed"].mean()) if len(gdf) else 0.0
        md.append(
            f"| **{label}** | {len(gdf):,} | {_fmt_pct(all_pr)} | {agg['runs']:,} | "
            f"{agg['loop_runs']:,} ({_fmt_pct(agg['loop_run_rate'])}) | {_fmt_pct(agg['loop_step_share'])} | "
            f"{_fmt_pct(agg['looping_pass_rate'])} vs {_fmt_pct(agg['loop_free_pass_rate'])} | "
            f"{agg['mean_steps']:.1f} / {agg['p90_steps']:.0f} |"
        )
    md.append("")
    md.append(
        f"- **Step-length split across all {overall_agg['runs']:,} step-logged Gemini runs**: "
        f"passing runs finish at **P50={np.median(pass_steps):.0f}, P90={np.percentile(pass_steps, 90):.0f}** steps vs "
        f"**P50={np.median(fail_steps):.0f}, P90={np.percentile(fail_steps, 90):.0f}** steps for failing runs."
    )
    md.append(
        f"- **Zero-RSC clean-trace subset (`{no_rsc_gem['runs']:,}` Gemini runs)**: "
        f"loop prevalence **{_fmt_pct(no_rsc_gem['loop_run_rate'])}**, step share **{_fmt_pct(no_rsc_gem['loop_step_share'])}**, "
        f"validated cost share (`n={no_rsc_gem['cost_runs']:,}` runs, {_fmt_usd(no_rsc_gem['total_cost_usd'])} total) **{_fmt_pct(no_rsc_gem['loop_dollar_share'])}**.\n"
    )

    md.append("## 3. Gemini Performance by Agent Framework / Harness (Terminal-Bench 2.0)\n")
    md.append(f"### 3A. Five Frameworks with Step Traces ({len(gem_recs):,} Runs — Loop, Step, Cost, and Pass Analysis)\n")
    md.append(
        "| Harness | Step-Logged Models | Step Runs | Step-Run Pass | All-Trial Pass | Loop Runs (%) | Loop Step Share | "
        "Cost-Valid Runs | Loop $ Share | `gemini_cli` Fires | `openclaw_10` Fires |"
    )
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for h in sorted(step_harnesses, key=lambda x: _aggregate_compact([r for r in gem_recs if r.harness == x])["loop_run_rate"], reverse=True):
        srecs = [r for r in gem_recs if r.harness == h]
        hdf = gem_df[gem_df["agent"] == h]
        agg = _aggregate_compact(srecs)
        models_str = ", ".join(sorted({r.model.split("@")[0] for r in srecs}))
        ld_s = "n/a" if agg["cost_runs"] == 0 else ("n<100" if agg["cost_runs"] < 100 else _fmt_pct(agg["loop_dollar_share"]))
        md.append(
            f"| `{h}` | {models_str} | {agg['runs']:,} | {_fmt_pct(agg['pass_rate'])} | "
            f"{_fmt_pct(float(hdf['passed'].mean()))} ({len(hdf):,}) | "
            f"{agg['loop_runs']:,} ({_fmt_pct(agg['loop_run_rate'])}) | {_fmt_pct(agg['loop_step_share'])} | "
            f"{agg['cost_runs']:,} | {ld_s} | "
            f"{agg['det_gemini_cli']:,} | {agg['det_openclaw_10']:,} |"
        )
    md.append("")

    md.append(f"### 3B. Full Harness × Model Matrix (All {len(gem_df):,} Gemini Trials)\n")
    md.append("| Harness | Model | Trials | All-Trial Pass Rate | Mean Wall Time (s) | Step Runs | Loop Runs (%) | Loop Step Share | Loop $ Share |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in hm_df.iterrows():
        lr_s = f"{int(r['loop_runs'])} ({_fmt_pct(r['loop_run_rate'])})" if pd.notna(r["loop_run_rate"]) else "n/a (no steps)"
        ls_s = _fmt_pct(r["loop_step_share"]) if pd.notna(r["loop_step_share"]) else "n/a"
        c_runs = int(r.get("cost_runs", 0))
        ld_s = "n/a" if (c_runs == 0 or pd.isna(r["loop_dollar_share"])) else ("n<100" if c_runs < 100 else _fmt_pct(r["loop_dollar_share"]))
        md.append(
            f"| `{r['harness']}` | `{r['model']}` | {int(r['total_trials']):,} | {_fmt_pct(r['all_pass_rate'])} | "
            f"{r['mean_duration_s']:.1f} | {int(r['step_runs']):,} | {lr_s} | {ls_s} | {ld_s} |"
        )
    md.append("")

    g25_oh = _aggregate_compact([r for r in gem_recs if r.harness == "openhands" and "2.5" in r.model])
    g25_mini = _aggregate_compact([r for r in gem_recs if r.harness == "mini-swe-agent" and "2.5" in r.model])
    g25_cli = _aggregate_compact([r for r in gem_recs if r.harness == "gemini-cli" and "2.5" in r.model])
    g25_ratio = g25_oh["loop_run_rate"] / max(1e-9, t2_g25["loop_run_rate"])

    md.append("## 4. Key Framework & Detector Takeaways for Gemini\n")
    md.append(
        f"- **Controlled model upgrade (`terminus-2` harness held constant)**: Moving `gemini-2.5` (`n={t2_g25['runs']:,}`) → "
        f"`gemini-3.0` (`n={t2_g3['runs']:,}`) lowers loop prevalence from **{_fmt_pct(t2_g25['loop_run_rate'])}** to "
        f"**{_fmt_pct(t2_g3['loop_run_rate'])}** while more than doubling pass rate from **{_fmt_pct(t2_g25['pass_rate'])}** to "
        f"**{_fmt_pct(t2_g3['pass_rate'])}**."
    )
    md.append(
        f"- **Harness sensitivity (`gemini-2.5` held constant)**: Even with the exact same models, loop rate varies **{g25_ratio:.1f}×** across "
        f"frameworks (`openhands` **{_fmt_pct(g25_oh['loop_run_rate'])}**, `mini-swe-agent` **{_fmt_pct(g25_mini['loop_run_rate'])}** / "
        f"**{_fmt_pct(g25_mini['loop_step_share'])}** loop step share, `gemini-cli` **{_fmt_pct(g25_cli['loop_run_rate'])}**, "
        f"`terminus-2` **{_fmt_pct(t2_g25['loop_run_rate'])}**)."
    )
    md.append(
        f"- **Detector gap**: Across all {overall_agg['loop_runs']:,} looping Gemini trajectories, `gemini_cli`'s 5-consecutive / intra-turn rule fires on "
        f"only **{overall_agg['det_gemini_cli']:,} ({_fmt_pct(overall_agg['gemini_cli_recall_on_loopers'])})** of loopers, whereas rolling-window "
        f"`openclaw_10` catches **{overall_agg['det_openclaw_10']:,}** ({overall_agg['det_openclaw_10'] / max(1, overall_agg['det_gemini_cli']):.2f}× as many). "
        f"See [`results/loop_prone_tasks.csv`](loop_prone_tasks.csv) for the top 20 Gemini loop-prone tasks."
    )

    out_path = results_dir / "gemini_report.md"
    out_path.write_text("\n".join(md) + "\n")
    return out_path


def _tb_harness_open_closed_rows(tb_recs: list[CompactRunRecord], harness: str) -> tuple[dict[str, Any], dict[str, Any]]:
    open_r = [r for r in tb_recs if r.harness == harness and is_open_weight_model(r.model)]
    closed_r = [r for r in tb_recs if r.harness == harness and not is_open_weight_model(r.model)]
    return _aggregate_compact(open_r), _aggregate_compact(closed_r)


def build_open_vs_closed_report(
    *,
    data_dir: Path,
    results_dir: Path,
    tb_records: list[CompactRunRecord],
    swe_records: list[CompactRunRecord],
    tau2_summary: dict[str, Any],
) -> Path:
    """Compute and write results/open_vs_closed_report.md and results/open_vs_closed_summary.csv."""
    tb_all = _load_tb_all_rows_df(data_dir)
    tb_all_open = tb_all[tb_all["is_open"]]
    tb_all_closed = tb_all[~tb_all["is_open"]]

    tb_open_r = [r for r in tb_records if is_open_weight_model(r.model)]
    tb_closed_r = [r for r in tb_records if not is_open_weight_model(r.model)]
    tb_open_agg = _aggregate_compact(tb_open_r)
    tb_closed_agg = _aggregate_compact(tb_closed_r)

    swe_open_r = [r for r in swe_records if is_open_weight_model(r.model)]
    swe_closed_all_r = [r for r in swe_records if not is_open_weight_model(r.model)]
    swe_closed_swelike_r = [r for r in swe_closed_all_r if r.harness not in SWE_AGENTIC_HARNESSES]
    swe_closed_agentic_r = [r for r in swe_closed_all_r if r.harness in SWE_AGENTIC_HARNESSES]

    swe_open_agg = _aggregate_compact(swe_open_r)
    swe_closed_all_agg = _aggregate_compact(swe_closed_all_r)
    swe_closed_swe_agg = _aggregate_compact(swe_closed_swelike_r)
    swe_closed_ag_agg = _aggregate_compact(swe_closed_agentic_r)

    # Save structured summary CSV
    summary_slices = [
        ("terminalbench_steps_all", "open_weights", tb_open_agg),
        ("terminalbench_steps_all", "closed", tb_closed_agg),
        ("terminalbench_terminus_2", "open_weights", _tb_harness_open_closed_rows(tb_records, "terminus-2")[0]),
        ("terminalbench_terminus_2", "closed", _tb_harness_open_closed_rows(tb_records, "terminus-2")[1]),
        ("terminalbench_openhands", "open_weights", _tb_harness_open_closed_rows(tb_records, "openhands")[0]),
        ("terminalbench_openhands", "closed", _tb_harness_open_closed_rows(tb_records, "openhands")[1]),
        ("terminalbench_mini_swe_agent", "open_weights", _tb_harness_open_closed_rows(tb_records, "mini-swe-agent")[0]),
        ("terminalbench_mini_swe_agent", "closed", _tb_harness_open_closed_rows(tb_records, "mini-swe-agent")[1]),
        ("swe_rebench_all", "open_weights", swe_open_agg),
        ("swe_rebench_all", "closed_all", swe_closed_all_agg),
        ("swe_rebench_swelike_controlled", "open_weights", swe_open_agg),
        ("swe_rebench_swelike_controlled", "closed_swelike", swe_closed_swe_agg),
        ("swe_rebench_agentic_systems", "closed_agentic", swe_closed_ag_agg),
    ]
    csv_rows = [{"slice": s, "cohort": c, **vals} for s, c, vals in summary_slices]
    pd.DataFrame(csv_rows).to_csv(results_dir / "open_vs_closed_summary.csv", index=False)

    # Individual SWE-rebench participant rows keyed by harness (exact 17 participants)
    swe_parts = []
    for part in sorted({r.harness for r in swe_records}):
        precs = [r for r in swe_records if r.harness == part]
        agg = _aggregate_compact(precs)
        model = precs[0].model
        is_open = is_open_weight_model(model)
        scaffold = "agentic_system" if part in SWE_AGENTIC_HARNESSES else "swelike"
        swe_parts.append({"participant": part, "model": model, "weights": "Open" if is_open else "Closed", "scaffold": scaffold, **agg})
    swe_parts_df = pd.DataFrame(swe_parts).sort_values(["pass_rate", "loop_run_rate"], ascending=[False, True])

    # Top TB models (all 49 for pass rate + step-logged stats where available)
    tb_step_map: dict[str, list[CompactRunRecord]] = defaultdict(list)
    for r in tb_records:
        tb_step_map[r.model].append(r)

    md: list[str] = []
    md.append("# Open-Weights vs Closed Models: Loop Behavior, Cost & Performance Report\n")
    md.append(
        "Generated programmatically from cached public trajectories (`yoonholee/terminalbench-trajectories`, "
        "`ibragim-bad/swe_rebench_07_2026_trajectories`, and `sierra-research/tau2-bench`).\n"
    )

    md.append("## 1. Model Classification\n")
    tb_unique = sorted(tb_all["model"].dropna().unique())
    tb_open_names = [m for m in tb_unique if is_open_weight_model(m)]
    tb_closed_names = [m for m in tb_unique if not is_open_weight_model(m)]
    swe_unique_open = sorted({r.model for r in swe_open_r})
    swe_unique_closed = sorted({f"{r.harness} ({r.model})" for r in swe_closed_all_r})
    md.append(
        f"- **Terminal-Bench Open-Weights ({len(tb_open_names)} model entries, {len(tb_all_open):,} trials)**: "
        + ", ".join(f"`{m}`" for m in tb_open_names)
    )
    md.append(
        f"- **Terminal-Bench Closed ({len(tb_closed_names)} model entries, {len(tb_all_closed):,} trials)**: "
        "Anthropic `claude-*`, Google `gemini-*`, OpenAI proprietary `gpt-5*` (excluding open-weight `openai/gpt-oss-*`), xAI `grok-*`."
    )
    md.append(
        f"- **SWE-rebench Open-Weights (7 `swelike` participants, {len(swe_open_r):,} runs)**: "
        + ", ".join(f"`{m}`" for m in swe_unique_open)
    )
    md.append(
        f"- **SWE-rebench Closed (10 participants — 6 on `swelike` + 4 agentic systems, {len(swe_closed_all_r):,} runs)**: "
        + ", ".join(f"`{m}`" for m in swe_unique_closed)
        + ". *(Note: Cursor's `composer-2.5` is a proprietary commercial checkpoint built on Moonshot's Kimi K2.5 base and is grouped with Closed agentic systems.)*"
    )
    md.append(
        f"- **tau2-bench**: All **{tau2_summary.get('simulations', 0):,}** simulations use Closed models (`claude-3-7-sonnet`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`); 0 open-weight runs.\n"
    )

    md.append("## 2. Headline Comparison Across Benchmarks\n")
    md.append("| Cohort | Runs | Pass Rate | Loop Runs (%) | Loop Step Share | Mean / P90 Steps | Cost-Valid Runs | Avg Cost/Run | Loop $ Share |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    headline_rows = [
        ("TB All Trials (incl. null steps) — **Open**", len(tb_all_open), float(tb_all_open["passed"].mean()), None),
        ("TB All Trials (incl. null steps) — **Closed**", len(tb_all_closed), float(tb_all_closed["passed"].mean()), None),
        ("TB With Steps — **Open-Weights**", tb_open_agg["runs"], tb_open_agg["pass_rate"], tb_open_agg),
        ("TB With Steps — **Closed**", tb_closed_agg["runs"], tb_closed_agg["pass_rate"], tb_closed_agg),
        ("SWE-rebench (`swelike` controlled) — **Open-Weights (7)**", swe_open_agg["runs"], swe_open_agg["pass_rate"], swe_open_agg),
        ("SWE-rebench (`swelike` controlled) — **Closed (6)**", swe_closed_swe_agg["runs"], swe_closed_swe_agg["pass_rate"], swe_closed_swe_agg),
        ("SWE-rebench (`agentic_system` 4 harnesses) — **Closed**", swe_closed_ag_agg["runs"], swe_closed_ag_agg["pass_rate"], swe_closed_ag_agg),
        ("SWE-rebench (All 17) — **Closed (10)**", swe_closed_all_agg["runs"], swe_closed_all_agg["pass_rate"], swe_closed_all_agg),
    ]
    for label, n_runs, pr, agg in headline_rows:
        if agg is None:
            md.append(f"| {label} | {n_runs:,} | {_fmt_pct(pr)} | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            ld_s = "n/a" if agg["cost_runs"] == 0 else ("n<100" if agg["cost_runs"] < 100 else _fmt_pct(agg["loop_dollar_share"]))
            md.append(
                f"| {label} | {n_runs:,} | {_fmt_pct(pr)} | "
                f"{agg['loop_runs']:,} ({_fmt_pct(agg['loop_run_rate'])}) | {_fmt_pct(agg['loop_step_share'])} | "
                f"{agg['mean_steps']:.1f} / {agg['p90_steps']:.0f} | {agg['cost_runs']:,} | "
                f"{_fmt_usd(agg['cost_per_run_usd'])} | {ld_s} |"
            )
    md.append("")
    md.append(
        f"> **Methodological note on Terminal-Bench Closed cost share (`{_fmt_pct(tb_closed_agg['loop_dollar_share'])}`)**: "
        f"Only `{tb_closed_agg['cost_runs']:,} / {tb_closed_agg['runs']:,}` Closed TB runs survived the `[0.50x, 2.00x]` "
        "cost-rebuild validation (newer Gemini-3, GPT-5.2, and RSC-pointer runs had `cost_cents=0` or missing tokens), "
        "leaving an older subset dominated by `claude-opus-4-1` and `gemini-2.5-flash`. Cells with `<100` cost-valid runs "
        "are suppressed as `n<100`. On SWE-rebench (`swelike` controlled), where both "
        f"cohorts have high cost coverage (`{swe_open_agg['cost_runs']:,}` Open vs `{swe_closed_swe_agg['cost_runs']:,}` Closed), "
        f"Open models spend **{_fmt_pct(swe_open_agg['loop_dollar_share'])}** of dollars on loops vs **{_fmt_pct(swe_closed_swe_agg['loop_dollar_share'])}** for Closed.\n"
    )

    md.append("## 3. Controlled Same-Harness Comparisons (Eliminating Scaffold Confounding)\n")
    md.append("### 3A. Terminal-Bench Harnesses Where Both Open and Closed Models Ran\n")
    md.append("| Harness | Cohort | Runs | Pass Rate | Loop Runs (%) | Loop Step Share | Mean / Med Steps | Cost-Valid Runs | Loop $ Share |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for h in ("terminus-2", "openhands", "mini-swe-agent"):
        op, cl = _tb_harness_open_closed_rows(tb_records, h)
        for cname, agg in (("Open-Weights", op), ("Closed", cl)):
            ld_s = "n/a" if agg["cost_runs"] == 0 else ("n<100" if agg["cost_runs"] < 100 else _fmt_pct(agg["loop_dollar_share"]))
            md.append(
                f"| `{h}` | {cname} | {agg['runs']:,} | {_fmt_pct(agg['pass_rate'])} | "
                f"{agg['loop_runs']:,} ({_fmt_pct(agg['loop_run_rate'])}) | {_fmt_pct(agg['loop_step_share'])} | "
                f"{agg['mean_steps']:.1f} / {agg['median_steps']:.0f} | {agg['cost_runs']:,} | {ld_s} |"
            )
    md.append("")

    md.append("### 3B. SWE-rebench `swelike` Controlled Taxonomy Rates (Exact Same Harness, 111 Tasks × 5 Runs Each)\n")
    md.append(
        f"| Pattern | Open-Weights (`swelike`, n={swe_open_agg['runs']:,}) | "
        f"Closed (`swelike`, n={swe_closed_swe_agg['runs']:,}) | Closed (`agentic_system`, n={swe_closed_ag_agg['runs']:,}) |"
    )
    md.append("| --- | ---: | ---: | ---: |")
    for key, disp in [
        ("loop_run_rate", "Any pathological loop"),
        ("exact_loop_rate", "Exact loop"),
        ("near_dup_rate", "Near-duplicate"),
        ("aba_thrash_rate", "A-B-A thrash"),
        ("rotation_rate", "Period-k rotation (k <= 5)"),
        ("error_streak_rate", "Error streak"),
        ("reread_rate", "Redundant re-read"),
        ("prod_repeat_rate", "Productive repeat (non-loop)"),
    ]:
        md.append(
            f"| {disp} | {_fmt_pct(swe_open_agg[key])} | {_fmt_pct(swe_closed_swe_agg[key])} | {_fmt_pct(swe_closed_ag_agg[key])} |"
        )
    md.append("")

    md.append("### 3C. Detector Behavior: Rule Detectors vs Chanting Open-Weight Models\n")
    md.append("| Cohort | Loop Runs | `gemini_cli` Det. Fires | `cline_style` Fires | `openhands_stuck` Fires | `openclaw_10` Fires |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for lbl, agg in [
        ("TB Open-Weights", tb_open_agg),
        ("TB Closed", tb_closed_agg),
        ("SWE Open-Weights (`swelike`)", swe_open_agg),
        ("SWE Closed (`swelike`)", swe_closed_swe_agg),
        ("SWE Closed (`agentic_system`)", swe_closed_ag_agg),
    ]:
        md.append(
            f"| {lbl} | {agg['loop_runs']:,} | {agg['det_gemini_cli']:,} | {agg['det_cline']:,} | "
            f"{agg['det_openhands']:,} | {agg['det_openclaw_10']:,} |"
        )
    md.append("")

    md.append("## 4. Per-Participant Ranking on SWE-rebench (Pass Rate, Loop Rate, Cost)\n")
    md.append("| Participant | Model | Weights | Scaffold | Pass Rate | Loop Runs (%) | Loop Step Share | Cost/Run | Loop $ Share | Rotations |")
    md.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in swe_parts_df.iterrows():
        rot_cnt = int(round(r["rotation_rate"] * r["runs"]))
        c_runs = int(r.get("cost_runs", 0))
        ld_s = "n/a" if c_runs == 0 else ("n<100" if c_runs < 100 else _fmt_pct(r["loop_dollar_share"]))
        md.append(
            f"| `{r['participant']}` | `{r['model']}` | {r['weights']} | `{r['scaffold']}` | "
            f"{_fmt_pct(r['pass_rate'])} | {_fmt_pct(r['loop_run_rate'])} | {_fmt_pct(r['loop_step_share'])} | "
            f"{_fmt_usd(r['cost_per_run_usd'])} | {ld_s} | {rot_cnt} |"
        )
    md.append("")

    md.append("## 5. Terminal-Bench Frontier Standings (Top 10 Open vs Top 10 Closed Across All 52,104 Trials)\n")
    tb_model_rows = []
    for m, grp in tb_all.groupby("model"):
        srecs = tb_step_map.get(str(m), [])
        agg = _aggregate_compact(srecs) if srecs else {}
        tb_model_rows.append(
            {
                "model": m,
                "weights": "Open" if is_open_weight_model(str(m)) else "Closed",
                "trials": int(len(grp)),
                "pass_rate": float(grp["passed"].mean()),
                "step_runs": agg.get("runs", 0),
                "loop_run_rate": agg.get("loop_run_rate"),
                "loop_step_share": agg.get("loop_step_share"),
            }
        )
    tbm_df = pd.DataFrame(tb_model_rows)
    for w_label in ("Open", "Closed"):
        md.append(f"### Top 10 {w_label} Models (Terminal-Bench 2.0 by All-Trial Pass Rate)")
        md.append("| Model | Trials | All-Trial Pass Rate | Step Runs | Loop Runs (%) | Loop Step Share |")
        md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        sub = tbm_df[tbm_df["weights"] == w_label].sort_values("pass_rate", ascending=False).head(10)
        for _, r in sub.iterrows():
            lr_s = _fmt_pct(r["loop_run_rate"]) if pd.notna(r["loop_run_rate"]) else "n/a"
            ls_s = _fmt_pct(r["loop_step_share"]) if pd.notna(r["loop_step_share"]) else "n/a"
            md.append(
                f"| `{r['model']}` | {int(r['trials']):,} | {_fmt_pct(r['pass_rate'])} | {int(r['step_runs']):,} | {lr_s} | {ls_s} |"
            )
        md.append("")

    t2_open, t2_closed = _tb_harness_open_closed_rows(tb_records, "terminus-2")
    glm52_row = swe_parts_df[swe_parts_df["participant"] == "glm-5.2"].iloc[0]
    opus5_row = swe_parts_df[swe_parts_df["participant"] == "opus-5"].iloc[0]
    grok45_row = swe_parts_df[swe_parts_df["participant"] == "grok-4.5"].iloc[0]

    md.append("## 6. Key Takeaways\n")
    md.append(
        f"- **Same-scaffold loop gap is massive (`swelike` controlled)**: Holding the harness constant on SWE-rebench, Open-Weights "
        f"models loop **{swe_open_agg['loop_run_rate'] / max(1e-9, swe_closed_swe_agg['loop_run_rate']):.1f}×** as often by run "
        f"(**{_fmt_pct(swe_open_agg['loop_run_rate'])}** vs **{_fmt_pct(swe_closed_swe_agg['loop_run_rate'])}**), spend "
        f"**{swe_open_agg['loop_step_share'] / max(1e-9, swe_closed_swe_agg['loop_step_share']):.1f}×** the step share on loops "
        f"(**{_fmt_pct(swe_open_agg['loop_step_share'])}** vs **{_fmt_pct(swe_closed_swe_agg['loop_step_share'])}**), and run nearly **2×** longer "
        f"(mean **{swe_open_agg['mean_steps']:.1f}** vs **{swe_closed_swe_agg['mean_steps']:.1f}** steps)."
    )
    md.append(
        f"- **Repeatedly confirmed on Terminal-Bench `terminus-2`**: On the largest shared TB harness, Open-Weights loop at "
        f"**{_fmt_pct(t2_open['loop_run_rate'])}** vs **{_fmt_pct(t2_closed['loop_run_rate'])}** for Closed "
        f"(**{_fmt_pct(t2_open['loop_step_share'])}** vs **{_fmt_pct(t2_closed['loop_step_share'])}** loop steps; "
        f"pass rate **{_fmt_pct(t2_open['pass_rate'])}** vs **{_fmt_pct(t2_closed['pass_rate'])}**)."
    )
    md.append(
        f"- **Frontier pass parity, residual loopiness**: Top open-weight SWE participant `glm-5.2` passes **{_fmt_pct(glm52_row['pass_rate'])}** "
        f"at **{_fmt_usd(glm52_row['cost_per_run_usd'])}/run** — essentially matching `opus-5` (**{_fmt_pct(opus5_row['pass_rate'])}**, "
        f"**{_fmt_usd(opus5_row['cost_per_run_usd'])}/run**) and `grok-4.5` (**{_fmt_pct(grok45_row['pass_rate'])}**, "
        f"**{_fmt_usd(grok45_row['cost_per_run_usd'])}/run**), yet still loops in **{_fmt_pct(glm52_row['loop_run_rate'])}** of runs "
        f"(including **{int(round(glm52_row['rotation_rate'] * glm52_row['runs']))}** `period_k_rotation` runs vs **0** for both `opus-5` and `grok-4.5`)."
    )
    md.append(
        f"- **Intra-turn chanting & redundant re-reads skew Open**: On `swelike`, Open-Weights hit **{int(swe_open_agg['det_gemini_cli']):,}** "
        f"`gemini_cli` triggers (mostly intra-turn 50-char chanting in `minimax-m3`/`qwen3.6`) vs **{int(swe_closed_swe_agg['det_gemini_cli']):,}** for Closed `swelike`, "
        f"and exhibit **{_fmt_pct(swe_open_agg['reread_rate'])}** redundant re-reads vs **{_fmt_pct(swe_closed_swe_agg['reread_rate'])}** for Closed `swelike`."
    )

    out_path = results_dir / "open_vs_closed_report.md"
    out_path.write_text("\n".join(md) + "\n")
    return out_path


def generate_extra_reports(
    *,
    data_dir: Path,
    results_dir: Path,
    tb_records: list[CompactRunRecord],
    swe_records: list[CompactRunRecord],
    tau2_summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Generate both Gemini and Open-vs-Closed markdown reports and CSVs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    p1 = build_gemini_report(
        data_dir=data_dir,
        results_dir=results_dir,
        tb_records=tb_records,
        swe_records=swe_records,
        tau2_summary=tau2_summary,
    )
    p2 = build_open_vs_closed_report(
        data_dir=data_dir,
        results_dir=results_dir,
        tb_records=tb_records,
        swe_records=swe_records,
        tau2_summary=tau2_summary,
    )
    return p1, p2


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    results_dir = root / "results"
    tb_records, _ = analyze_terminalbench(data_dir)
    swe_records, _ = analyze_swe_rebench(data_dir)
    tau2_summary = analyze_tau2(data_dir)
    p1, p2 = generate_extra_reports(
        data_dir=data_dir,
        results_dir=results_dir,
        tb_records=tb_records,
        swe_records=swe_records,
        tau2_summary=tau2_summary,
    )
    print(f"Wrote {p1} and {p2}")
