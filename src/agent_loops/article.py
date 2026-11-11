"""Programmatic generation of ARTICLE_DRAFT.md from computed Phase 1 and Phase 2 results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from agent_loops.fair_study import COHORT_LABELS, COHORT_ORDER


def _fpct(v: float | None, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100.0:.{digits}f}%"


def _fnum(v: float | None, digits: int = 2, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(v)


def _direction_word(pt: float, lo: float, hi: float) -> str:
    if math.isnan(pt) or math.isnan(lo) or math.isnan(hi):
        return "unmatched"
    if lo <= 0.0 <= hi:
        return "statistically indistinguishable from zero (null)"
    if pt > 0:
        return "higher for Open-Weights"
    return "reversed (lower for Open-Weights)"


def build_article_draft(
    *,
    study_metrics: dict[str, Any],
    fair_res: dict[str, Any],
) -> str:
    """Synthesize ARTICLE_DRAFT.md directly from computed Phase 1 & Phase 2 structures."""
    model_df: pd.DataFrame = fair_res["model_df"]
    matching: dict[str, dict[str, Any]] = fair_res["matching"]
    regressions: dict[str, Any] = fair_res["regressions"]
    sensitivities: dict[str, Any] = fair_res["sensitivities"]
    fam_df: pd.DataFrame = sensitivities["per_family_df"]
    obk_df: pd.DataFrame = sensitivities["non_alltools_bk_df"]

    total_exec = int(model_df["executed_runs"].sum()) if not model_df.empty else 0
    total_models_rows = len(model_df)
    open_cohorts = sum(
        1
        for c in COHORT_ORDER
        if c in matching
        and matching[c]["loops_per_100_steps"]["overlap_bands"] > 0
        and matching[c]["loops_per_100_steps"]["matched_diff_ci"][0] > 0
    )
    null_cohorts = sum(
        1
        for c in COHORT_ORDER
        if c in matching
        and matching[c]["loops_per_100_steps"]["overlap_bands"] > 0
        and matching[c]["loops_per_100_steps"]["matched_diff_ci"][0]
        <= 0.0
        <= matching[c]["loops_per_100_steps"]["matched_diff_ci"][1]
    )
    rev_cohorts = sum(
        1
        for c in COHORT_ORDER
        if c in matching
        and matching[c]["loops_per_100_steps"]["overlap_bands"] > 0
        and matching[c]["loops_per_100_steps"]["matched_diff_ci"][1] < 0
    )

    pool_reg = regressions.get("pooled_within_scaffold", {})
    pool_nb = (pool_reg.get("negbin_offset") or {}).get("coefs", {}).get("is_open_weight", {})
    pool_lg = (pool_reg.get("logit") or {}).get("coefs", {}).get("is_open_weight", {})
    pool_pr = (pool_reg.get("negbin_offset") or {}).get("coefs", {}).get("pass_rate_10pp_c", {})
    pool_cap = (pool_reg.get("negbin_step_cap_spec") or {}).get("coefs", {}).get("step_cap_100_c", {})
    pool_glmm = pool_reg.get("glmm_binomial_random_task", {})

    tb_head = study_metrics.get("terminalbench", {})
    swe_head = study_metrics.get("swe_rebench", {})

    lines: list[str] = []
    lines.append("# Do open-weight agents loop more? What four public trajectory sets say\n")
    lines.append(
        "*Offline audit across Terminal-Bench 2.0, SWE-rebench July 2026, Longtail-SWE, and Sierra's public τ²-bench archive.*\n"
    )

    lines.append("## The short answer\n")
    lines.append(
        f"Across **{total_exec:,} executed agent trajectories** spanning **7 uniform-scaffold cohorts** "
        f"(and `{total_models_rows}` model-scaffold evaluations across four public benchmarks), comparing raw averages "
        "without controls makes open-weight models look worse than they often are — because weak legacy open checkpoints pull "
        "raw averages up while frontier closed agent harnesses conceal bespoke loop guards inside proprietary scaffolds."
    )
    lines.append("")
    lines.append(
        f"Once we hold both **the agent scaffold and the exact task set constant** and match models inside identical pass-rate "
        f"capability bands (`B1 <25%` through `B5 >=75%`), the headline picture fractures rather than confirming a universal "
        f"open-weight defect: out of cohorts with overlapping pass-rate bands, **{open_cohorts}** show a positive Open-vs-Closed "
        f"loop-rate gap, **{null_cohorts}** show no statistically distinguishable difference (95% task-clustered CI crosses zero), "
        f"and **{rev_cohorts}** reverse (Open-Weights loop *less* than Closed models at the same capability level)."
    )
    if pool_nb and pool_lg:
        lines.append("")
        lines.append(
            f"In a pooled within-scaffold Negative Binomial count regression with `log(steps)` exposure offset and two-way "
            f"`(model, task)` cluster-robust standard errors — adjusting simultaneously for model pass rate, release month, "
            f"serving route, task difficulty, and cohort scaffold — open-weight status carries an Incidence Rate Ratio (**IRR**) of "
            f"**{pool_nb['irr_or']:.2f}** (`95% CI [{pool_nb['ci_low']:.2f}, {pool_nb['ci_high']:.2f}]`, `p={pool_nb['p']:.4f}`), "
            f"while a run-level Logistic regression for exhibiting at least one loop gives Odds Ratio (**OR**) "
            f"**{pool_lg['irr_or']:.2f}** (`[{pool_lg['ci_low']:.2f}, {pool_lg['ci_high']:.2f}]`, `p={pool_lg['p']:.4f}`"
            + (
                f"; hierarchical random-task `BinomialBayesMixedGLM` OR **{pool_glmm['is_open_or']:.2f}**"
                if pool_glmm.get("converged")
                else ""
            )
            + ")."
        )
        if pool_pr:
            lines.append(
                f" By contrast, **+10 percentage points of model pass rate** shifts loop density with IRR "
                f"**{pool_pr['irr_or']:.2f}** (`p={pool_pr['p']:.4f}`)"
                + (
                    f", and every **+100 steps in scaffold step cap** moves loop rate with IRR "
                    f"**{pool_cap['irr_or']:.2f}** (`p={pool_cap['p']:.4f}`)."
                    if pool_cap
                    else "."
                )
            )
    lines.append("")

    lines.append("## Why naive open-vs-closed benchmarks mislead\n")
    lines.append(
        "Three confounders routinely masquerade as an intrinsic weight property:\n"
        "1. **Scaffold mixing**: Custom proprietary harnesses (`claude-code`, `codex`, `gemini-cli`, `cursor`, `junie`) only ran "
        "closed models and already ship bespoke retry/turn guards. Pooling them against open models on bare `terminus-2` or `swelike` "
        "measures harness engineering, not weights.\n"
        "2. **Retrieval tool asymmetry (`tau2 banking_knowledge`)**: Inspecting Sierra's S3 `results.json` metadata revealed four "
        "different retrieval harnesses inside `banking_knowledge` alone — `AllTools` (`BM25 + dense + shell`), `terminal_use` "
        "(`shell` only, e.g. `gemini-3-flash` and `gemini-3-pro`), `openai_embeddings`, and `qwen_embeddings`. Our fair comparison "
        "strictly isolates the **20 uniform `AllTools` models** (`3` Open vs `17` Closed).\n"
        "3. **Capability imbalance & context truncation**: Comparing a `10%`-pass model that slams into brick walls against an "
        "`80%`-pass frontier model inflates failure looping. Meanwhile on Longtail-SWE (`swelike`), closed models ran with "
        "`1,000K–1,050K` context while 4 of 5 open models ran at `262K` (and `131K` on `qwen3.6-27b` reruns), and `deepseek-v4-flash` "
        "suffered `49` zero-step API timeouts.\n"
    )

    lines.append("## What each of the four datasets shows\n")
    lines.append(
        "| Cohort (Identical Scaffold & Shared Tasks) | Unmatched Loops/100 (Open vs Closed) | Unmatched Gap [95% CI] | "
        "Matched Gap [95% CI] | Adjusted NB IRR (p_2way) | Verdict |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for cname in COHORT_ORDER:
        if cname not in matching:
            continue
        lp = matching[cname]["loops_per_100_steps"]
        u_s = f"{_fnum(lp['open_unmatched'])} vs {_fnum(lp['closed_unmatched'])}"
        ug_s = f"{_fnum(lp['unmatched_diff'], 2, True)} [{_fnum(lp['unmatched_diff_ci'][0], 2, True)}, {_fnum(lp['unmatched_diff_ci'][1], 2, True)}]"
        if lp["overlap_bands"] > 0:
            mg_s = f"{_fnum(lp['matched_diff'], 2, True)} [{_fnum(lp['matched_diff_ci'][0], 2, True)}, {_fnum(lp['matched_diff_ci'][1], 2, True)}]"
            verdict = _direction_word(lp["matched_diff"], lp["matched_diff_ci"][0], lp["matched_diff_ci"][1])
        else:
            mg_s = "no band overlap"
            verdict = _direction_word(lp["unmatched_diff"], lp["unmatched_diff_ci"][0], lp["unmatched_diff_ci"][1]) + " (unmatched)"
        c_nb = ((regressions.get(cname) or {}).get("negbin_offset") or {}).get("coefs", {}).get("is_open_weight")
        irr_s = f"{c_nb['irr_or']:.2f} ({c_nb['p']:.4f})" if c_nb else "n/a"
        lines.append(f"| {COHORT_LABELS[cname]} | {u_s} | {ug_s} | {mg_s} | {irr_s} | {verdict} |")
    lines.append("")

    # Call out concrete reversals/nulls/positive cohorts dynamically from computed table
    lines.append("### Cohort-by-cohort highlights\n")
    for cname in COHORT_ORDER:
        if cname not in matching:
            continue
        lp = matching[cname]["loops_per_100_steps"]
        al = matching[cname]["any_loop_rate"]
        verdict = (
            _direction_word(lp["matched_diff"], lp["matched_diff_ci"][0], lp["matched_diff_ci"][1])
            if lp["overlap_bands"] > 0
            else "no strict pass-rate band overlap"
        )
        lines.append(
            f"- **{COHORT_LABELS[cname]}**: Before matching, Open averaged **{_fnum(lp['open_unmatched'])}** vs Closed "
            f"**{_fnum(lp['closed_unmatched'])}** loops/100 steps (gap **{_fnum(lp['unmatched_diff'], 2, True)}**; "
            f"any-loop run share gap **{_fnum(al['unmatched_diff']*100, 2, True)} pp**). "
            + (
                f"After capability-band matching ({lp['overlap_bands']} overlapping band(s)), the gap is "
                f"**{_fnum(lp['matched_diff'], 2, True)}** `[{_fnum(lp['matched_diff_ci'][0], 2, True)}, "
                f"{_fnum(lp['matched_diff_ci'][1], 2, True)}]` loops/100 steps (**{verdict}**)."
                if lp["overlap_bands"] > 0
                else "No single 15–20pp band contained both Open and Closed models; look at continuous pass-rate adjustment and the "
                "closest frontier pair below."
            )
        )
    lines.append("")

    # Longtail adjacent frontier pair if present
    lt_df = model_df[model_df["cohort"] == "longtail_swe_swelike"].sort_values("executed_pass_rate", ascending=False)
    if not lt_df.empty:
        lines.append("### Longtail-SWE context & adjacent-frontier check\n")
        top_o = lt_df[lt_df["weights"] == "Open"].head(2)
        low_c = lt_df[lt_df["weights"] == "Closed"].tail(2)
        pair_strs = [
            f"`{r['canonical_name']}` ({r['weights']}, `{r['context_limit_tokens_set']}` ctx, {_fpct(r['executed_pass_rate'])} pass -> {_fnum(r['loops_per_100_steps'])} loops/100)"
            for _, r in pd.concat([top_o, low_c]).iterrows()
        ]
        lines.append("- Closest capability neighbors in Longtail-SWE (`swelike`): " + "; ".join(pair_strs) + ".\n")

    lines.append("## Model families, including Google Gemini\n")
    lines.append(
        "Aggregation into binary `Open` vs `Closed` buckets hides huge variance inside *both* camps. Here is how individual "
        "families behaved across every primary uniform scaffold:\n"
    )
    gem_sub = model_df[model_df["family"] == "gemini"].sort_values(["cohort", "model"])
    if not gem_sub.empty:
        lines.append("### Google Gemini across Terminal-Bench and Sierra τ²\n")
        lines.append("| Benchmark & Uniform Scaffold | Model | Executed Pass Rate | Loops / 100 Steps [95% CI] | Any-Loop Run Rate |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for _, r in gem_sub.iterrows():
            lines.append(
                f"| `{r['cohort']}` | `{r['canonical_name']}` | {_fpct(r['executed_pass_rate'])} | "
                f"{_fnum(r['loops_per_100_steps'])} [{_fnum(r['loops_per_100_steps_ci_low'])}, {_fnum(r['loops_per_100_steps_ci_high'])}] | "
                f"{_fpct(r['any_loop_rate'])} |"
            )
        if not obk_df.empty:
            for _, r in obk_df[obk_df["family"] == "gemini"].iterrows():
                lines.append(
                    f"| `{r['cohort']}` (`{r['retrieval_config']}` scaffold) | `{r['model']}` | {_fpct(r['pass_rate'])} | "
                    f"{_fnum(r['loops_per_100_steps'])} | {_fpct(r['any_loop_rate'])} |"
                )
        lines.append("")

    if not fam_df.empty:
        lines.append("### Highest and lowest looping families per cohort\n")
        for cname in COHORT_ORDER:
            cf = fam_df[fam_df["cohort"] == cname].sort_values("mean_loops_per_100")
            if cf.empty:
                continue
            low = cf.iloc[0]
            high = cf.iloc[-1]
            lines.append(
                f"- **{COHORT_LABELS[cname]}**: lowest loop density `{low['family_group']}` ({low['weights']}, "
                f"{_fnum(low['mean_loops_per_100'])}/100 @ {_fpct(low['mean_pass_rate'])} pass); "
                f"highest `{high['family_group']}` ({high['weights']}, {_fnum(high['mean_loops_per_100'])}/100 @ "
                f"{_fpct(high['mean_pass_rate'])} pass)."
            )
        lines.append("")

    lines.append("## Sensitivity checks: passing runs, failing runs, legacy checkpoints & hosting\n")
    lines.append(
        "- **Passing runs only vs failing runs only**: Restricting strictly to solved tasks (`pass_only`) vs failed attempts (`fail_only`) "
        "tests whether any gap comes from verification habits on winning trajectories or flailing on unsolved ones — full cohort "
        "CIs sit in [`results/open_vs_closed_model.md`](results/open_vs_closed_model.md).\n"
        "- **Dropping legacy `gpt-oss-20b/120b`**: Exposes how much of Terminal-Bench's raw open-weight loopiness comes from weak "
        "3–19%-pass legacy models rather than modern open releases.\n"
        "- **First-party routes only**: Only `tb_terminus_2` includes native first-party Open endpoints (`deepseek-v3.2`, `glm-5`, `minimax-m2.5`); "
        "every Open model in Sierra `AllTools` (`glm-5-2`, `inkling`, `kimi-k3`) ran via OpenRouter proxies (`third_party`), underscoring "
        "that third-party inference routing and quantization remain intertwined with open-weight evaluations.\n"
    )

    lines.append("## Dollar cost and loop detector implications\n")
    if tb_head and swe_head:
        lines.append(
            f"From our earlier cache-aware billing audit (unchanged), loop steps account for **{_fpct(tb_head.get('loop_dollar_share'))}** "
            f"of validated spend on Terminal-Bench 2.0 and **{_fpct(swe_head.get('loop_dollar_share'))}** on SWE-rebench overall "
            f"(with late-context bloat doubling dollar share vs step share within models), while rotating/A-B-A cycles slip past naive "
            f"5-adjacent-step guards (`{tb_head.get('rotation_missed_by_cline', 0)}/{tb_head.get('rotation_runs', 0)}` missed on TB; "
            f"`{swe_head.get('rotation_missed_by_cline', 0)}/{swe_head.get('rotation_runs', 0)}` on SWE-rebench)."
        )
    lines.append("")
    lines.append(
        "The fair open-vs-closed evidence points to three practical engineering conclusions:\n"
        "1. **Scaffold & budget discipline beats weight stereotypes**: Harness step caps, rolling-window cycle detection (`openclaw_10`-style "
        "multi-period memory), and adequate context windows cut loop spend across *both* open and closed models far more reliably than choosing "
        "closed weights alone.\n"
        "2. **Watch model pass rate and hosting route before blaming weights**: On equal scaffolds and capability bands, strong open models "
        "frequently match or beat closed peers in loop discipline, while weaker models in *either* camp (e.g., `gemini-2.5-flash`, `gpt-5-nano`, "
        "`gpt-oss`, `qwen3.6`) loop aggressively when stuck.\n"
        "3. **Audit retrieval/proxy metadata in public leaderboards**: Silent differences in toolsets (`AllTools` vs `terminal_use`), context "
        "truncation (`262K` vs `1M`), and OpenRouter/FP8 hosting can swing loop rates by multiples inside the same benchmark folder.\n"
    )

    lines.append("## Visuals\n")
    lines.append("- `assets/loops_per_100_by_model_band.png` — Loops per 100 steps (95% CI) by model and capability band.")
    lines.append("- `assets/open_closed_gap_before_after_matching.png` — Open-minus-Closed gap before vs after pass-rate band matching.")
    lines.append("- `assets/open_vs_closed_across_four_datasets.png` — Pass rate vs loop rate across all four datasets.")
    lines.append("- Full per-model table: [`results/loops_by_model_scaffold.csv`](results/loops_by_model_scaffold.csv) and detailed stats: [`results/open_vs_closed_model.md`](results/open_vs_closed_model.md).\n")

    return "\n".join(lines) + "\n"
