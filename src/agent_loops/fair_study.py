"""Pre-registered Open-Weights vs Closed fair-comparison analysis across four trajectory datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from agent_loops.model_catalog import (
    COHORT_STEP_CAPS,
    get_model_meta,
    get_serving_route,
)
from agent_loops.study import CompactRunRecord

RANDOM_SEED = 42
N_BOOT = 1000

# Pre-registered uniform capability pass-rate bands across all benchmarks
CAPABILITY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("B1 (<25%)", 0.00, 0.25),
    ("B2 (25-40%)", 0.25, 0.40),
    ("B3 (40-55%)", 0.40, 0.55),
    ("B4 (55-75%)", 0.55, 0.75),
    ("B5 (>=75%)", 0.75, 1.01),
)

COHORT_ORDER: tuple[str, ...] = (
    "tb_terminus_2",
    "tb_openhands",
    "tb_mini_swe_agent",
    "swe_rebench_swelike",
    "longtail_swe_swelike",
    "tau2_classic",
    "tau2_banking_knowledge_alltools",
)

COHORT_LABELS: dict[str, str] = {
    "tb_terminus_2": "Terminal-Bench 2.0 (terminus-2)",
    "tb_openhands": "Terminal-Bench 2.0 (openhands)",
    "tb_mini_swe_agent": "Terminal-Bench 2.0 (mini-swe-agent)",
    "swe_rebench_swelike": "SWE-rebench Jul 2026 (swelike)",
    "longtail_swe_swelike": "Longtail-SWE (swelike)",
    "tau2_classic": "Sierra tau2 Classic (airline+retail+telecom)",
    "tau2_banking_knowledge_alltools": "Sierra tau2 Banking Knowledge (AllTools)",
}


def assign_capability_band(pass_rate: float) -> str:
    """Map pass rate in [0, 1] into one pre-registered capability band."""
    for label, lo, hi in CAPABILITY_BANDS:
        if lo <= pass_rate < hi:
            return label
    return CAPABILITY_BANDS[-1][0]


@dataclass(slots=True)
class CohortData:
    """Common-task-filtered uniform-scaffold trajectory slice."""

    cohort: str
    dataset: str
    scaffold: str
    step_cap: int | None
    common_tasks: list[str]
    all_records: list[CompactRunRecord]  # includes scheduled 0-step runs on common tasks
    executed_records: list[CompactRunRecord]  # n_steps >= 1 on common tasks


def extract_fair_cohort_datasets(
    *,
    tb_records: list[CompactRunRecord],
    swe_records: list[CompactRunRecord],
    longtail_records: list[CompactRunRecord],
    sierra_records: list[CompactRunRecord],
) -> tuple[dict[str, CohortData], list[CompactRunRecord]]:
    """Build the 7 uniform-scaffold common-task cohorts plus non-AllTools tau2_bk remainder."""
    raw_groups: dict[str, tuple[str, str, list[CompactRunRecord]]] = {
        "tb_terminus_2": (
            "terminalbench",
            "terminus-2",
            [r for r in tb_records if r.harness == "terminus-2"],
        ),
        "tb_openhands": (
            "terminalbench",
            "openhands",
            [r for r in tb_records if r.harness == "openhands"],
        ),
        "tb_mini_swe_agent": (
            "terminalbench",
            "mini-swe-agent",
            [r for r in tb_records if r.harness == "mini-swe-agent"],
        ),
        "swe_rebench_swelike": (
            "swe_rebench",
            "swelike",
            [r for r in swe_records if r.cohort == "swe_rebench_swelike"],
        ),
        "longtail_swe_swelike": (
            "longtail_swe",
            "swelike",
            list(longtail_records),
        ),
        "tau2_classic": (
            "sierra_tau2",
            "tau2_classic",
            [r for r in sierra_records if r.cohort == "tau2_classic"],
        ),
        "tau2_banking_knowledge_alltools": (
            "sierra_tau2",
            "AllTools",
            [r for r in sierra_records if r.cohort == "tau2_banking_knowledge_alltools"],
        ),
    }

    cohorts: dict[str, CohortData] = {}
    for cname in COHORT_ORDER:
        dname, scaf, recs = raw_groups[cname]
        if not recs:
            continue
        tasks_by_model: dict[str, set[str]] = defaultdict(set)
        for r in recs:
            tasks_by_model[r.model].add(r.task_name)
        if not tasks_by_model:
            continue
        common_set = set.intersection(*tasks_by_model.values())
        common_tasks = sorted(common_set)
        kept = [r for r in recs if r.task_name in common_set]
        executed = [r for r in kept if r.n_steps >= 1]
        cohorts[cname] = CohortData(
            cohort=cname,
            dataset=dname,
            scaffold=scaf,
            step_cap=COHORT_STEP_CAPS.get(cname),
            common_tasks=common_tasks,
            all_records=kept,
            executed_records=executed,
        )

    other_bk = [
        r
        for r in sierra_records
        if r.cohort is not None
        and r.cohort.startswith("tau2_bk_")
        and r.cohort != "tau2_banking_knowledge_alltools"
    ]
    return cohorts, other_bk


def _cluster_boot_ci(
    num_mat: np.ndarray,
    den_mat: np.ndarray,
    boot_idx: np.ndarray,
    *,
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """Compute ratio sum(num)/sum(den)*scale and percentile 95% CI from pre-drawn task indices.

    num_mat, den_mat have shape (K,) or (M, K) over K tasks.
    boot_idx has shape (n_boot, K).
    """
    if num_mat.ndim == 1:
        total_den = float(den_mat.sum())
        est = (float(num_mat.sum()) / total_den * scale) if total_den > 0 else float("nan")
        b_num = num_mat[boot_idx].sum(axis=1)
        b_den = den_mat[boot_idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.where(b_den > 0, (b_num / b_den) * scale, np.nan)
    else:
        # Equal model weighting across rows M: for each replicate b, ratio per model m then mean over m
        m_den = den_mat.sum(axis=1)
        valid_m = m_den > 0
        if not np.any(valid_m):
            return float("nan"), float("nan"), float("nan")
        per_m = (num_mat[valid_m].sum(axis=1) / m_den[valid_m]) * scale
        est = float(np.mean(per_m))

        b_num = num_mat[valid_m][:, boot_idx].sum(axis=2)  # (M', B)
        b_den = den_mat[valid_m][:, boot_idx].sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            b_per_m = np.where(b_den > 0, (b_num / b_den) * scale, np.nan)
        # mean across valid models per replicate b
        vals = np.nanmean(b_per_m, axis=0)

    valid_vals = vals[~np.isnan(vals)]
    if valid_vals.size == 0:
        return est, float("nan"), float("nan")
    lo, hi = np.percentile(valid_vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def _prepare_cohort_matrices(
    cohort_data: CohortData,
    records: list[CompactRunRecord],
) -> dict[str, Any]:
    """Pre-aggregate per-(model, task) arrays for instant vectorized task-clustered bootstrapping."""
    models = sorted({r.model for r in records})
    tasks = cohort_data.common_tasks
    m_to_i = {m: i for i, m in enumerate(models)}
    t_to_j = {t: j for j, t in enumerate(tasks)}
    M, K = len(models), len(tasks)

    shape = (M, K)
    runs = np.zeros(shape, dtype=np.float64)
    loop_runs = np.zeros(shape, dtype=np.float64)
    steps = np.zeros(shape, dtype=np.float64)
    tool_steps = np.zeros(shape, dtype=np.float64)
    loop_steps = np.zeros(shape, dtype=np.float64)
    cost_usd = np.zeros(shape, dtype=np.float64)
    loop_cost_usd = np.zeros(shape, dtype=np.float64)
    graded_runs = np.zeros(shape, dtype=np.float64)
    passed_runs = np.zeros(shape, dtype=np.float64)

    for r in records:
        i = m_to_i[r.model]
        j = t_to_j[r.task_name]
        runs[i, j] += 1.0
        if r.has_any_loop:
            loop_runs[i, j] += 1.0
        steps[i, j] += float(r.n_steps)
        tool_steps[i, j] += float(r.n_tool_steps)
        loop_steps[i, j] += float(r.loop_steps)
        if r.cost_included:
            cost_usd[i, j] += float(r.total_cost_usd)
            loop_cost_usd[i, j] += float(r.loop_cost_usd)
        if r.graded:
            graded_runs[i, j] += 1.0
            if r.succeeded:
                passed_runs[i, j] += 1.0

    return {
        "models": models,
        "tasks": tasks,
        "runs": runs,
        "loop_runs": loop_runs,
        "steps": steps,
        "tool_steps": tool_steps,
        "loop_steps": loop_steps,
        "cost_usd": cost_usd,
        "loop_cost_usd": loop_cost_usd,
        "graded_runs": graded_runs,
        "passed_runs": passed_runs,
    }


def _draw_task_boot_indices(n_tasks: int, *, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tasks, size=(n_boot, n_tasks))


def compute_model_scaffold_table(
    cohorts: dict[str, CohortData],
    *,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute per-(cohort, model) loop rates, pass rates, dollar share, and 95% clustered CIs."""
    rows: list[dict[str, Any]] = []
    for c_idx, cname in enumerate(COHORT_ORDER):
        cdata = cohorts.get(cname)
        if cdata is None or not cdata.executed_records:
            continue
        boot_idx = _draw_task_boot_indices(len(cdata.common_tasks), n_boot=n_boot, seed=seed + c_idx * 97)
        mats = _prepare_cohort_matrices(cdata, cdata.executed_records)
        models = mats["models"]

        by_m_all: dict[str, list[CompactRunRecord]] = defaultdict(list)
        by_m_exec: dict[str, list[CompactRunRecord]] = defaultdict(list)
        for r in cdata.all_records:
            by_m_all[r.model].append(r)
        for r in cdata.executed_records:
            by_m_exec[r.model].append(r)

        for i, model in enumerate(models):
            meta = get_model_meta(model)
            all_r = by_m_all[model]
            exec_r = by_m_exec[model]
            litellm_vals = [r.litellm_llm for r in exec_r if r.litellm_llm]
            serving = get_serving_route(
                cdata.dataset,
                model,
                litellm_llm=litellm_vals[0] if litellm_vals else None,
            )

            sched_graded = [r for r in all_r if r.graded]
            sched_pass = (
                sum(1 for r in sched_graded if r.succeeded) / len(sched_graded)
                if sched_graded
                else float("nan")
            )
            exec_graded = [r for r in exec_r if r.graded]
            exec_pass = (
                sum(1 for r in exec_graded if r.succeeded) / len(exec_graded)
                if exec_graded
                else float("nan")
            )
            band = assign_capability_band(exec_pass) if not math.isnan(exec_pass) else "unknown"

            lp100, lp100_lo, lp100_hi = _cluster_boot_ci(
                mats["loop_steps"][i], mats["steps"][i], boot_idx, scale=100.0
            )
            alr, alr_lo, alr_hi = _cluster_boot_ci(
                mats["loop_runs"][i], mats["runs"][i], boot_idx, scale=1.0
            )
            lds, lds_lo, lds_hi = _cluster_boot_ci(
                mats["loop_cost_usd"][i], mats["cost_usd"][i], boot_idx, scale=1.0
            )

            tot_steps = int(mats["steps"][i].sum())
            tot_tool_steps = int(mats["tool_steps"][i].sum())
            tot_loop_steps = int(mats["loop_steps"][i].sum())
            n_exec = len(exec_r)
            n_loop_exec = sum(1 for r in exec_r if r.has_any_loop)

            loop_graded = [r for r in exec_graded if r.has_any_loop]
            clean_graded = [r for r in exec_graded if not r.has_any_loop]
            pass_loop = (
                sum(1 for r in loop_graded if r.succeeded) / len(loop_graded)
                if loop_graded
                else float("nan")
            )
            pass_clean = (
                sum(1 for r in clean_graded if r.succeeded) / len(clean_graded)
                if clean_graded
                else float("nan")
            )

            cost_recs = [r for r in exec_r if r.cost_included]
            cost_sources = sorted({r.cost_source for r in cost_recs if r.cost_source})
            ctxs = sorted({r.context_limit_tokens for r in exec_r if r.context_limit_tokens})

            rows.append(
                {
                    "cohort": cname,
                    "dataset": cdata.dataset,
                    "scaffold": cdata.scaffold,
                    "model": model,
                    "canonical_name": meta.canonical_name,
                    "family": meta.family,
                    "weights": "Open" if meta.is_open_weight else "Closed",
                    "release_month": meta.release_month,
                    "release_months_since_2025": meta.release_months_since_2025,
                    "serving_route": serving,
                    "step_cap": cdata.step_cap if cdata.step_cap is not None else -1,
                    "common_tasks": len(cdata.common_tasks),
                    "scheduled_runs": len(all_r),
                    "executed_runs": n_exec,
                    "zero_step_runs": len(all_r) - n_exec,
                    "graded_runs": len(exec_graded),
                    "executed_pass_rate": exec_pass,
                    "scheduled_pass_rate": sched_pass,
                    "pass_rate_band": band,
                    "total_steps": tot_steps,
                    "total_tool_steps": tot_tool_steps,
                    "loop_steps": tot_loop_steps,
                    "any_loop_runs": n_loop_exec,
                    "loops_per_100_steps": lp100,
                    "loops_per_100_steps_ci_low": lp100_lo,
                    "loops_per_100_steps_ci_high": lp100_hi,
                    "loops_per_100_tool_steps": (100.0 * tot_loop_steps / tot_tool_steps) if tot_tool_steps else float("nan"),
                    "any_loop_rate": alr,
                    "any_loop_rate_ci_low": alr_lo,
                    "any_loop_rate_ci_high": alr_hi,
                    "pass_rate_looping": pass_loop,
                    "pass_rate_clean": pass_clean,
                    "cost_runs": len(cost_recs),
                    "cost_source": "+".join(cost_sources) if cost_sources else "none",
                    "total_cost_usd": float(sum(r.total_cost_usd for r in cost_recs)),
                    "loop_cost_usd": float(sum(r.loop_cost_usd for r in cost_recs)),
                    "loop_dollar_share": lds,
                    "loop_dollar_share_ci_low": lds_lo,
                    "loop_dollar_share_ci_high": lds_hi,
                    "exact_loop_rate": sum(1 for r in exec_r if r.has_exact_loop) / n_exec,
                    "near_duplicate_rate": sum(1 for r in exec_r if r.has_near_duplicate) / n_exec,
                    "aba_thrash_rate": sum(1 for r in exec_r if r.has_aba_thrash) / n_exec,
                    "period_k_rotation_rate": sum(1 for r in exec_r if r.has_rotation_loop) / n_exec,
                    "error_streak_rate": sum(1 for r in exec_r if r.has_error_streak) / n_exec,
                    "redundant_reread_rate": sum(1 for r in exec_r if r.has_redundant_reread) / n_exec,
                    "productive_repeat_rate": sum(1 for r in exec_r if r.has_productive_repeat) / n_exec,
                    "context_limit_tokens_set": ",".join(str(x) for x in ctxs) if ctxs else "",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["cohort", "weights", "executed_pass_rate", "model"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
    return df


def _matched_group_boot(
    num_mat: np.ndarray,
    den_mat: np.ndarray,
    open_mask: np.ndarray,
    model_bands: list[str],
    boot_idx: np.ndarray,
    *,
    scale: float,
) -> dict[str, Any]:
    """Compute unmatched equal-model, run-weighted pooled, and balanced-band-matched Open vs Closed."""

    def _eval_idx(indices: np.ndarray, model_subset_mask: np.ndarray) -> float:
        m_num = num_mat[model_subset_mask][:, indices].sum(axis=1)
        m_den = den_mat[model_subset_mask][:, indices].sum(axis=1)
        valid = m_den > 0
        if not np.any(valid):
            return float("nan")
        return float(np.mean((m_num[valid] / m_den[valid]) * scale))

    # Pre-vectorized replicate evaluator for any subset of models M_sub: (B,)
    def _eval_boot(model_subset_mask: np.ndarray) -> tuple[float, np.ndarray]:
        sub_num = num_mat[model_subset_mask]
        sub_den = den_mat[model_subset_mask]
        pt_den = sub_den.sum(axis=1)
        valid_pt = pt_den > 0
        pt = float(np.mean((sub_num[valid_pt].sum(axis=1) / pt_den[valid_pt]) * scale)) if np.any(valid_pt) else float("nan")

        b_num = sub_num[:, boot_idx].sum(axis=2)  # (m, B)
        b_den = sub_den[:, boot_idx].sum(axis=2)
        b_rates = np.where(b_den > 0, (b_num / b_den) * scale, np.nan)
        mean_arr = np.full(b_rates.shape[1], np.nan, dtype=np.float64)
        valid_cols = np.any(~np.isnan(b_rates), axis=0)
        if np.any(valid_cols):
            mean_arr[valid_cols] = np.nanmean(b_rates[:, valid_cols], axis=0)
        return pt, mean_arr

    pt_open, boot_open = _eval_boot(open_mask)
    pt_closed, boot_closed = _eval_boot(~open_mask)
    unmatched_diff = pt_open - pt_closed
    boot_diff = boot_open - boot_closed
    ud_valid = boot_diff[~np.isnan(boot_diff)]
    ud_lo, ud_hi = (np.percentile(ud_valid, [2.5, 97.5]) if ud_valid.size else (float("nan"), float("nan")))

    # Pooled run-weighted sensitivity (exact numerator/denominator sum across cohort models)
    rw_open_den = float(den_mat[open_mask].sum())
    rw_closed_den = float(den_mat[~open_mask].sum())
    rw_open = float(num_mat[open_mask].sum()) / rw_open_den * scale if rw_open_den > 0 else float("nan")
    rw_closed = float(num_mat[~open_mask].sum()) / rw_closed_den * scale if rw_closed_den > 0 else float("nan")

    # Capability-band matching over overlapping bands (weighted by min(N_open_b, N_closed_b))
    band_rows: list[dict[str, Any]] = []
    matched_num_pt = 0.0
    matched_den_pt = 0.0
    matched_open_num_pt = 0.0
    matched_closed_num_pt = 0.0
    matched_boot_diff = np.zeros(boot_idx.shape[0], dtype=np.float64)
    matched_boot_w = 0.0

    for blabel, _, _ in CAPABILITY_BANDS:
        in_b = np.array([mb == blabel for mb in model_bands], dtype=bool)
        b_open_mask = in_b & open_mask
        b_closed_mask = in_b & (~open_mask)
        n_bo = int(np.sum(b_open_mask))
        n_bc = int(np.sum(b_closed_mask))
        bo_pt, bo_boot = _eval_boot(b_open_mask) if n_bo > 0 else (float("nan"), np.full(boot_idx.shape[0], np.nan))
        bc_pt, bc_boot = _eval_boot(b_closed_mask) if n_bc > 0 else (float("nan"), np.full(boot_idx.shape[0], np.nan))
        overlap = n_bo > 0 and n_bc > 0
        delta_b = bo_pt - bc_pt if overlap else float("nan")
        b_diff_arr = bo_boot - bc_boot if overlap else np.full(boot_idx.shape[0], np.nan)
        b_v = b_diff_arr[~np.isnan(b_diff_arr)]
        b_lo, b_hi = (np.percentile(b_v, [2.5, 97.5]) if b_v.size else (float("nan"), float("nan")))

        w_b = float(min(n_bo, n_bc)) if overlap else 0.0
        if overlap and not math.isnan(delta_b):
            matched_num_pt += w_b * delta_b
            matched_open_num_pt += w_b * bo_pt
            matched_closed_num_pt += w_b * bc_pt
            matched_den_pt += w_b
            matched_boot_diff += w_b * np.where(np.isnan(b_diff_arr), 0.0, b_diff_arr)
            matched_boot_w += w_b

        if n_bo > 0 or n_bc > 0:
            band_rows.append(
                {
                    "band": blabel,
                    "n_open_models": n_bo,
                    "n_closed_models": n_bc,
                    "overlap": overlap,
                    "weight": w_b,
                    "open_mean": bo_pt,
                    "closed_mean": bc_pt,
                    "diff": delta_b,
                    "diff_ci_low": float(b_lo),
                    "diff_ci_high": float(b_hi),
                }
            )

    if matched_den_pt > 0:
        m_open = matched_open_num_pt / matched_den_pt
        m_closed = matched_closed_num_pt / matched_den_pt
        m_diff = matched_num_pt / matched_den_pt
        m_boot_vals = matched_boot_diff / matched_boot_w
        m_lo, m_hi = np.percentile(m_boot_vals, [2.5, 97.5])
    else:
        m_open = m_closed = m_diff = m_lo = m_hi = float("nan")

    o_v = boot_open[~np.isnan(boot_open)]
    c_v = boot_closed[~np.isnan(boot_closed)]
    o_lo, o_hi = (np.percentile(o_v, [2.5, 97.5]) if o_v.size else (float("nan"), float("nan")))
    c_lo, c_hi = (np.percentile(c_v, [2.5, 97.5]) if c_v.size else (float("nan"), float("nan")))

    return {
        "open_unmatched": pt_open,
        "open_unmatched_ci": (float(o_lo), float(o_hi)),
        "closed_unmatched": pt_closed,
        "closed_unmatched_ci": (float(c_lo), float(c_hi)),
        "unmatched_diff": unmatched_diff,
        "unmatched_diff_ci": (float(ud_lo), float(ud_hi)),
        "open_run_weighted": rw_open,
        "closed_run_weighted": rw_closed,
        "open_matched": m_open,
        "closed_matched": m_closed,
        "matched_diff": m_diff,
        "matched_diff_ci": (float(m_lo), float(m_hi)),
        "overlap_bands": sum(1 for b in band_rows if b["overlap"]),
        "bands": band_rows,
    }


def compute_before_after_matching(
    cohorts: dict[str, CohortData],
    model_df: pd.DataFrame,
    *,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
    subset_records_fn: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute unmatched vs band-matched Open-vs-Closed differences per cohort."""
    band_lookup = {
        (row["cohort"], row["model"]): row["pass_rate_band"]
        for _, row in model_df.iterrows()
    }
    out: dict[str, dict[str, Any]] = {}

    for c_idx, cname in enumerate(COHORT_ORDER):
        cdata = cohorts.get(cname)
        if cdata is None or not cdata.executed_records:
            continue
        recs = (
            subset_records_fn(cname, cdata.executed_records)
            if subset_records_fn is not None
            else cdata.executed_records
        )
        if not recs:
            continue

        mats = _prepare_cohort_matrices(cdata, recs)
        models = mats["models"]
        open_mask = np.array([get_model_meta(m).is_open_weight for m in models], dtype=bool)
        if not np.any(open_mask) or not np.any(~open_mask):
            continue

        boot_idx = _draw_task_boot_indices(len(cdata.common_tasks), n_boot=n_boot, seed=seed + c_idx * 97)
        model_bands = [band_lookup.get((cname, m), "unknown") for m in models]

        lp_res = _matched_group_boot(
            mats["loop_steps"], mats["steps"], open_mask, model_bands, boot_idx, scale=100.0
        )
        alr_res = _matched_group_boot(
            mats["loop_runs"], mats["runs"], open_mask, model_bands, boot_idx, scale=1.0
        )
        lds_res = _matched_group_boot(
            mats["loop_cost_usd"], mats["cost_usd"], open_mask, model_bands, boot_idx, scale=1.0
        )

        # Adjacent capability comparison on Longtail-SWE if no strict band overlap or as diagnostic
        out[cname] = {
            "n_open_models": int(np.sum(open_mask)),
            "n_closed_models": int(np.sum(~open_mask)),
            "n_open_runs": int(sum(1 for r in recs if get_model_meta(r.model).is_open_weight)),
            "n_closed_runs": int(sum(1 for r in recs if not get_model_meta(r.model).is_open_weight)),
            "loops_per_100_steps": lp_res,
            "any_loop_rate": alr_res,
            "loop_dollar_share": lds_res,
        }
    return out


def _twoway_cluster_cov(
    bread: np.ndarray,
    scores: np.ndarray,
    clusters_a: np.ndarray,
    clusters_b: np.ndarray,
    n_focal: int,
) -> np.ndarray:
    """Cameron-Gelbach-Miller two-way cluster-robust covariance for top-left n_focal params."""
    n = len(clusters_a)

    def _meat(groups: np.ndarray) -> tuple[np.ndarray, int]:
        uniq, inv = np.unique(groups, return_inverse=True)
        g = len(uniq)
        k = scores.shape[1]
        g_sums = np.zeros((g, k), dtype=np.float64)
        np.add.at(g_sums, inv, scores)
        adj = (g / max(1, g - 1)) * ((n - 1) / max(1, n - scores.shape[1]))
        return adj * (g_sums.T @ g_sums), g

    ma, ga = _meat(clusters_a)
    mb, gb = _meat(clusters_b)
    pair_ids = clusters_a.astype(str) + "||" + clusters_b.astype(str)
    mab, _ = _meat(pair_ids)

    bread_sub = bread[:n_focal, :]
    va = bread_sub @ ma @ bread_sub.T
    vb = bread_sub @ mb @ bread_sub.T
    vab = bread_sub @ mab @ bread_sub.T
    v2 = va + vb - vab

    # PSD clip if numerical intersection slightly negative, and guard against smaller than one-way model cluster SE
    eigvals, eigvecs = np.linalg.eigh(v2)
    eigvals = np.maximum(eigvals, 0.0)
    v2_psd = (eigvecs * eigvals) @ eigvecs.T
    diag_floor = np.maximum(np.diag(va), np.diag(vb))
    for j in range(n_focal):
        if v2_psd[j, j] < diag_floor[j]:
            v2_psd[j, j] = diag_floor[j]
    return v2_psd


def _fit_glm_with_twoway_se(
    *,
    y: np.ndarray,
    exog_focal: pd.DataFrame,
    task_ids: np.ndarray,
    model_ids: np.ndarray,
    family: str,
    offset: np.ndarray | None = None,
    include_task_fe: bool = True,
    drop_invariant_tasks: bool = True,
) -> dict[str, Any]:
    """Fit GLM (optional task fixed effects) and two-way CGM cluster SEs on (model, task)."""
    mask = np.ones(len(y), dtype=bool)
    n_tasks_total = len(np.unique(task_ids))
    if include_task_fe and drop_invariant_tasks:
        # Tasks where y is strictly invariant across all runs cause complete FE separation
        t_df = pd.DataFrame({"t": task_ids, "y": y})
        grp = t_df.groupby("t")["y"]
        if family == "logit":
            stats_t = grp.agg(["min", "max"])
            good_tasks = set(stats_t.index[(stats_t["min"] < 1.0) & (stats_t["max"] > 0.0)])
        else:
            stats_t = grp.sum()
            good_tasks = set(stats_t.index[stats_t > 0.0])
        mask = np.isin(task_ids, list(good_tasks))
        n_tasks_kept = len(good_tasks)
    else:
        n_tasks_kept = n_tasks_total

    if np.sum(mask) < 20 or len(np.unique(model_ids[mask])) < 2:
        return {"converged": False, "reason": "insufficient_informative_observations"}

    y_sub = y[mask]
    xf = exog_focal.iloc[mask].reset_index(drop=True)
    t_sub = task_ids[mask]
    m_sub = model_ids[mask]
    off_sub = offset[mask] if offset is not None else None

    if include_task_fe and n_tasks_kept > 1:
        task_dummies = pd.get_dummies(t_sub, prefix="t", drop_first=True, dtype=np.float64)
        X_full = pd.concat([xf, task_dummies.reset_index(drop=True)], axis=1)
    else:
        X_full = xf
    n_focal = xf.shape[1]

    if family == "negbin":
        # Estimate overdispersion alpha via auxiliary Poisson Pearson residuals
        pois_mod = sm.GLM(y_sub, X_full, family=sm.families.Poisson(), offset=off_sub)
        pois_res = pois_mod.fit(maxiter=100, disp=0)
        mu_p = np.maximum(pois_res.mu, 1e-8)
        pearson_chi2 = float(np.sum(((y_sub - mu_p) ** 2) / mu_p))
        df_resid = max(1, len(y_sub) - X_full.shape[1])
        disp = pearson_chi2 / df_resid
        mean_mu = float(np.mean(mu_p))
        alpha_hat = max(1e-4, min(20.0, (disp - 1.0) / max(1e-6, mean_mu)))
        glm_family = sm.families.NegativeBinomial(alpha=alpha_hat)
        used_family = f"NegBin(alpha={alpha_hat:.3f})"
        mod = sm.GLM(y_sub, X_full, family=glm_family, offset=off_sub)
        res = mod.fit(maxiter=120, disp=0)
    elif family == "poisson":
        mod = sm.GLM(y_sub, X_full, family=sm.families.Poisson(), offset=off_sub)
        res = mod.fit(maxiter=120, disp=0)
        used_family = "Poisson"
    else:
        mod = sm.GLM(y_sub, X_full, family=sm.families.Binomial())
        res = mod.fit(maxiter=120, disp=0)
        used_family = "Logit"

    params = np.asarray(res.params, dtype=np.float64)
    hess = mod.hessian(params)
    bread = np.linalg.pinv(-hess)
    scores = mod.score_obs(params)
    vcov = _twoway_cluster_cov(bread, scores, m_sub, t_sub, n_focal)

    coef_rows: dict[str, dict[str, float]] = {}
    for j, col in enumerate(xf.columns):
        if col == "const":
            continue
        b = float(params[j])
        se = float(math.sqrt(max(1e-18, vcov[j, j])))
        z = b / se
        p = float(2.0 * stats.norm.sf(abs(z)))
        coef_rows[col] = {
            "coef": b,
            "se": se,
            "irr_or": float(math.exp(min(20.0, max(-20.0, b)))),
            "ci_low": float(math.exp(min(20.0, max(-20.0, b - 1.96 * se)))),
            "ci_high": float(math.exp(min(20.0, max(-20.0, b + 1.96 * se)))),
            "z": z,
            "p": p,
        }

    return {
        "converged": bool(getattr(res, "converged", True)),
        "family": used_family,
        "task_fe": bool(include_task_fe and n_tasks_kept > 1),
        "n_obs": int(len(y_sub)),
        "n_models": int(len(np.unique(m_sub))),
        "n_tasks": int(n_tasks_kept),
        "n_tasks_total": int(n_tasks_total),
        "coefs": coef_rows,
    }


def _fit_glmm_and_gee_checks(
    df_reg: pd.DataFrame,
    focal_cols: list[str],
) -> dict[str, Any]:
    """Run GEE (Poisson log-steps offset, task cluster) and BinomialBayesMixedGLM (random task_id)."""
    res: dict[str, Any] = {}
    xf_no_const = df_reg[focal_cols].to_numpy(dtype=np.float64)
    xf_const = np.column_stack([np.ones(len(df_reg), dtype=np.float64), xf_no_const])
    cols_with_const = ["const", *focal_cols]

    try:
        order = np.argsort(df_reg["task_cluster"].to_numpy())
        y_g = df_reg["loop_steps"].to_numpy(dtype=np.float64)[order]
        X_g = xf_const[order]
        off_g = np.log(df_reg["n_steps"].to_numpy(dtype=np.float64)[order])
        grp_g = df_reg["task_cluster"].to_numpy()[order]
        gee_mod = sm.GEE(
            y_g,
            X_g,
            groups=grp_g,
            family=sm.families.Poisson(),
            offset=off_g,
            cov_struct=sm.cov_structs.Exchangeable(),
        )
        gee_fit = gee_mod.fit(maxiter=60)
        j_open = cols_with_const.index("is_open_weight")
        b = float(gee_fit.params[j_open])
        se = float(gee_fit.bse[j_open])
        res["gee_poisson_task"] = {
            "converged": bool(gee_fit.converged),
            "n_obs": int(len(df_reg)),
            "is_open_coef": b,
            "is_open_se_task_robust": se,
            "is_open_irr": float(math.exp(min(20.0, max(-20.0, b)))),
            "p_task_robust": float(2.0 * stats.norm.sf(abs(b / max(1e-12, se)))),
        }
    except Exception as exc:
        res["gee_poisson_task"] = {"converged": False, "error": str(exc)}

    try:
        t_codes, uniq_t = pd.factorize(df_reg["task_cluster"])
        exog_vc = sparse.csc_matrix(
            (np.ones(len(df_reg), dtype=np.float64), (np.arange(len(df_reg)), t_codes)),
            shape=(len(df_reg), len(uniq_t)),
        )
        ident = np.zeros(len(uniq_t), dtype=int)
        bm = BinomialBayesMixedGLM(
            df_reg["any_loop"].to_numpy(dtype=np.float64),
            xf_const,
            exog_vc,
            ident,
            vcp_names=["task_sd"],
            fep_names=cols_with_const,
        )
        bm_fit = bm.fit_vb()
        j_open = cols_with_const.index("is_open_weight")
        b = float(bm_fit.fe_mean[j_open])
        se = float(bm_fit.fe_sd[j_open])
        re_sd = float(math.exp(bm_fit.vcp_mean[0]))
        res["glmm_binomial_random_task"] = {
            "converged": True,
            "n_obs": int(len(df_reg)),
            "n_random_tasks": int(len(uniq_t)),
            "task_re_sd": re_sd,
            "is_open_coef": b,
            "is_open_or": float(math.exp(min(20.0, max(-20.0, b)))),
            "is_open_sd_vb": se,
            "p_vb": float(2.0 * stats.norm.sf(abs(b / max(1e-12, se)))),
        }
    except Exception as exc:
        res["glmm_binomial_random_task"] = {"converged": False, "error": str(exc)}

    return res


def _build_regression_df(
    cohorts: dict[str, CohortData],
    model_df: pd.DataFrame,
    cohort_names: tuple[str, ...],
) -> pd.DataFrame:
    pass_map = {
        (row["cohort"], row["model"]): float(row["executed_pass_rate"])
        for _, row in model_df.iterrows()
    }
    # Task-level empirical pass rate within each cohort as continuous task difficulty covariate
    task_pass_acc: dict[tuple[str, str], list[int]] = defaultdict(list)
    for cname in cohort_names:
        cdata = cohorts.get(cname)
        if cdata is None:
            continue
        for r in cdata.executed_records:
            if r.graded:
                task_pass_acc[(cname, r.task_name)].append(1 if r.succeeded else 0)
    task_diff_map = {
        k: (sum(v) / len(v) if v else 0.5)
        for k, v in task_pass_acc.items()
    }

    rows: list[dict[str, Any]] = []
    for cname in cohort_names:
        cdata = cohorts.get(cname)
        if cdata is None:
            continue
        cap_val = float(cdata.step_cap) if cdata.step_cap is not None else 500.0
        for r in cdata.executed_records:
            meta = get_model_meta(r.model)
            serving = get_serving_route(cdata.dataset, r.model, litellm_llm=r.litellm_llm)
            mpr = pass_map.get((cname, r.model), 0.0)
            tpr = task_diff_map.get((cname, r.task_name), 0.5)
            rows.append(
                {
                    "cohort": cname,
                    "model_cluster": f"{cname}::{r.model}" if len(cohort_names) > 1 else r.model,
                    "task_cluster": f"{cname}::{r.task_name}" if len(cohort_names) > 1 else r.task_name,
                    "loop_steps": r.loop_steps,
                    "any_loop": 1 if r.has_any_loop else 0,
                    "n_steps": r.n_steps,
                    "log1p_steps": math.log1p(r.n_steps),
                    "is_open_weight": 1.0 if meta.is_open_weight else 0.0,
                    "pass_rate_10pp": mpr * 10.0,
                    "task_pass_10pp": tpr * 10.0,
                    "release_months": float(meta.release_months_since_2025),
                    "is_third_party": 1.0 if serving == "third_party" else 0.0,
                    "step_cap_100": cap_val / 100.0,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pass_rate_10pp_c"] = df["pass_rate_10pp"] - df["pass_rate_10pp"].mean()
        df["task_pass_10pp_c"] = df["task_pass_10pp"] - df["task_pass_10pp"].mean()
        df["release_months_c"] = df["release_months"] - df["release_months"].mean()
        df["log1p_steps_c"] = df["log1p_steps"] - df["log1p_steps"].mean()
        df["step_cap_100_c"] = df["step_cap_100"] - df["step_cap_100"].mean()
    return df


def _select_identifiable_covariates(df: pd.DataFrame, candidates: list[str]) -> tuple[list[str], list[str]]:
    """Keep linearly independent predictors in candidates; return (kept, omitted_reasons)."""
    kept: list[str] = []
    omitted: list[str] = []
    X_cur = np.ones((len(df), 1), dtype=np.float64)
    for col in candidates:
        vals = df[col].to_numpy(dtype=np.float64)
        if np.std(vals) < 1e-9:
            omitted.append(f"{col} (constant)")
            continue
        X_cand = np.column_stack([X_cur, vals])
        beta, *_ = np.linalg.lstsq(X_cur, vals, rcond=None)
        resid = vals - X_cur @ beta
        r2 = 1.0 - float(np.var(resid) / max(1e-12, np.var(vals)))
        if r2 > 0.995 or np.linalg.matrix_rank(X_cand) < X_cand.shape[1]:
            omitted.append(f"{col} (collinear R2={r2:.3f})")
            continue
        kept.append(col)
        X_cur = X_cand
    return kept, omitted


def fit_cohort_and_pooled_regressions(
    cohorts: dict[str, CohortData],
    model_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit Negative Binomial (log steps offset), Logit, GEE Poisson, and Binomial GLMM."""
    results: dict[str, Any] = {}

    for cname in COHORT_ORDER:
        if cname not in cohorts:
            continue
        df = _build_regression_df(cohorts, model_df, (cname,))
        if df.empty or df["is_open_weight"].nunique() < 2:
            continue
        base_cands = ["is_open_weight", "pass_rate_10pp_c", "release_months_c", "is_third_party"]
        if cname == "swe_rebench_swelike":
            kept, omitted = _select_identifiable_covariates(
                df, ["is_open_weight", "pass_rate_10pp_c", "release_months_c"]
            )
            omitted.append("is_third_party (uniform gateway)")
        else:
            kept, omitted = _select_identifiable_covariates(df, base_cands)

        xf = pd.DataFrame({"const": 1.0, **{k: df[k] for k in kept}})
        xf_exp = xf.copy()
        if float(df["log1p_steps_c"].std()) > 1e-9:
            xf_exp["log1p_steps_c"] = df["log1p_steps_c"]

        nb_res = _fit_glm_with_twoway_se(
            y=df["loop_steps"].to_numpy(dtype=np.float64),
            exog_focal=xf,
            task_ids=df["task_cluster"].to_numpy(),
            model_ids=df["model_cluster"].to_numpy(),
            family="negbin",
            offset=np.log(df["n_steps"].to_numpy(dtype=np.float64)),
            include_task_fe=True,
        )
        logit_res = _fit_glm_with_twoway_se(
            y=df["any_loop"].to_numpy(dtype=np.float64),
            exog_focal=xf,
            task_ids=df["task_cluster"].to_numpy(),
            model_ids=df["model_cluster"].to_numpy(),
            family="logit",
            include_task_fe=True,
        )
        logit_exp_res = _fit_glm_with_twoway_se(
            y=df["any_loop"].to_numpy(dtype=np.float64),
            exog_focal=xf_exp,
            task_ids=df["task_cluster"].to_numpy(),
            model_ids=df["model_cluster"].to_numpy(),
            family="logit",
            include_task_fe=True,
        )
        glmm_gee = _fit_glmm_and_gee_checks(df, kept)

        results[cname] = {
            "omitted_covariates": omitted,
            "negbin_offset": nb_res,
            "logit": logit_res,
            "logit_with_steps": logit_exp_res,
            **glmm_gee,
        }

    # Pooled within-scaffold fits across all active cohorts:
    # Spec A: Cohort FE + task_pass_10pp_c + two-way (model, task) cluster SEs + GEE/GLMM random task RE
    # Spec B: Explicit step_cap_100_c without Cohort FE (avoiding cohort-step_cap collinearity)
    active_cohorts = tuple(c for c in COHORT_ORDER if c in cohorts)
    df_pool = _build_regression_df(cohorts, model_df, active_cohorts)
    if not df_pool.empty and df_pool["is_open_weight"].nunique() == 2:
        cohort_dummies = pd.get_dummies(df_pool["cohort"], prefix="c", drop_first=True, dtype=np.float64)
        pool_focal, omitted_pool = _select_identifiable_covariates(
            df_pool,
            ["is_open_weight", "pass_rate_10pp_c", "release_months_c", "is_third_party", "task_pass_10pp_c"],
        )
        omitted_pool.append("step_cap_100_c (absorbed by cohort FE; tested separately in step_cap spec)")
        xf_pool = pd.concat(
            [pd.DataFrame({"const": 1.0, **{k: df_pool[k] for k in pool_focal}}), cohort_dummies.reset_index(drop=True)],
            axis=1,
        )
        xf_pool_exp = xf_pool.copy()
        xf_pool_exp["log1p_steps_c"] = df_pool["log1p_steps_c"].to_numpy()

        nb_pool = _fit_glm_with_twoway_se(
            y=df_pool["loop_steps"].to_numpy(dtype=np.float64),
            exog_focal=xf_pool,
            task_ids=df_pool["task_cluster"].to_numpy(),
            model_ids=df_pool["model_cluster"].to_numpy(),
            family="negbin",
            offset=np.log(df_pool["n_steps"].to_numpy(dtype=np.float64)),
            include_task_fe=False,
        )
        logit_pool = _fit_glm_with_twoway_se(
            y=df_pool["any_loop"].to_numpy(dtype=np.float64),
            exog_focal=xf_pool,
            task_ids=df_pool["task_cluster"].to_numpy(),
            model_ids=df_pool["model_cluster"].to_numpy(),
            family="logit",
            include_task_fe=False,
        )
        logit_pool_exp = _fit_glm_with_twoway_se(
            y=df_pool["any_loop"].to_numpy(dtype=np.float64),
            exog_focal=xf_pool_exp,
            task_ids=df_pool["task_cluster"].to_numpy(),
            model_ids=df_pool["model_cluster"].to_numpy(),
            family="logit",
            include_task_fe=False,
        )
        glmm_gee_pool = _fit_glmm_and_gee_checks(
            pd.concat([df_pool.reset_index(drop=True), cohort_dummies.reset_index(drop=True)], axis=1),
            [*pool_focal, *cohort_dummies.columns],
        )

        # Spec B: replace cohort FE with explicit scaffold step_cap_100_c when multiple step caps exist
        cap_cols, _ = _select_identifiable_covariates(
            df_pool,
            ["is_open_weight", "pass_rate_10pp_c", "release_months_c", "is_third_party", "step_cap_100_c", "task_pass_10pp_c"],
        )
        xf_cap = pd.DataFrame({"const": 1.0, **{k: df_pool[k] for k in cap_cols}})
        nb_cap = _fit_glm_with_twoway_se(
            y=df_pool["loop_steps"].to_numpy(dtype=np.float64),
            exog_focal=xf_cap,
            task_ids=df_pool["task_cluster"].to_numpy(),
            model_ids=df_pool["model_cluster"].to_numpy(),
            family="negbin",
            offset=np.log(df_pool["n_steps"].to_numpy(dtype=np.float64)),
            include_task_fe=False,
        )

        results["pooled_within_scaffold"] = {
            "omitted_covariates": omitted_pool,
            "negbin_offset": nb_pool,
            "logit": logit_pool,
            "logit_with_steps": logit_pool_exp,
            "negbin_step_cap_spec": nb_cap,
            **glmm_gee_pool,
        }

    return results


def _family_display(family: str, is_open: bool) -> str:
    if family == "qwen":
        return "qwen (open)" if is_open else "qwen (closed Max)"
    return family


def compute_sensitivity_analyses(
    cohorts: dict[str, CohortData],
    model_df: pd.DataFrame,
    sierra_other_bk: list[CompactRunRecord],
    *,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Compute all 5 pre-registered sensitivity checks plus Longtail context & Gemini dossiers."""
    pass_only = compute_before_after_matching(
        cohorts,
        model_df,
        n_boot=n_boot,
        seed=seed + 11,
        subset_records_fn=lambda c, recs: [r for r in recs if r.graded and r.succeeded],
    )
    fail_only = compute_before_after_matching(
        cohorts,
        model_df,
        n_boot=n_boot,
        seed=seed + 23,
        subset_records_fn=lambda c, recs: [r for r in recs if r.graded and not r.succeeded],
    )
    drop_gpt_oss = compute_before_after_matching(
        cohorts,
        model_df,
        n_boot=n_boot,
        seed=seed + 37,
        subset_records_fn=lambda c, recs: [r for r in recs if "gpt-oss-" not in r.model.lower()],
    )
    first_party_only = compute_before_after_matching(
        cohorts,
        model_df,
        n_boot=n_boot,
        seed=seed + 53,
        subset_records_fn=lambda c, recs: [
            r
            for r in recs
            if get_serving_route(cohorts[c].dataset, r.model, litellm_llm=r.litellm_llm) == "first_party"
        ],
    )

    # 5. Per-family summary across cohorts (showing every vendor family incl. Google Gemini)
    fam_rows: list[dict[str, Any]] = []
    for cname in COHORT_ORDER:
        cdata = cohorts.get(cname)
        if cdata is None or not cdata.executed_records:
            continue
        sub_df = model_df[model_df["cohort"] == cname].copy()
        sub_df["family_group"] = [
            _family_display(str(r["family"]), r["weights"] == "Open")
            for _, r in sub_df.iterrows()
        ]
        for (fg, w), grp in sub_df.groupby(["family_group", "weights"], sort=True):
            cost_sub = grp[grp["cost_runs"] > 0]
            fam_rows.append(
                {
                    "cohort": cname,
                    "family_group": fg,
                    "weights": w,
                    "models": int(len(grp)),
                    "model_list": ", ".join(grp["canonical_name"].tolist()),
                    "executed_runs": int(grp["executed_runs"].sum()),
                    "mean_pass_rate": float(grp["executed_pass_rate"].mean()),
                    "mean_loops_per_100": float(grp["loops_per_100_steps"].mean()),
                    "mean_any_loop_rate": float(grp["any_loop_rate"].mean()),
                    "pooled_loops_per_100": 100.0 * float(grp["loop_steps"].sum()) / max(1, int(grp["total_steps"].sum())),
                    "pooled_any_loop_rate": float(grp["any_loop_runs"].sum()) / max(1, int(grp["executed_runs"].sum())),
                    "mean_loop_dollar_share": float(cost_sub["loop_dollar_share"].mean()) if not cost_sub.empty else float("nan"),
                }
            )
    fam_df = pd.DataFrame(fam_rows)

    # Longtail context-limit confounder breakdown (262K/131K vs 1M) & adjacent pair (kimi-k3 vs gpt-5.6-luna)
    lt_rows = model_df[model_df["cohort"] == "longtail_swe_swelike"].copy()

    # Non-AllTools Sierra banking_knowledge summary (incl. gemini-3-flash / gemini-3-pro terminal_use)
    other_bk_rows: list[dict[str, Any]] = []
    by_cohort_model: dict[tuple[str, str], list[CompactRunRecord]] = defaultdict(list)
    for r in sierra_other_bk:
        if r.n_steps >= 1:
            by_cohort_model[(str(r.cohort), r.model)].append(r)
    for (co, mod), recs in sorted(by_cohort_model.items()):
        meta = get_model_meta(mod)
        tot_s = sum(r.n_steps for r in recs)
        lp_s = sum(r.loop_steps for r in recs)
        other_bk_rows.append(
            {
                "cohort": co,
                "retrieval_config": recs[0].retrieval_config or "",
                "model": mod,
                "family": meta.family,
                "weights": "Open" if meta.is_open_weight else "Closed",
                "executed_runs": len(recs),
                "pass_rate": sum(1 for r in recs if r.succeeded) / len(recs),
                "loops_per_100_steps": 100.0 * lp_s / tot_s if tot_s else 0.0,
                "any_loop_rate": sum(1 for r in recs if r.has_any_loop) / len(recs),
            }
        )

    return {
        "pass_only": pass_only,
        "fail_only": fail_only,
        "drop_gpt_oss": drop_gpt_oss,
        "first_party_only": first_party_only,
        "per_family_df": fam_df,
        "longtail_models_df": lt_rows,
        "non_alltools_bk_df": pd.DataFrame(other_bk_rows),
    }


def _fmt_pct(v: float | None, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100.0:.{digits}f}%"


def _fmt_num(v: float | None, digits: int = 2, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(v)


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def build_open_vs_closed_markdown(
    *,
    cohorts: dict[str, CohortData],
    model_df: pd.DataFrame,
    matching: dict[str, dict[str, Any]],
    regressions: dict[str, Any],
    sensitivities: dict[str, Any],
) -> str:
    """Render results/open_vs_closed_model.md directly from computed statistical structures."""
    md: list[str] = []
    md.append("# Open-Weights vs Closed Models: Fair Within-Scaffold Loop Analysis\n")
    md.append(
        "Generated programmatically (CPU-only, fixed seed `42`, `1,000` task-clustered bootstrap replicates, "
        "unchanged step labeler). Negative differences (`Open - Closed < 0`) indicate **reversed** gaps where "
        "Open-Weights looped *less* than Closed models; confidence intervals crossing `0` are reported neutrally as **null**.\n"
    )

    md.append("## 1. Fair Uniform-Scaffold Cohorts & Denominators\n")
    md.append(
        "| Cohort | Dataset & Scaffold | Shared Tasks | Open Models (Exec Runs) | Closed Models (Exec Runs) | Zero-Step Crashes | Step Cap |"
    )
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for cname in COHORT_ORDER:
        cdata = cohorts.get(cname)
        mres = matching.get(cname)
        if not cdata or not mres:
            continue
        zstep = len(cdata.all_records) - len(cdata.executed_records)
        cap_s = str(cdata.step_cap) if cdata.step_cap is not None else "none observed"
        md.append(
            f"| `{cname}` | {COHORT_LABELS[cname]} | {len(cdata.common_tasks)} | "
            f"{mres['n_open_models']} ({mres['n_open_runs']:,}) | "
            f"{mres['n_closed_models']} ({mres['n_closed_runs']:,}) | {zstep} | {cap_s} |"
        )
    md.append("")
    md.append(
        "- Excluded from Open-vs-Closed pooling: custom closed TB harnesses (`claude-code`, `codex`, `gemini-cli`, etc.), "
        "TB null-step rows (`glm-4.7@z-ai`, `gpt-5.3-codex@openai`, `kimi-k2.5@kimi`), SWE-rebench agentic harnesses "
        "(`claude-code`, `codex`, `cursor`, `junie`), Sierra `gpt-5_sierra_2025-08-09` (`gpt-4.1` user simulator), "
        "and the 6 non-`AllTools` Sierra `banking_knowledge` submissions (`terminal_use` / `embeddings`).\n"
    )

    md.append("## 2. Before vs After Capability-Band Matching (Equal-Model Cohort Means, 95% Task-Clustered CI)\n")
    md.append("### 2A. Loops per 100 Steps & Any-Loop Run Rate (`Open - Closed`)\n")
    md.append(
        "| Cohort | Unmatched Loops/100 (Open vs Closed) | Unmatched Gap [95% CI] | Overlap Bands | "
        "Matched Gap Loops/100 [95% CI] | Unmatched Any-Loop Gap (pp) | Matched Any-Loop Gap (pp) [95% CI] |"
    )
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cname in COHORT_ORDER:
        if cname not in matching:
            continue
        lp = matching[cname]["loops_per_100_steps"]
        al = matching[cname]["any_loop_rate"]
        u_gap = f"{_fmt_num(lp['unmatched_diff'], 2, True)} [{_fmt_num(lp['unmatched_diff_ci'][0], 2, True)}, {_fmt_num(lp['unmatched_diff_ci'][1], 2, True)}]"
        if lp["overlap_bands"] > 0:
            m_gap = f"{_fmt_num(lp['matched_diff'], 2, True)} [{_fmt_num(lp['matched_diff_ci'][0], 2, True)}, {_fmt_num(lp['matched_diff_ci'][1], 2, True)}]"
            al_m_gap = f"{_fmt_num(al['matched_diff']*100, 2, True)} [{_fmt_num(al['matched_diff_ci'][0]*100, 2, True)}, {_fmt_num(al['matched_diff_ci'][1]*100, 2, True)}]"
        else:
            m_gap = "no band overlap"
            al_m_gap = "no band overlap"
        md.append(
            f"| `{cname}` | {_fmt_num(lp['open_unmatched'])} vs {_fmt_num(lp['closed_unmatched'])} | "
            f"{u_gap} | {lp['overlap_bands']} | {m_gap} | "
            f"{_fmt_num(al['unmatched_diff']*100, 2, True)} | {al_m_gap} |"
        )
    md.append("")

    md.append("### 2B. Capability-Band Breakdown (Overlapping Pass-Rate Strata)\n")
    md.append("| Cohort | Pass-Rate Band | Open Models | Closed Models | Loops/100 (Open vs Closed) | Gap [95% CI] | Any-Loop Gap (pp) |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for cname in COHORT_ORDER:
        if cname not in matching:
            continue
        lp_bands = matching[cname]["loops_per_100_steps"]["bands"]
        al_bands = {b["band"]: b for b in matching[cname]["any_loop_rate"]["bands"]}
        for b in lp_bands:
            if not b["overlap"]:
                continue
            ab = al_bands[b["band"]]
            md.append(
                f"| `{cname}` | {b['band']} | {b['n_open_models']} | {b['n_closed_models']} | "
                f"{_fmt_num(b['open_mean'])} vs {_fmt_num(b['closed_mean'])} | "
                f"{_fmt_num(b['diff'], 2, True)} [{_fmt_num(b['diff_ci_low'], 2, True)}, {_fmt_num(b['diff_ci_high'], 2, True)}] | "
                f"{_fmt_num(ab['diff']*100, 2, True)} |"
            )
    md.append("")

    md.append("## 3. Statistical Regression Models (Adjusted for Task, Capability, Vintage & Serving Route)\n")
    md.append(
        "Count models fit **Negative Binomial GLM** with `log(n_steps)` offset (`IRR > 1` means higher loop rate per step for Open); "
        "binary models fit **Logit** for `any_loop` (`OR > 1` means higher probability of >=1 loop), with task fixed effects and "
        "**two-way Cameron-Gelbach-Miller cluster-robust SEs** on `(model, task)`. Hierarchical checks add task-clustered Poisson GEE "
        "and sparse random-intercept `BinomialBayesMixedGLM` (`1 | task_id`) over 100% of executed trajectories.\n"
    )
    md.append(
        "| Cohort | NB Offset `is_open` IRR [95% CI] | p (2-way) | Logit FE `is_open` OR [95% CI] | p (2-way) | "
        "Logit+Steps OR (p) | PassRate (+10pp) IRR (p) | Omitted Covariates |"
    )
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for key in [*COHORT_ORDER, "pooled_within_scaffold"]:
        reg = regressions.get(key)
        if not reg:
            continue
        nb = reg.get("negbin_offset", {})
        lg = reg.get("logit", {})
        lg_e = reg.get("logit_with_steps", {})
        om = ", ".join(reg.get("omitted_covariates") or []) or "none"

        def _fmt_irr_ci(r: dict[str, Any], var: str = "is_open_weight") -> tuple[str, str]:
            c = (r.get("coefs") or {}).get(var)
            if not c:
                return "n/a", "n/a"
            return f"{c['irr_or']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}]", _fmt_p(c["p"])

        nb_s, nb_p = _fmt_irr_ci(nb, "is_open_weight")
        lg_s, lg_p = _fmt_irr_ci(lg, "is_open_weight")
        le_c = (lg_e.get("coefs") or {}).get("is_open_weight")
        le_s = f"{le_c['irr_or']:.2f} ({_fmt_p(le_c['p'])})" if le_c else "n/a"
        pr_c = (nb.get("coefs") or {}).get("pass_rate_10pp_c")
        pr_s = f"{pr_c['irr_or']:.2f} ({_fmt_p(pr_c['p'])})" if pr_c else "n/a"
        md.append(f"| `{key}` | {nb_s} | {nb_p} | {lg_s} | {lg_p} | {le_s} | {pr_s} | {om} |")
    md.append("")

    md.append("### 3B. Additional Covariates & Mixed-Effects / GEE Task-Cluster Robustness\n")
    md.append(
        "| Cohort | NB Release (+1mo) IRR (p) | NB Third-Party IRR (p) | Poisson GEE `is_open` IRR (p_task) | "
        "Binomial GLMM Random-Task OR (p_vb) | Task RE SD |"
    )
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for key in [*COHORT_ORDER, "pooled_within_scaffold"]:
        reg = regressions.get(key)
        if not reg:
            continue
        nb = reg.get("negbin_offset", {})
        rel_c = (nb.get("coefs") or {}).get("release_months_c")
        tp_c = (nb.get("coefs") or {}).get("is_third_party")
        rel_s = f"{rel_c['irr_or']:.3f} ({_fmt_p(rel_c['p'])})" if rel_c else "omitted"
        tp_s = f"{tp_c['irr_or']:.2f} ({_fmt_p(tp_c['p'])})" if tp_c else "omitted"
        gee = reg.get("gee_poisson_task", {})
        glmm = reg.get("glmm_binomial_random_task", {})
        gee_s = f"{gee['is_open_irr']:.2f} ({_fmt_p(gee['p_task_robust'])})" if gee.get("converged") else "failed"
        if glmm.get("converged"):
            glmm_s = f"{glmm['is_open_or']:.2f} ({_fmt_p(glmm['p_vb'])})"
            sd_s = f"{glmm['task_re_sd']:.2f}"
        else:
            glmm_s, sd_s = "failed", "n/a"
        md.append(f"| `{key}` | {rel_s} | {tp_s} | {gee_s} | {glmm_s} | {sd_s} |")
    md.append("")

    pooled_cap = (regressions.get("pooled_within_scaffold") or {}).get("negbin_step_cap_spec", {})
    if pooled_cap.get("coefs"):
        sc_c = (pooled_cap["coefs"]).get("step_cap_100_c")
        if sc_c:
            md.append(
                f"- **Pooled step-cap covariate (separate non-cohort-FE specification)**: `+100` step cap IRR = "
                f"**{sc_c['irr_or']:.2f}** (`[{sc_c['ci_low']:.2f}, {sc_c['ci_high']:.2f}]`, `p={_fmt_p(sc_c['p'])}`), "
                f"while `is_open_weight` IRR = **{pooled_cap['coefs']['is_open_weight']['irr_or']:.2f}** "
                f"(`p={_fmt_p(pooled_cap['coefs']['is_open_weight']['p'])}`).\n"
            )

    md.append("## 4. Pre-Registered Sensitivity Checks\n")
    md.append("### 4A. Pass-Only, Fail-Only, Drop `gpt-oss`, and First-Party-Only (Open - Closed Gap, Loops/100 Steps [95% CI])\n")
    md.append("| Cohort | Baseline Matched | `pass_only` (Matched) | `fail_only` (Matched) | `drop_gpt_oss` (Matched) | `first_party_only` (Matched / Unmatched) |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")

    def _sens_cell(d: dict[str, Any], cname: str, allow_unmatched: bool = False) -> str:
        entry = d.get(cname)
        if not entry:
            return "n/a (single weights)"
        lp = entry["loops_per_100_steps"]
        if lp["overlap_bands"] > 0:
            return f"{_fmt_num(lp['matched_diff'], 2, True)} [{_fmt_num(lp['matched_diff_ci'][0], 2, True)}, {_fmt_num(lp['matched_diff_ci'][1], 2, True)}]"
        if allow_unmatched:
            return f"no overlap (unmatched {_fmt_num(lp['unmatched_diff'], 2, True)})"
        return f"no band overlap (unmatched {_fmt_num(lp['unmatched_diff'], 2, True)})"

    for cname in COHORT_ORDER:
        if cname not in matching:
            continue
        md.append(
            f"| `{cname}` | {_sens_cell(matching, cname)} | "
            f"{_sens_cell(sensitivities['pass_only'], cname)} | "
            f"{_sens_cell(sensitivities['fail_only'], cname)} | "
            f"{_sens_cell(sensitivities['drop_gpt_oss'], cname)} | "
            f"{_sens_cell(sensitivities['first_party_only'], cname, allow_unmatched=True)} |"
        )
    md.append("")

    md.append("### 4B. Per-Family Breakdown Across All Cohorts (Every Vendor Family Including Google Gemini)\n")
    md.append("| Cohort | Family | Weights | Models | Exec Runs | Mean Pass | Mean Loops/100 | Mean Any-Loop | Mean Loop $ Share |")
    md.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    fam_df: pd.DataFrame = sensitivities["per_family_df"]
    for _, r in fam_df.sort_values(["cohort", "mean_loops_per_100"], ascending=[True, False]).iterrows():
        md.append(
            f"| `{r['cohort']}` | `{r['family_group']}` | {r['weights']} | {int(r['models'])} | "
            f"{int(r['executed_runs']):,} | {_fmt_pct(r['mean_pass_rate'])} | "
            f"{_fmt_num(r['mean_loops_per_100'])} | {_fmt_pct(r['mean_any_loop_rate'])} | "
            f"{_fmt_pct(r['mean_loop_dollar_share'])} |"
        )
    md.append("")

    md.append("### 4C. Google Gemini Detailed Profile Across Uniform Scaffolds & Sierra Variants\n")
    gem_rows = model_df[model_df["family"] == "gemini"].copy()
    if not gem_rows.empty:
        md.append("| Cohort / Scaffold | Model | Pass Rate | Loops/100 [95% CI] | Any-Loop Rate [95% CI] | Loop $ Share |")
        md.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for _, r in gem_rows.sort_values(["cohort", "model"]).iterrows():
            md.append(
                f"| `{r['cohort']}` | `{r['model']}` | {_fmt_pct(r['executed_pass_rate'])} | "
                f"{_fmt_num(r['loops_per_100_steps'])} [{_fmt_num(r['loops_per_100_steps_ci_low'])}, {_fmt_num(r['loops_per_100_steps_ci_high'])}] | "
                f"{_fmt_pct(r['any_loop_rate'])} [{_fmt_pct(r['any_loop_rate_ci_low'])}, {_fmt_pct(r['any_loop_rate_ci_high'])}] | "
                f"{_fmt_pct(r['loop_dollar_share'])} |"
            )
    obk_df: pd.DataFrame = sensitivities["non_alltools_bk_df"]
    gem_bk = obk_df[obk_df["family"] == "gemini"] if not obk_df.empty else pd.DataFrame()
    if not gem_bk.empty:
        for _, r in gem_bk.iterrows():
            md.append(
                f"| `{r['cohort']}` (`{r['retrieval_config']}`, excluded from AllTools) | `{r['model']}` | "
                f"{_fmt_pct(r['pass_rate'])} | {_fmt_num(r['loops_per_100_steps'])} | {_fmt_pct(r['any_loop_rate'])} | n/a |"
            )
    md.append("")

    md.append("## 5. Open vs Closed Dollar Share & Longtail Context Confounders\n")
    md.append("| Cohort | Open Mean Loop $ Share (Cost-Valid) | Closed Mean Loop $ Share (Cost-Valid) | Cost Provenance Note |")
    md.append("| --- | ---: | ---: | --- |")
    for cname in COHORT_ORDER:
        if cname not in matching:
            continue
        sub = model_df[(model_df["cohort"] == cname) & (model_df["cost_runs"] > 0)]
        o_lds = sub[sub["weights"] == "Open"]["loop_dollar_share"].mean() if not sub[sub["weights"] == "Open"].empty else float("nan")
        c_lds = sub[sub["weights"] == "Closed"]["loop_dollar_share"].mean() if not sub[sub["weights"] == "Closed"].empty else float("nan")
        provs = ", ".join(sorted(sub["cost_source"].unique())) if not sub.empty else "none"
        md.append(f"| `{cname}` | {_fmt_pct(o_lds)} | {_fmt_pct(c_lds)} | `{provs}` |")
    md.append("")

    lt_df: pd.DataFrame = sensitivities["longtail_models_df"]
    if not lt_df.empty:
        md.append("### Longtail-SWE Context Limit & Adjacent Frontier Pair Check (`swelike`)\n")
        md.append("| Model | Weights | Context Limit | Exec Runs (Zero-Step Drops) | Exec Pass | Loops/100 | Any-Loop Rate | Loop $ Share |")
        md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in lt_df.sort_values("executed_pass_rate", ascending=False).iterrows():
            md.append(
                f"| `{r['model']}` | {r['weights']} | `{r['context_limit_tokens_set']}` | "
                f"{int(r['executed_runs'])} ({int(r['zero_step_runs'])}) | {_fmt_pct(r['executed_pass_rate'])} | "
                f"{_fmt_num(r['loops_per_100_steps'])} | {_fmt_pct(r['any_loop_rate'])} | {_fmt_pct(r['loop_dollar_share'])} |"
            )
        md.append("")

    if not obk_df.empty:
        md.append("### Sierra `banking_knowledge` Non-`AllTools` Submissions (Excluded From Uniform `AllTools` Slice)\n")
        md.append("| Cohort | Retrieval Scaffold | Model | Weights | Exec Runs | Pass Rate | Loops/100 | Any-Loop Rate |")
        md.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for _, r in obk_df.iterrows():
            md.append(
                f"| `{r['cohort']}` | `{r['retrieval_config']}` | `{r['model']}` | {r['weights']} | "
                f"{int(r['executed_runs'])} | {_fmt_pct(r['pass_rate'])} | {_fmt_num(r['loops_per_100_steps'])} | {_fmt_pct(r['any_loop_rate'])} |"
            )
        md.append("")

    return "\n".join(md) + "\n"


def run_fair_open_vs_closed_study(
    *,
    results_dir: Path,
    tb_records: list[CompactRunRecord],
    swe_records: list[CompactRunRecord],
    longtail_records: list[CompactRunRecord],
    sierra_records: list[CompactRunRecord],
) -> dict[str, Any]:
    """Run full Phase 2 fair comparison, save CSV & Markdown outputs, and return structured results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cohorts, other_bk = extract_fair_cohort_datasets(
        tb_records=tb_records,
        swe_records=swe_records,
        longtail_records=longtail_records,
        sierra_records=sierra_records,
    )
    model_df = compute_model_scaffold_table(cohorts)
    matching = compute_before_after_matching(cohorts, model_df)
    regressions = fit_cohort_and_pooled_regressions(cohorts, model_df)
    sensitivities = compute_sensitivity_analyses(cohorts, model_df, other_bk)

    csv_path = results_dir / "loops_by_model_scaffold.csv"
    model_df.to_csv(csv_path, index=False)

    md_text = build_open_vs_closed_markdown(
        cohorts=cohorts,
        model_df=model_df,
        matching=matching,
        regressions=regressions,
        sensitivities=sensitivities,
    )
    md_path = results_dir / "open_vs_closed_model.md"
    md_path.write_text(md_text)

    return {
        "cohorts": cohorts,
        "other_bk": other_bk,
        "model_df": model_df,
        "matching": matching,
        "regressions": regressions,
        "sensitivities": sensitivities,
        "csv_path": csv_path,
        "md_path": md_path,
    }

