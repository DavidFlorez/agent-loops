"""Main offline study pipeline across Terminal-Bench, SWE-rebench, and tau2-bench."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import glob
import gzip
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq

from agent_loops.canonicalize import (
    NormalizedStep,
    collapse_openhands_tb_steps,
    is_rsc_ref,
    make_step,
)
from agent_loops.cost import (
    detect_compaction_or_subagents,
    rebuild_trajectory_costs,
)
from agent_loops.detectors import (
    DetectorResult,
    ReplayTrajectory,
    compute_harness_turn_budgets,
    detect_cline_style,
    detect_gemini_cli,
    detect_openhands_stuck,
    make_openclaw_detector,
    make_step_cap_detector,
    make_turn_budget_detector,
    run_matched_random_baseline,
)
from agent_loops.labeler import ALL_STEP_LABELS, PATHOLOGICAL_LOOP_LABELS, label_trajectory

STEP_CAP_VALUES = (30, 50, 75, 100, 150)

# Thresholds for fast per-trajectory evaluation
_OC10 = make_openclaw_detector(10, window=30)
_OC20 = make_openclaw_detector(20, window=30)
_STATIC_DETECTORS = {
    "cline_style": detect_cline_style,
    "openhands_stuck": detect_openhands_stuck,
    "gemini_cli": detect_gemini_cli,
    "openclaw_10": _OC10,
    "openclaw_20": _OC20,
    **{f"step_cap_{cap}": make_step_cap_detector(cap) for cap in STEP_CAP_VALUES},
}


@dataclass(slots=True)
class CompactRunRecord:
    """Lightweight per-trajectory summary retaining exact stop steps and step costs."""

    dataset: str
    run_id: str
    harness: str
    model: str
    task_name: str
    succeeded: bool
    n_steps: int
    n_tool_steps: int
    has_any_loop: bool
    has_rotation_loop: bool
    has_aba_thrash: bool
    has_exact_loop: bool
    has_near_duplicate: bool
    has_error_streak: bool
    has_redundant_reread: bool
    has_productive_repeat: bool
    loop_steps: int
    productive_repeat_steps: int
    label_step_counts: dict[str, int]
    cost_included: bool
    drop_reason: str | None
    mismatch_ratio: float | None
    total_cost_usd: float
    loop_cost_usd: float
    step_costs: np.ndarray  # float64 array of length n_steps (empty if not cost_included)
    stop_steps: dict[str, int | None]
    has_rsc_ref: bool


def _is_finite_pos(val: Any) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
        return not math.isnan(f) and not math.isinf(f) and f > 0.0
    except (TypeError, ValueError):
        return False


def _is_finite_nonneg(val: Any) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
        return not math.isnan(f) and not math.isinf(f) and f >= 0.0
    except (TypeError, ValueError):
        return False


def _analyze_single_trajectory(
    *,
    dataset: str,
    run_id: str,
    harness: str,
    model: str,
    task_name: str,
    succeeded: bool,
    steps: list[NormalizedStep],
    initial_prompt_chars: int,
    reported_cost_usd: float | None,
    reported_uncached_in: float | None,
    reported_cached_in: float | None,
    reported_out: float | None,
    reported_total_in: float | None,
    disable_cache_discount: bool,
    compaction_reason: str | None,
    has_rsc: bool,
) -> CompactRunRecord:
    step_labels = label_trajectory(steps)
    cost_res = rebuild_trajectory_costs(
        steps,
        model_name=model,
        initial_prompt_chars=initial_prompt_chars,
        reported_cost_usd=reported_cost_usd,
        reported_uncached_input_tokens=reported_uncached_in,
        reported_cached_input_tokens=reported_cached_in,
        reported_output_tokens=reported_out,
        reported_total_input_tokens=reported_total_in,
        disable_cache_discount=disable_cache_discount,
        compaction_or_subagent_reason=compaction_reason,
    )

    label_counts: dict[str, int] = {k: 0 for k in ALL_STEP_LABELS}
    all_tags: set[str] = set()
    loop_steps = 0
    prod_steps = 0
    loop_cost = 0.0
    for idx, sl in enumerate(step_labels):
        label_counts[sl.primary_label] = label_counts.get(sl.primary_label, 0) + 1
        all_tags.update(sl.labels)
        if sl.is_loop:
            loop_steps += 1
            if cost_res.included:
                loop_cost += cost_res.step_costs_usd[idx]
        elif "productive_repeat" in sl.labels:
            prod_steps += 1

    has_any_loop = loop_steps > 0
    replay_obj = ReplayTrajectory(
        run_id=run_id,
        harness=harness,
        model=model,
        task_name=task_name,
        succeeded=succeeded,
        steps=steps,
        step_costs_usd=cost_res.step_costs_usd,
        cost_included=cost_res.included,
        has_any_loop=has_any_loop,
        has_rotation_loop=("period_k_rotation" in all_tags),
    )
    stop_steps = {dname: fn(replay_obj) for dname, fn in _STATIC_DETECTORS.items()}

    sc_arr = (
        np.array(cost_res.step_costs_usd, dtype=np.float64)
        if cost_res.included
        else np.empty(0, dtype=np.float64)
    )
    return CompactRunRecord(
        dataset=dataset,
        run_id=run_id,
        harness=harness,
        model=model,
        task_name=task_name,
        succeeded=succeeded,
        n_steps=len(steps),
        n_tool_steps=sum(1 for s in steps if s.has_tool),
        has_any_loop=has_any_loop,
        has_rotation_loop=("period_k_rotation" in all_tags),
        has_aba_thrash=("aba_thrash" in all_tags),
        has_exact_loop=("exact_loop" in all_tags),
        has_near_duplicate=("near_duplicate" in all_tags),
        has_error_streak=("error_streak" in all_tags),
        has_redundant_reread=("redundant_reread" in all_tags),
        has_productive_repeat=("productive_repeat" in all_tags),
        loop_steps=loop_steps,
        productive_repeat_steps=prod_steps,
        label_step_counts=label_counts,
        cost_included=cost_res.included,
        drop_reason=cost_res.drop_reason,
        mismatch_ratio=cost_res.mismatch_ratio,
        total_cost_usd=cost_res.total_cost_usd,
        loop_cost_usd=loop_cost,
        step_costs=sc_arr,
        stop_steps=stop_steps,
        has_rsc_ref=has_rsc,
    )


def _process_tb_batch(rows: list[dict[str, Any]]) -> tuple[list[CompactRunRecord], dict[str, int]]:
    records: list[CompactRunRecord] = []
    counts = Counter()

    for idx_in_batch, row in enumerate(rows):
        counts["total_rows"] += 1
        raw_steps_str = row.get("steps")
        raw_steps = orjson.loads(raw_steps_str) if raw_steps_str else None
        if not raw_steps:
            counts["dropped_null_or_empty_steps"] += 1
            continue

        agent = row.get("agent") or "unknown"
        model = row.get("model") or "unknown"
        task_name = row.get("task_name") or "unknown"
        trial_id = row.get("trial_id") or row.get("trial_name") or f"row_{idx_in_batch}"
        succeeded = int(row.get("reward") or 0) == 1

        if agent == "openhands":
            raw_steps = collapse_openhands_tb_steps(raw_steps)

        init_chars = 0
        started_agent = False
        interjected_chars = 0
        extra_texts: list[str] = []
        norm_steps: list[NormalizedStep] = []
        has_rsc = False

        for s in raw_steps:
            src = s.get("src") or ""
            msg = s.get("msg") or ""
            obs = s.get("obs")
            tools = s.get("tools") or []

            if is_rsc_ref(msg) or is_rsc_ref(obs):
                has_rsc = True
            for tl in tools:
                if is_rsc_ref(tl.get("cmd")):
                    has_rsc = True

            if not started_agent and src in ("system", "user") and not tools:
                if not is_rsc_ref(msg):
                    init_chars += len(msg)
                extra_texts.append(msg)
                continue

            if src in ("system", "user") and not tools:
                extra_texts.append(msg)
                if not is_rsc_ref(msg):
                    interjected_chars += len(msg)
                continue

            started_agent = True
            step_idx = len(norm_steps)
            msg_eff = "" if is_rsc_ref(msg) else msg
            obs_eff_chars = 0 if is_rsc_ref(obs) else len(obs or "")

            if not tools:
                ns = make_step(
                    step_idx,
                    tool_name=None,
                    raw_args=None,
                    obs=obs,
                    assistant_text=msg_eff,
                    msg_chars_override=len(msg_eff),
                    obs_chars_override=obs_eff_chars + interjected_chars,
                )
            elif len(tools) == 1:
                tl = tools[0]
                fn = tl.get("fn") or "unknown"
                cmd = tl.get("cmd")
                cmd_chars = 0 if is_rsc_ref(cmd) else len(str(cmd or ""))
                ns = make_step(
                    step_idx,
                    tool_name=fn,
                    raw_args=cmd,
                    obs=obs,
                    assistant_text=msg_eff,
                    is_tb_coarse_schema=True,
                    msg_chars_override=len(msg_eff) + len(fn) + cmd_chars,
                    obs_chars_override=obs_eff_chars + interjected_chars,
                )
            else:
                sub_names = [str(tl.get("fn") or "unknown") for tl in tools]
                if any(sn.strip().lower() in {"task", "taskoutput", "call_llm_batch", "agent"} for sn in sub_names):
                    comp_name = "Task"
                else:
                    comp_name = "+".join(sn.strip().lower() for sn in sub_names)
                any_bad_arg = any(
                    is_rsc_ref(tl.get("cmd"))
                    or (
                        (tl.get("fn") or "").strip().lower()
                        not in {"finish", "submit", "mark_task_complete", "end_execution", "finish_verification"}
                        and not str(tl.get("cmd") or "").strip()
                    )
                    for tl in tools
                )
                batch_args = (
                    None
                    if any_bad_arg
                    else [{"fn": tl.get("fn"), "cmd": tl.get("cmd")} for tl in tools]
                )
                cmd_chars = sum(
                    0 if is_rsc_ref(tl.get("cmd")) else len(str(tl.get("cmd") or "")) + len(str(tl.get("fn") or ""))
                    for tl in tools
                )
                ns = make_step(
                    step_idx,
                    tool_name=comp_name,
                    raw_args=batch_args,
                    obs=obs,
                    assistant_text=msg_eff,
                    is_tb_coarse_schema=True,
                    msg_chars_override=len(msg_eff) + cmd_chars,
                    obs_chars_override=obs_eff_chars + interjected_chars,
                )
            interjected_chars = 0
            norm_steps.append(ns)

        comp_reason = detect_compaction_or_subagents(norm_steps, extra_texts=extra_texts)

        rep_cents = row.get("cost_cents")
        rep_cost_usd = (float(rep_cents) / 100.0) if _is_finite_pos(rep_cents) else None
        cache_val = row.get("cache_tokens")
        disable_cache = _is_finite_nonneg(cache_val) and float(cache_val) == 0.0

        rec = _analyze_single_trajectory(
            dataset="terminalbench",
            run_id=str(trial_id),
            harness=agent,
            model=model,
            task_name=task_name,
            succeeded=succeeded,
            steps=norm_steps,
            initial_prompt_chars=init_chars,
            reported_cost_usd=rep_cost_usd,
            reported_uncached_in=None,
            reported_cached_in=None,
            reported_out=None,
            reported_total_in=None,
            disable_cache_discount=disable_cache,
            compaction_reason=comp_reason,
            has_rsc=has_rsc,
        )
        records.append(rec)
        counts["rows_with_steps"] += 1
        if rec.cost_included:
            counts["cost_included"] += 1
        else:
            counts[f"cost_dropped_{rec.drop_reason}"] += 1

    return records, dict(counts)


def analyze_terminalbench(data_dir: Path) -> tuple[list[CompactRunRecord], dict[str, int]]:
    """Load and analyze all Terminal-Bench 2.0 trajectories using parallel worker batches."""
    parquets = sorted((data_dir / "terminalbench" / "data").glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found under {data_dir / 'terminalbench' / 'data'}")

    batches: list[list[dict[str, Any]]] = []
    for pf in parquets:
        pf_reader = pq.ParquetFile(pf)
        for batch in pf_reader.iter_batches(batch_size=1000):
            batches.append(batch.to_pylist())

    records: list[CompactRunRecord] = []
    total_counts = Counter()
    workers = min(6, max(1, (os.cpu_count() or 4) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for recs, cnts in ex.map(_process_tb_batch, batches):
            records.extend(recs)
            total_counts.update(cnts)

    records.sort(key=lambda r: (r.harness, r.model, r.task_name, r.run_id))
    return records, dict(total_counts)


def _extract_swe_tokens(rec: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None, float | None, int]:
    """Return (cost_usd, uncached_in, cached_in, out_tok, total_in_all, aux_tok) for SWE-rebench."""
    u = rec.get("usage") or {}
    cost_usd = u.get("cost_usd")
    rep_cost = float(cost_usd) if _is_finite_pos(cost_usd) else None
    tok = u.get("tokens") or {}
    inp = tok.get("input")
    cached = tok.get("cached_input")
    cc = tok.get("cache_creation") or 0
    out = tok.get("output")
    aux = int(tok.get("aux_models") or 0)
    primary = tok.get("primary_model") or tok.get("total")

    if inp is None or out is None:
        return rep_cost, None, None, None, None, aux

    inp_f = float(inp)
    cached_f = float(cached or 0.0)
    cc_f = float(cc or 0.0)
    out_f = float(out)

    if primary is not None:
        p_f = float(primary)
        err_excl = abs((inp_f + cached_f + cc_f + out_f) - p_f)
        err_incl = abs((inp_f + out_f) - p_f)
        if err_excl <= err_incl:
            uncached_in = inp_f + cc_f
            total_in_all = inp_f + cc_f + cached_f
        else:
            uncached_in = max(0.0, inp_f - cached_f)
            total_in_all = inp_f
    else:
        uncached_in = max(0.0, inp_f - cached_f)
        total_in_all = inp_f

    return rep_cost, uncached_in, cached_f, out_f, total_in_all, aux


def _process_swe_shard(sh: Path) -> tuple[list[CompactRunRecord], dict[str, int]]:
    records: list[CompactRunRecord] = []
    counts = Counter()

    with gzip.open(sh, "rt", encoding="utf-8") as fh:
        for line in fh:
            counts["total_rows"] += 1
            rec = orjson.loads(line)
            part = rec["participant"]
            harness = part["key"]
            model = part["model"]
            task_name = rec["instance_id"]
            traj_id = rec["trajectory_id"]
            succeeded = bool(rec["evaluation"]["resolved"])

            events = rec.get("events") or []
            init_chars = 0
            has_multi_prompts = False
            extra_texts: list[str] = []
            tool_results_by_id: dict[str, dict[str, Any]] = {}
            for ev in events:
                if ev.get("type") == "tool_result" and ev.get("tool_call_id"):
                    tool_results_by_id[ev["tool_call_id"]] = ev

            norm_steps: list[NormalizedStep] = []
            pending_text_parts: list[str] = []

            for ev in events:
                et = ev.get("type")
                if et == "prompt":
                    msgs = ev.get("messages") or []
                    pchars = sum(len(str(m.get("content") or "")) for m in msgs)
                    if ev.get("seq", 0) == 0:
                        init_chars += pchars
                    else:
                        has_multi_prompts = True
                    for m in msgs:
                        extra_texts.append(str(m.get("content") or ""))
                elif et in ("assistant_message", "assistant_analysis", "final_message"):
                    c = ev.get("content")
                    if isinstance(c, str) and c:
                        pending_text_parts.append(c)
                    elif c is not None:
                        pending_text_parts.append(str(c))
                elif et == "tool_call":
                    tc_id = ev.get("tool_call_id") or ""
                    tr = tool_results_by_id.get(tc_id)
                    obs_raw = tr.get("content") if tr else None
                    obs_str = (
                        obs_raw
                        if isinstance(obs_raw, str)
                        else ("" if obs_raw is None else orjson.dumps(obs_raw).decode("utf-8", errors="replace"))
                    )
                    orig_bytes = tr.get("content_bytes_original") if tr else None
                    obs_chars = max(len(obs_str), int(orig_bytes or 0))
                    assistant_txt = "\n".join(pending_text_parts)
                    pending_text_parts.clear()

                    tname = ev.get("tool") or "unknown"
                    inp = ev.get("input")
                    inp_len = len(orjson.dumps(inp)) if inp is not None else 0
                    ns = make_step(
                        len(norm_steps),
                        tool_name=tname,
                        raw_args=inp,
                        obs=obs_str,
                        assistant_text=assistant_txt,
                        status=tr.get("status") if tr else None,
                        exit_code=tr.get("exit_code") if tr else None,
                        error_field=tr.get("error") if tr else None,
                        is_tb_coarse_schema=False,
                        msg_chars_override=len(assistant_txt) + len(tname) + inp_len,
                        obs_chars_override=obs_chars,
                    )
                    norm_steps.append(ns)
                elif et == "agent_step":
                    assistant_txt = "\n".join(pending_text_parts)
                    pending_text_parts.clear()
                    sname = (ev.get("name") or "").strip()
                    scontent = ev.get("content")
                    obs_str = scontent if isinstance(scontent, str) else str(scontent or "")
                    first_line = obs_str.splitlines()[0].strip() if obs_str else ""
                    if sname in ("Opened file", "Open file"):
                        tname_step = "read"
                        raw_a: Any = {"file_path": first_line} if first_line else None
                    elif sname in ("Edited files", "Edited file"):
                        tname_step = "edit"
                        raw_a = None  # junie summary omits diff/path details; file_op becomes write_any
                    elif sname in ("Fetch URL", "Read skill", "Web search"):
                        tname_step = sname.lower().replace(" ", "_")
                        raw_a = first_line if first_line else None
                    else:
                        tname_step = "agent_step"
                        raw_a = sname if sname else None
                    ns = make_step(
                        len(norm_steps),
                        tool_name=tname_step,
                        raw_args=raw_a,
                        obs=obs_str,
                        assistant_text=assistant_txt,
                        msg_chars_override=len(assistant_txt) + len(sname),
                        obs_chars_override=len(obs_str),
                    )
                    norm_steps.append(ns)

            if pending_text_parts:
                trailing = "\n".join(pending_text_parts)
                norm_steps.append(
                    make_step(
                        len(norm_steps),
                        tool_name=None,
                        raw_args=None,
                        obs=None,
                        assistant_text=trailing,
                        msg_chars_override=len(trailing),
                        obs_chars_override=0,
                    )
                )

            rep_cost, uncached_in, cached_in, out_tok, total_in, aux_tok = _extract_swe_tokens(rec)
            comp_reason = detect_compaction_or_subagents(
                norm_steps,
                extra_texts=extra_texts,
                aux_models_tokens=aux_tok,
                has_multiple_prompts=has_multi_prompts,
            )

            out_rec = _analyze_single_trajectory(
                dataset="swe_rebench",
                run_id=traj_id,
                harness=harness,
                model=model,
                task_name=task_name,
                succeeded=succeeded,
                steps=norm_steps,
                initial_prompt_chars=init_chars,
                reported_cost_usd=rep_cost,
                reported_uncached_in=uncached_in,
                reported_cached_in=cached_in,
                reported_out=out_tok,
                reported_total_in=total_in,
                disable_cache_discount=(cached_in == 0.0),
                compaction_reason=comp_reason,
                has_rsc=False,
            )
            records.append(out_rec)
            if out_rec.cost_included:
                counts["cost_included"] += 1
            else:
                counts[f"cost_dropped_{out_rec.drop_reason}"] += 1

    return records, dict(counts)


def analyze_swe_rebench(data_dir: Path) -> tuple[list[CompactRunRecord], dict[str, int]]:
    """Load and analyze all SWE-rebench July 2026 trajectories across parallel shard workers."""
    shards = sorted((data_dir / "swe_rebench_07_2026" / "trajectories").glob("*/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {data_dir / 'swe_rebench_07_2026'}")

    records: list[CompactRunRecord] = []
    total_counts = Counter()
    workers = min(6, max(1, (os.cpu_count() or 4) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for recs, cnts in ex.map(_process_swe_shard, shards):
            records.extend(recs)
            total_counts.update(cnts)

    records.sort(key=lambda r: (r.harness, r.model, r.task_name, r.run_id))
    return records, dict(total_counts)


def analyze_tau2(data_dir: Path) -> dict[str, Any]:
    """Sanity check on tau2-bench bundled result JSONs to test whether tau2 agents loop."""
    tau2_files = sorted(
        p for p in (data_dir / "tau2_results").glob("*.json") if p.name != "_files.json"
    )
    if not tau2_files:
        return {"files": 0, "simulations": 0}

    total_sims = 0
    total_tool_steps = 0
    any_loop_sims = 0
    exact_loop_sims = 0
    near_dup_sims = 0
    aba_sims = 0
    rot_sims = 0
    err_streak_sims = 0
    reread_sims = 0
    prod_repeat_sims = 0
    loop_steps = 0
    prod_steps = 0
    succ_total = 0
    succ_loop = 0
    fail_total = 0
    fail_loop = 0
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)

    for fp in tau2_files:
        obj = orjson.loads(fp.read_bytes())
        domain = (obj.get("info") or {}).get("environment_info", {}).get("domain_name") or fp.stem.split("_")[1]
        for sim in obj.get("simulations", []):
            total_sims += 1
            reward = float((sim.get("reward_info") or {}).get("reward") or 0.0)
            succeeded = reward >= 0.999
            if succeeded:
                succ_total += 1
            else:
                fail_total += 1

            msgs = sim.get("messages") or []
            tool_res_by_id: dict[str, dict[str, Any]] = {}
            for m in msgs:
                if m.get("role") == "tool" and m.get("id"):
                    tool_res_by_id[m["id"]] = m

            steps: list[NormalizedStep] = []
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                for tc in tcs:
                    tc_id = tc.get("id") or ""
                    tr = tool_res_by_id.get(tc_id, {})
                    obs = tr.get("content")
                    err_flag = bool(tr.get("error"))
                    ns = make_step(
                        len(steps),
                        tool_name=tc.get("name") or "unknown",
                        raw_args=tc.get("arguments"),
                        obs=obs if isinstance(obs, str) else str(obs or ""),
                        assistant_text=m.get("content") or "",
                        status="error" if err_flag else "success",
                        error_field="error" if err_flag else None,
                    )
                    steps.append(ns)
            total_tool_steps += len(steps)
            lbls = label_trajectory(steps)
            tags = {tag for l in lbls for tag in l.labels}
            n_lp = sum(1 for l in lbls if l.is_loop)
            n_pr = sum(1 for l in lbls if not l.is_loop and "productive_repeat" in l.labels)
            loop_steps += n_lp
            prod_steps += n_pr
            by_domain[domain]["sims"] += 1
            if n_lp > 0:
                any_loop_sims += 1
                by_domain[domain]["loop_sims"] += 1
                if succeeded:
                    succ_loop += 1
                else:
                    fail_loop += 1
            if "exact_loop" in tags:
                exact_loop_sims += 1
            if "near_duplicate" in tags:
                near_dup_sims += 1
            if "aba_thrash" in tags:
                aba_sims += 1
            if "period_k_rotation" in tags:
                rot_sims += 1
            if "error_streak" in tags:
                err_streak_sims += 1
            if "redundant_reread" in tags:
                reread_sims += 1
            if "productive_repeat" in tags:
                prod_repeat_sims += 1

    return {
        "files": len(tau2_files),
        "simulations": total_sims,
        "total_tool_steps": total_tool_steps,
        "any_loop_sims": any_loop_sims,
        "any_loop_rate": any_loop_sims / total_sims if total_sims else 0.0,
        "loop_steps": loop_steps,
        "loop_step_share": loop_steps / total_tool_steps if total_tool_steps else 0.0,
        "productive_repeat_sims": prod_repeat_sims,
        "productive_repeat_steps": prod_steps,
        "exact_loop_sims": exact_loop_sims,
        "near_duplicate_sims": near_dup_sims,
        "aba_thrash_sims": aba_sims,
        "period_k_rotation_sims": rot_sims,
        "error_streak_sims": err_streak_sims,
        "redundant_reread_sims": reread_sims,
        "pass_rate_looping": succ_loop / any_loop_sims if any_loop_sims else 0.0,
        "pass_rate_loop_free": (succ_total - succ_loop) / (total_sims - any_loop_sims)
        if (total_sims - any_loop_sims)
        else 0.0,
        "by_domain": {k: dict(v) for k, v in sorted(by_domain.items())},
    }


def evaluate_all_detectors(
    records: list[CompactRunRecord],
    *,
    n_seeds: int = 200,
    seed: int = 42,
) -> tuple[list[DetectorResult], dict[str, int]]:
    """Evaluate all detectors + 1.25x P90 harness turn budget with 200-seed matched-random baselines."""
    if not records:
        return [], {}

    # Compute 1.25x P90 turn budget per harness
    by_h_succ: dict[str, list[int]] = defaultdict(list)
    all_succ: list[int] = []
    for r in records:
        if r.succeeded and r.n_steps > 0:
            by_h_succ[r.harness].append(r.n_steps)
            all_succ.append(r.n_steps)
    global_p90 = float(np.percentile(all_succ, 90.0)) if all_succ else 50.0
    budgets: dict[str, int] = {}
    for h in {r.harness for r in records}:
        vals = by_h_succ.get(h)
        p90 = float(np.percentile(vals, 90.0)) if vals else global_p90
        budgets[h] = max(1, int(math.ceil(1.25 * p90)))

    # Add turn_budget stop steps to each record's stop_steps
    for r in records:
        cap = budgets[r.harness]
        r.stop_steps["turn_budget_1_25x_p90"] = (cap - 1) if r.n_steps > cap else None

    detector_names = [
        "cline_style",
        "openhands_stuck",
        "gemini_cli",
        "openclaw_10",
        "openclaw_20",
        *[f"step_cap_{cap}" for cap in STEP_CAP_VALUES],
        "turn_budget_1_25x_p90",
    ]

    total_runs = len(records)
    cost_eval_runs = sum(1 for r in records if r.cost_included)
    total_eval_cost = sum(r.total_cost_usd for r in records if r.cost_included)

    n_steps_arr = np.array([r.n_steps for r in records], dtype=np.int32)
    succ_arr = np.array([r.succeeded for r in records], dtype=bool)
    cost_inc_arr = np.array([r.cost_included for r in records], dtype=bool)

    suffix_costs: list[np.ndarray | None] = [None] * total_runs
    for idx, r in enumerate(records):
        if r.cost_included and r.n_steps > 0:
            c = r.step_costs
            rev_cumsum = np.cumsum(c[::-1])[::-1]
            suf = np.zeros(len(c), dtype=np.float64)
            if len(c) > 1:
                suf[:-1] = rev_cumsum[1:]
            suffix_costs[idx] = suf

    harness_list = [r.harness for r in records]
    results: list[DetectorResult] = []

    for idx_d, dname in enumerate(detector_names):
        stops = [r.stop_steps[dname] for r in records]
        fired_indices = [i for i, st in enumerate(stops) if st is not None]
        fired_runs = len(fired_indices)
        false_stops = sum(1 for i in fired_indices if records[i].succeeded)
        firing_rate = fired_runs / total_runs if total_runs else 0.0
        false_stop_rate = false_stops / fired_runs if fired_runs else 0.0
        success_loss_pp = 100.0 * (false_stops / total_runs) if total_runs else 0.0

        saved_usd = 0.0
        for i in fired_indices:
            if records[i].cost_included:
                st = stops[i]
                assert st is not None
                suf = suffix_costs[i]
                if suf is not None:
                    saved_usd += float(suf[st])
        saved_pct = (saved_usd / total_eval_cost) if total_eval_cost > 0 else 0.0

        r_mean, r_lo, r_hi, s_mean, s_lo, s_hi = run_matched_random_baseline(
            harness_list=harness_list,
            n_steps_arr=n_steps_arr,
            succ_arr=succ_arr,
            cost_inc_arr=cost_inc_arr,
            suffix_costs=suffix_costs,
            stop_steps=stops,
            total_eval_cost=total_eval_cost,
            n_seeds=n_seeds,
            seed=seed + idx_d * 1000,
        )

        results.append(
            DetectorResult(
                name=dname,
                total_runs=total_runs,
                fired_runs=fired_runs,
                firing_rate=firing_rate,
                false_stop_count=false_stops,
                false_stop_rate=false_stop_rate,
                success_loss_pp=success_loss_pp,
                cost_eval_runs=cost_eval_runs,
                total_eval_cost_usd=total_eval_cost,
                dollars_saved_usd=saved_usd,
                dollars_saved_pct=saved_pct,
                rand_false_stop_mean=r_mean,
                rand_false_stop_ci_low=r_lo,
                rand_false_stop_ci_high=r_hi,
                rand_dollars_saved_pct_mean=s_mean,
                rand_dollars_saved_pct_ci_low=s_lo,
                rand_dollars_saved_pct_ci_high=s_hi,
            )
        )

    return results, budgets


def aggregate_group_metrics(records: list[CompactRunRecord], group_attr: str) -> pd.DataFrame:
    """Aggregate loop prevalence, step share, dollar share, and pass rates by model or harness."""
    buckets: dict[str, list[CompactRunRecord]] = defaultdict(list)
    for r in records:
        buckets[getattr(r, group_attr)].append(r)

    rows: list[dict[str, Any]] = []
    for key in sorted(buckets):
        grp = buckets[key]
        n_runs = len(grp)
        loop_runs = sum(1 for r in grp if r.has_any_loop)
        loop_free_runs = n_runs - loop_runs
        succ_loop = sum(1 for r in grp if r.has_any_loop and r.succeeded)
        succ_free = sum(1 for r in grp if (not r.has_any_loop) and r.succeeded)
        total_steps = sum(r.n_steps for r in grp)
        loop_steps = sum(r.loop_steps for r in grp)
        prod_steps = sum(r.productive_repeat_steps for r in grp)

        cost_grp = [r for r in grp if r.cost_included]
        n_cost_runs = len(cost_grp)
        total_usd = sum(r.total_cost_usd for r in cost_grp)
        loop_usd = sum(r.loop_cost_usd for r in cost_grp)

        rows.append(
            {
                group_attr: key,
                "runs": n_runs,
                "loop_runs": loop_runs,
                "loop_run_rate": loop_runs / n_runs if n_runs else 0.0,
                "total_steps": total_steps,
                "loop_steps": loop_steps,
                "loop_step_share": loop_steps / total_steps if total_steps else 0.0,
                "productive_repeat_step_share": prod_steps / total_steps if total_steps else 0.0,
                "cost_runs": n_cost_runs,
                "total_cost_usd": total_usd,
                "loop_cost_usd": loop_usd,
                "loop_dollar_share": loop_usd / total_usd if total_usd > 0 else float("nan"),
                "overall_pass_rate": (succ_loop + succ_free) / n_runs if n_runs else 0.0,
                "looping_pass_rate": succ_loop / loop_runs if loop_runs else float("nan"),
                "loop_free_pass_rate": succ_free / loop_free_runs if loop_free_runs else float("nan"),
                "exact_loop_runs": sum(1 for r in grp if r.has_exact_loop),
                "near_duplicate_runs": sum(1 for r in grp if r.has_near_duplicate),
                "aba_thrash_runs": sum(1 for r in grp if r.has_aba_thrash),
                "rotation_runs": sum(1 for r in grp if r.has_rotation_loop),
                "error_streak_runs": sum(1 for r in grp if r.has_error_streak),
                "redundant_reread_runs": sum(1 for r in grp if r.has_redundant_reread),
                "productive_repeat_runs": sum(1 for r in grp if r.has_productive_repeat),
            }
        )
    return pd.DataFrame(rows)


def select_gemini_loop_prone_tasks(records: list[CompactRunRecord], top_n: int = 20) -> pd.DataFrame:
    """Pick the top Terminal-Bench tasks where Gemini-family models loop most often."""
    gem_recs = [
        r for r in records if r.dataset == "terminalbench" and "gemini" in r.model.lower()
    ]
    by_task: dict[str, list[CompactRunRecord]] = defaultdict(list)
    for r in gem_recs:
        by_task[r.task_name].append(r)

    rows: list[dict[str, Any]] = []
    for task, grp in by_task.items():
        n_runs = len(grp)
        n_loop = sum(1 for r in grp if r.has_any_loop)
        n_rot = sum(1 for r in grp if r.has_rotation_loop)
        n_pass = sum(1 for r in grp if r.succeeded)
        tot_steps = sum(r.n_steps for r in grp)
        lp_steps = sum(r.loop_steps for r in grp)
        cost_grp = [r for r in grp if r.cost_included]
        tot_usd = sum(r.total_cost_usd for r in cost_grp)
        lp_usd = sum(r.loop_cost_usd for r in cost_grp)
        rows.append(
            {
                "task_name": task,
                "gemini_runs": n_runs,
                "gemini_loop_runs": n_loop,
                "gemini_loop_run_rate": n_loop / n_runs if n_runs else 0.0,
                "gemini_loop_step_share": lp_steps / tot_steps if tot_steps else 0.0,
                "gemini_loop_dollar_share": lp_usd / tot_usd if tot_usd > 0 else float("nan"),
                "gemini_rotation_runs": n_rot,
                "gemini_pass_rate": n_pass / n_runs if n_runs else 0.0,
                "gemini_mean_steps": tot_steps / n_runs if n_runs else 0.0,
                "gemini_models": ",".join(sorted({r.model for r in grp})),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["gemini_loop_run_rate", "gemini_loop_step_share", "gemini_loop_runs", "task_name"],
        ascending=[False, False, False, True],
    ).head(top_n)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
