"""Loop detector replay and matched-random stopping baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from agent_loops.canonicalize import NormalizedStep


@dataclass(slots=True)
class ReplayTrajectory:
    """Compact trajectory payload needed for detector replay and matched-random sampling."""

    run_id: str
    harness: str
    model: str
    task_name: str
    succeeded: bool
    steps: list[NormalizedStep]
    step_costs_usd: list[float]
    cost_included: bool
    has_any_loop: bool
    has_rotation_loop: bool

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def total_cost_usd(self) -> float:
        return float(sum(self.step_costs_usd))


def detect_cline_style(traj: ReplayTrajectory) -> int | None:
    """Cline-style: warn at 3, stop at 5 adjacent identical tool calls (args_known=True)."""
    streak = 0
    prev_hash: str | None = None
    for idx, s in enumerate(traj.steps):
        if not s.has_tool:
            continue
        if not s.args_known:
            streak = 0
            prev_hash = None
            continue
        if s.call_hash == prev_hash:
            streak += 1
        else:
            prev_hash = s.call_hash
            streak = 1
        if streak >= 5:
            return idx
    return None


def detect_openhands_stuck(traj: ReplayTrajectory) -> int | None:
    """OpenHands stuck detector:
    - 4 identical (action, observation) pairs consecutively
    - 3 identical (action, error) pairs consecutively
    - 6 alternating A-B steps (A, B, A, B, A, B) on (action, obs)
    """
    tool_indices: list[int] = []
    for idx, s in enumerate(traj.steps):
        if not s.has_tool:
            continue
        tool_indices.append(idx)
        k = len(tool_indices)

        # 1. 3 identical (action, error)
        if k >= 3:
            last3 = [traj.steps[i] for i in tool_indices[-3:]]
            if all(x.args_known and x.is_error and x.error_sig is not None for x in last3):
                if len({(x.call_hash, x.error_sig) for x in last3}) == 1:
                    return idx

        # 2. 4 identical (action, observation)
        if k >= 4:
            last4 = [traj.steps[i] for i in tool_indices[-4:]]
            if all(x.args_known and x.obs_known for x in last4):
                if len({(x.call_hash, x.obs_norm_hash) for x in last4}) == 1:
                    return idx

        # 3. 6 alternating A-B steps (3 A-B cycles: A, B, A, B, A, B)
        if k >= 6:
            last6 = [traj.steps[i] for i in tool_indices[-6:]]
            if all(x.args_known and x.obs_known for x in last6):
                a_pair = (last6[0].call_hash, last6[0].obs_norm_hash)
                b_pair = (last6[1].call_hash, last6[1].obs_norm_hash)
                if (
                    last6[0].call_hash != last6[1].call_hash
                    and (last6[2].call_hash, last6[2].obs_norm_hash) == a_pair
                    and (last6[4].call_hash, last6[4].obs_norm_hash) == a_pair
                    and (last6[3].call_hash, last6[3].obs_norm_hash) == b_pair
                    and (last6[5].call_hash, last6[5].obs_norm_hash) == b_pair
                ):
                    return idx
    return None


def detect_gemini_cli(traj: ReplayTrajectory) -> int | None:
    """Gemini CLI loop detector:
    - 5 consecutive repeated tool calls across steps
    - 10 repeated non-overlapping 50-char content chunks within a single assistant turn
    """
    streak = 0
    prev_hash: str | None = None
    chunk_size = 50

    for idx, s in enumerate(traj.steps):
        # 1. Intra-turn content chunk repetition check (resets each turn, matching Gemini CLI)
        if s.assistant_text and len(s.assistant_text) >= chunk_size * 10:
            clean = " ".join(s.assistant_text.split())
            if len(clean) >= chunk_size * 10:
                # Fast C-level necessary check before tracking non-overlapping spacing
                raw_counts = Counter(
                    clean[i : i + chunk_size] for i in range(len(clean) - chunk_size + 1)
                )
                if max(raw_counts.values(), default=0) >= 10:
                    chunk_counts: dict[str, int] = {}
                    last_pos: dict[str, int] = {}
                    for pos in range(len(clean) - chunk_size + 1):
                        chunk = clean[pos : pos + chunk_size]
                        prev_p = last_pos.get(chunk)
                        if prev_p is None or (pos - prev_p) >= chunk_size:
                            if len(set(chunk.replace(" ", ""))) > 3:
                                cnt = chunk_counts.get(chunk, 0) + 1
                                chunk_counts[chunk] = cnt
                                last_pos[chunk] = pos
                                if cnt >= 10:
                                    return idx

        # 2. 5 repeated tool calls
        if s.has_tool:
            if not s.args_known:
                streak = 0
                prev_hash = None
            elif s.call_hash == prev_hash:
                streak += 1
                if streak >= 5:
                    return idx
            else:
                prev_hash = s.call_hash
                streak = 1

    return None


def make_openclaw_detector(threshold: int, window: int = 30) -> Callable[[ReplayTrajectory], int | None]:
    """OpenClaw-style sliding-window repeated (call, observation) detector."""

    def _detect(traj: ReplayTrajectory) -> int | None:
        buf: list[tuple[str, str]] = []
        counts: Counter[tuple[str, str]] = Counter()
        for idx, s in enumerate(traj.steps):
            if not s.has_tool or not s.args_known or not s.obs_known:
                continue
            key = (s.call_hash, s.obs_norm_hash)
            buf.append(key)
            counts[key] += 1
            if len(buf) > window:
                old = buf.pop(0)
                counts[old] -= 1
                if counts[old] <= 0:
                    del counts[old]
            if counts[key] >= threshold:
                return idx
        return None

    return _detect


def make_step_cap_detector(cap: int) -> Callable[[ReplayTrajectory], int | None]:
    """Plain step-cap detector stopping after `cap` steps whenever trajectory exceeds `cap`."""

    def _detect(traj: ReplayTrajectory) -> int | None:
        if traj.n_steps > cap:
            return cap - 1
        return None

    return _detect


def compute_harness_turn_budgets(
    trajectories: list[ReplayTrajectory],
    *,
    percentile: float = 90.0,
    multiplier: float = 1.25,
) -> dict[str, int]:
    """Compute per-harness turn budget = ceil(multiplier * P_percentile(success_turns))."""
    by_harness: dict[str, list[int]] = defaultdict(list)
    all_succ: list[int] = []
    for t in trajectories:
        if t.succeeded and t.n_steps > 0:
            by_harness[t.harness].append(t.n_steps)
            all_succ.append(t.n_steps)
    global_p = float(np.percentile(all_succ, percentile)) if all_succ else 50.0
    budgets: dict[str, int] = {}
    all_harnesses = {t.harness for t in trajectories}
    for h in all_harnesses:
        vals = by_harness.get(h)
        p_val = float(np.percentile(vals, percentile)) if vals else global_p
        budgets[h] = max(1, int(math.ceil(multiplier * p_val)))
    return budgets


def make_turn_budget_detector(budgets: dict[str, int]) -> Callable[[ReplayTrajectory], int | None]:
    """Turn budget detector using 1.25x P90 of successful runs of the same harness."""

    def _detect(traj: ReplayTrajectory) -> int | None:
        cap = budgets.get(traj.harness, 50)
        if traj.n_steps > cap:
            return cap - 1
        return None

    return _detect


@dataclass(slots=True)
class DetectorResult:
    """Aggregated metrics for one detector (and its matched-random baseline)."""

    name: str
    total_runs: int
    fired_runs: int
    firing_rate: float
    false_stop_count: int
    false_stop_rate: float
    success_loss_pp: float
    cost_eval_runs: int
    total_eval_cost_usd: float
    dollars_saved_usd: float
    dollars_saved_pct: float
    # Matched-random (200 seeds) summary
    rand_false_stop_mean: float
    rand_false_stop_ci_low: float
    rand_false_stop_ci_high: float
    rand_dollars_saved_pct_mean: float
    rand_dollars_saved_pct_ci_low: float
    rand_dollars_saved_pct_ci_high: float


def run_matched_random_baseline(
    *,
    harness_list: list[str],
    n_steps_arr: np.ndarray,
    succ_arr: np.ndarray,
    cost_inc_arr: np.ndarray,
    suffix_costs: list[np.ndarray | None],
    stop_steps: list[int | None],
    total_eval_cost: float,
    n_seeds: int = 200,
    seed: int = 42,
) -> tuple[float, float, float, float, float, float]:
    """Run n_seeds matched-random stoppers preserving per-harness firing rate and step distribution."""
    harness_run_indices: dict[str, list[int]] = defaultdict(list)
    for idx, h in enumerate(harness_list):
        harness_run_indices[h].append(idx)

    prepared_harnesses: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for h, r_idxs in harness_run_indices.items():
        h_stops = [stop_steps[i] for i in r_idxs if stop_steps[i] is not None]
        if not h_stops:
            continue
        r_arr = np.array(r_idxs, dtype=np.int32)
        order = np.argsort(-n_steps_arr[r_arr], kind="mergesort")
        r_sorted = r_arr[order]
        l_sorted = n_steps_arr[r_sorted]
        stops_desc = np.sort(np.array(h_stops, dtype=np.int32))[::-1]
        p_counts = np.searchsorted(-l_sorted, -stops_desc, side="left")
        prepared_harnesses.append((r_sorted, stops_desc, p_counts))

    rng = np.random.default_rng(seed)
    rand_fsr = np.zeros(n_seeds, dtype=np.float64)
    rand_saved_pct = np.zeros(n_seeds, dtype=np.float64)

    for s_idx in range(n_seeds):
        r_false = 0
        r_fired = 0
        r_saved = 0.0
        for r_sorted, stops_desc, p_counts in prepared_harnesses:
            active: list[int] = []
            prev_p = 0
            for j in range(len(stops_desc)):
                new_p = int(p_counts[j])
                if new_p > prev_p:
                    active.extend(range(prev_p, new_p))
                    prev_p = new_p
                pick_pos = int(rng.integers(len(active)))
                chosen_sorted_idx = active[pick_pos]
                active[pick_pos] = active[-1]
                active.pop()

                run_i = int(r_sorted[chosen_sorted_idx])
                st = int(stops_desc[j])
                r_fired += 1
                if succ_arr[run_i]:
                    r_false += 1
                if cost_inc_arr[run_i]:
                    suf = suffix_costs[run_i]
                    if suf is not None:
                        r_saved += float(suf[st])
        rand_fsr[s_idx] = (r_false / r_fired) if r_fired else 0.0
        rand_saved_pct[s_idx] = (r_saved / total_eval_cost) if total_eval_cost > 0 else 0.0

    return (
        float(np.mean(rand_fsr)),
        float(np.percentile(rand_fsr, 2.5)),
        float(np.percentile(rand_fsr, 97.5)),
        float(np.mean(rand_saved_pct)),
        float(np.percentile(rand_saved_pct, 2.5)),
        float(np.percentile(rand_saved_pct, 97.5)),
    )


def evaluate_detector_with_matched_random(
    name: str,
    detector_fn: Callable[[ReplayTrajectory], int | None],
    trajectories: list[ReplayTrajectory],
    *,
    n_seeds: int = 200,
    seed: int = 42,
) -> DetectorResult:
    """Replay detector over trajectories and compare against 200 matched-random seeds."""
    total_runs = len(trajectories)
    stop_steps: list[int | None] = [detector_fn(t) for t in trajectories]

    fired_indices = [i for i, st in enumerate(stop_steps) if st is not None]
    fired_runs = len(fired_indices)
    false_stops = sum(1 for i in fired_indices if trajectories[i].succeeded)
    firing_rate = fired_runs / total_runs if total_runs else 0.0
    false_stop_rate = false_stops / fired_runs if fired_runs else 0.0
    success_loss_pp = 100.0 * (false_stops / total_runs) if total_runs else 0.0

    cost_eval_runs = sum(1 for t in trajectories if t.cost_included)
    total_eval_cost = sum(t.total_cost_usd for t in trajectories if t.cost_included)
    saved_usd = 0.0
    for i in fired_indices:
        t = trajectories[i]
        if t.cost_included:
            st = stop_steps[i]
            assert st is not None
            saved_usd += sum(t.step_costs_usd[st + 1 :])
    saved_pct = (saved_usd / total_eval_cost) if total_eval_cost > 0 else 0.0

    n_steps_arr = np.array([t.n_steps for t in trajectories], dtype=np.int32)
    succ_arr = np.array([t.succeeded for t in trajectories], dtype=bool)
    cost_inc_arr = np.array([t.cost_included for t in trajectories], dtype=bool)
    suffix_costs: list[np.ndarray | None] = [None] * total_runs
    for idx, t in enumerate(trajectories):
        if t.cost_included and t.n_steps > 0:
            c = np.array(t.step_costs_usd, dtype=np.float64)
            rev_cumsum = np.cumsum(c[::-1])[::-1]
            suf = np.zeros(len(c), dtype=np.float64)
            if len(c) > 1:
                suf[:-1] = rev_cumsum[1:]
            suffix_costs[idx] = suf

    r_mean, r_lo, r_hi, s_mean, s_lo, s_hi = run_matched_random_baseline(
        harness_list=[t.harness for t in trajectories],
        n_steps_arr=n_steps_arr,
        succ_arr=succ_arr,
        cost_inc_arr=cost_inc_arr,
        suffix_costs=suffix_costs,
        stop_steps=stop_steps,
        total_eval_cost=total_eval_cost,
        n_seeds=n_seeds,
        seed=seed,
    )

    return DetectorResult(
        name=name,
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
