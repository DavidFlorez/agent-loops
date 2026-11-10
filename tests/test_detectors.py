"""Unit tests for loop detectors and matched-random baseline."""

from agent_loops.canonicalize import make_step
from agent_loops.detectors import (
    ReplayTrajectory,
    compute_harness_turn_budgets,
    detect_cline_style,
    detect_gemini_cli,
    detect_openhands_stuck,
    evaluate_detector_with_matched_random,
    make_openclaw_detector,
    make_step_cap_detector,
    make_turn_budget_detector,
)


def _traj(run_id: str, cmds_obs: list[tuple[str, str]], *, succeeded: bool = False, harness: str = "h1", texts: list[str] | None = None):
    steps = []
    for i, (cmd, obs) in enumerate(cmds_obs):
        txt = texts[i] if texts else ""
        steps.append(make_step(i, tool_name="Bash", raw_args={"command": cmd}, obs=obs, assistant_text=txt))
    costs = [0.01 * (i + 1) for i in range(len(steps))]
    return ReplayTrajectory(
        run_id=run_id,
        harness=harness,
        model="m",
        task_name="t",
        succeeded=succeeded,
        steps=steps,
        step_costs_usd=costs,
        cost_included=True,
        has_any_loop=True,
        has_rotation_loop=False,
    )


def test_cline_and_gemini_tool_detectors():
    tr = _traj("r1", [("ls", "a")] * 6)
    assert detect_cline_style(tr) == 4  # 5th adjacent identical call (0..4)
    assert detect_gemini_cli(tr) == 4


def test_gemini_cli_content_chunk_detector():
    chunk = "Repeating the exact same fifty character sentence!! " * 10
    tr = _traj("r_txt", [("ls", "1"), ("pwd", "2")], texts=[chunk, ""])
    assert detect_gemini_cli(tr) == 0

    # Normal 12-turn trajectory emitting a 55-char tool XML tag once per turn must NOT trigger
    xml_tag = "<tool_call> <function=bash_terminal> <parameter=command>"
    normal_turns = _traj(
        "r_xml",
        [(f"cmd_{i}", f"out_{i}") for i in range(12)],
        texts=[xml_tag for _ in range(12)],
    )
    assert detect_gemini_cli(normal_turns) is None


def test_openhands_stuck_all_three_rules():
    # 4 identical action+obs
    t1 = _traj("o1", [("ls", "same")] * 5)
    assert detect_openhands_stuck(t1) == 3

    # 3 identical action+error
    t2 = _traj("o2", [("bad", "Exit code 1\nSyntaxError: bad")] * 4)
    assert detect_openhands_stuck(t2) == 2

    # 6 alternating A-B steps (3 A-B cycles)
    t3 = _traj("o3", [("a", "oa"), ("b", "ob")] * 3)
    assert detect_openhands_stuck(t3) == 5


def test_openclaw_catches_rotation_missed_by_cline():
    # Period-3 rotation for 30 steps: Cline never sees 5 adjacent identical, OpenClaw-10 catches at 10th repeat of A
    rot = [("a", "oa"), ("b", "ob"), ("c", "oc")] * 11
    tr = _traj("rot", rot)
    assert detect_cline_style(tr) is None
    oc10 = make_openclaw_detector(10, window=30)
    assert oc10(tr) == 27  # 10th occurrence of 'a' at index 3*9 = 27


def test_turn_budget_and_matched_random_exact_firing_match():
    trajs = [
        _traj(f"s{i}", [(f"c{j}", f"o{j}") for j in range(4)], succeeded=True) for i in range(9)
    ] + [
        _traj("s9", [(f"c{j}", f"o{j}") for j in range(8)], succeeded=True),
        _traj("f0", [("loop", "same")] * 12, succeeded=False),
        _traj("f1", [("loop", "same")] * 15, succeeded=False),
    ]
    budgets = compute_harness_turn_budgets(trajs)
    # P90 of [4]*9 + [8] is ~4.4 -> ceil(1.25 * 4.4) = 6
    assert budgets["h1"] >= 5
    res = evaluate_detector_with_matched_random(
        "cline_style",
        detect_cline_style,
        trajs,
        n_seeds=20,
        seed=7,
    )
    assert res.fired_runs == 2
    assert res.false_stop_rate == 0.0
    assert res.dollars_saved_usd > 0
    assert 0.0 <= res.rand_false_stop_mean <= 1.0


def test_step_cap_detector():
    tr = _traj("cap", [(f"c{i}", f"o{i}") for i in range(35)])
    cap30 = make_step_cap_detector(30)
    assert cap30(tr) == 29
    cap50 = make_step_cap_detector(50)
    assert cap50(tr) is None


def test_matched_random_preserves_per_harness_counts():
    # Two harnesses with different lengths and success rates; verify 200 seeds complete cleanly
    trajs = (
        [_traj(f"a_s{i}", [(f"x{j}", "1") for j in range(6)], succeeded=True, harness="ha") for i in range(8)]
        + [_traj(f"a_f{i}", [("loop", "err")] * 10, succeeded=False, harness="ha") for i in range(4)]
        + [_traj(f"b_s{i}", [(f"x{j}", "1") for j in range(3)], succeeded=True, harness="hb") for i in range(5)]
        + [_traj(f"b_f{i}", [("loop", "err")] * 8, succeeded=False, harness="hb") for i in range(5)]
    )
    res = evaluate_detector_with_matched_random("cline", detect_cline_style, trajs, n_seeds=200, seed=42)
    assert res.fired_runs == 9  # 4 in ha + 5 in hb
    assert res.false_stop_rate == 0.0
    assert res.rand_false_stop_mean > 0.1
    assert 0.0 <= res.rand_false_stop_ci_low <= res.rand_false_stop_mean <= res.rand_false_stop_ci_high <= 1.0

