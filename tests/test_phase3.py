"""Unit tests for Phase 3 strict labeler, read-only command classifier, verdict rule, and Parquet caching."""

from __future__ import annotations

from pathlib import Path

from agent_loops.canonicalize import make_step
from agent_loops.labeler import (
    classify_readonly_command,
    is_readonly_command_step,
    label_trajectory,
    label_trajectory_strict,
)
from agent_loops.phase3 import (
    _records_from_parquet,
    _records_to_parquet,
    apply_verdict_rule,
)
from agent_loops.study import _analyze_single_trajectory


def _bash(idx: int, cmd: str, obs: str):
    return make_step(
        idx,
        tool_name="Bash",
        raw_args={"command": cmd},
        obs=obs,
    )


def test_strict_labeler_drops_single_repeat_and_keeps_triple_repeat() -> None:
    # 2 identical calls in a row (1 repeat): flagged by default, dropped by strict
    steps_two = [
        _bash(0, "ls -la", "file1.py"),
        _bash(1, "ls -la", "file1.py"),
    ]
    def_two = label_trajectory(steps_two)
    str_two = label_trajectory_strict(steps_two)
    assert [x.primary_label for x in def_two] == ["normal", "exact_loop"]
    assert [x.is_loop for x in str_two] == [False, False]

    # 3 identical calls in a row (2 repeats): flagged by both default and strict
    steps_three = [
        _bash(0, "ls -la", "file1.py"),
        _bash(1, "ls -la", "file1.py"),
        _bash(2, "ls -la", "file1.py"),
    ]
    def_three = label_trajectory(steps_three)
    str_three = label_trajectory_strict(steps_three)
    assert [x.primary_label for x in def_three] == ["normal", "exact_loop", "exact_loop"]
    assert [x.primary_label for x in str_three] == ["normal", "exact_loop", "exact_loop"]
    assert [x.is_loop for x in str_three] == [False, True, True]


def test_strict_labeler_aba_thrash_requires_three_cycles() -> None:
    # 2 cycles (A, B, A, B -> 4 steps total): dropped by strict
    steps_2 = [
        _bash(0, "cmd_a", "out_a"),
        _bash(1, "cmd_b", "out_b"),
        _bash(2, "cmd_a", "out_a"),
        _bash(3, "cmd_b", "out_b"),
    ]
    str_2 = label_trajectory_strict(steps_2)
    assert not any(x.is_loop for x in str_2)

    # 3 cycles (A, B, A, B, A, B -> 6 steps total): kept by strict
    steps_3 = steps_2 + [
        _bash(4, "cmd_a", "out_a"),
        _bash(5, "cmd_b", "out_b"),
    ]
    str_3 = label_trajectory_strict(steps_3)
    assert [x.is_loop for x in str_3] == [False, True, True, True, True, True]
    assert [x.primary_label for x in str_3[1:]] == ["aba_thrash"] * 5


def test_readonly_command_classification() -> None:
    assert classify_readonly_command(make_step(0, tool_name="Read", raw_args={"path": "/repo/a.py"}, obs="x")) == "read_or_search_tool"
    assert classify_readonly_command(_bash(0, "git status && git diff HEAD", "clean")) == "git_readonly"
    assert classify_readonly_command(_bash(0, "pytest -q tests/test_foo.py", "1 passed")) == "pytest_or_test_runner"
    assert classify_readonly_command(_bash(0, "grep -rn 'foo' src/ | head -n 20", "match")) == "shell_inspect"
    assert classify_readonly_command(_bash(0, "echo 'x' > src/foo.py", "")) is None
    assert classify_readonly_command(_bash(0, "git commit -m 'fix'", "")) is None
    assert classify_readonly_command(make_step(0, tool_name="str_replace_editor", raw_args={"command": "view", "path": "/repo/a.py"}, obs="x")) == "read_or_search_tool"
    assert classify_readonly_command(make_step(0, tool_name="str_replace_editor", raw_args={"command": "str_replace", "path": "/repo/a.py", "old_str": "a", "new_str": "b"}, obs="ok")) is None

    steps = [
        _bash(0, "git diff", "diff --git a/x b/x"),
        _bash(1, "rm -f /tmp/foo", ""),
    ]
    assert is_readonly_command_step(steps[0]) is True
    assert is_readonly_command_step(steps[1]) is False


def test_apply_verdict_rule() -> None:
    assert apply_verdict_rule(0.45, 1.48, 2.69, 0.0001, True) == "higher for open"
    assert apply_verdict_rule(1.90, 6.60, 1.41, 0.33, True) == "mixed"
    assert apply_verdict_rule(-0.50, 2.10, 0.92, 0.84, True) == "null"
    assert apply_verdict_rule(-1.80, -0.20, 0.45, 0.01, True) == "lower for open"
    assert apply_verdict_rule(-0.80, 0.10, 0.03, 0.017, False) == "null"


def test_parquet_roundtrip_preserves_phase3_fields(tmp_path: Path) -> None:
    steps = [
        _bash(0, "git status", "clean"),
        _bash(1, "git status", "clean"),
        _bash(2, "git status", "clean"),
    ]
    rec = _analyze_single_trajectory(
        dataset="swe_rebench",
        run_id="r1",
        harness="swelike",
        model="glm-5.2",
        task_name="task_1",
        succeeded=True,
        steps=steps,
        initial_prompt_chars=500,
        reported_cost_usd=0.25,
        reported_uncached_in=500.0,
        reported_cached_in=500.0,
        reported_out=200.0,
        reported_total_in=1000.0,
        disable_cache_discount=False,
        compaction_reason=None,
        has_rsc=False,
        cohort="swe_rebench_swelike",
    )
    assert rec.has_strict_loop is True
    assert rec.strict_loop_steps == 2
    assert rec.exact_loop_steps_total == 2
    assert rec.exact_loop_steps_readonly == 2

    pq_file = tmp_path / "cache.parquet"
    meta_file = tmp_path / "cache.json"
    _records_to_parquet([rec], pq_file, {"test": 1}, meta_file)
    loaded_recs, loaded_meta = _records_from_parquet(pq_file, meta_file)
    assert len(loaded_recs) == 1
    r2 = loaded_recs[0]
    assert r2.has_strict_loop is True
    assert r2.strict_loop_steps == 2
    assert r2.exact_loop_steps_total == 2
    assert r2.exact_loop_steps_readonly == 2
    assert r2.readonly_exact_by_cat.get("git_readonly") == 2
    assert loaded_meta == {"test": 1}
