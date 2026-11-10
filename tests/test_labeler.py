"""Unit tests for the trajectory step labeler on hand-made trajectories."""

from agent_loops.canonicalize import make_step
from agent_loops.labeler import label_trajectory


def _bash(idx: int, cmd: str, obs: str, *, status: str | None = None, exit_code: int | None = None):
    return make_step(
        idx,
        tool_name="Bash",
        raw_args={"command": cmd},
        obs=obs,
        status=status,
        exit_code=exit_code,
    )


def test_exact_loop_and_productive_repeat():
    steps = [
        _bash(0, "pytest tests/test_a.py", "FAILED 1"),
        _bash(1, "pytest tests/test_a.py", "FAILED 1"),  # exact_loop
        _bash(2, "pytest tests/test_a.py", "PASSED 1"),  # productive_repeat (new obs!)
    ]
    labels = label_trajectory(steps)
    assert not labels[0].is_loop
    assert "exact_loop" in labels[1].labels
    assert labels[1].is_loop is True
    assert "productive_repeat" in labels[2].labels
    assert labels[2].is_loop is False


def test_near_duplicate_vs_normal_test_progression():
    # Near-duplicate: minor tweak with same stalled empty/error observation
    loop_steps = [
        _bash(0, "grep -rn TargetSymbol /repo/packages/core/src", ""),
        _bash(1, "grep -rn TargetSymbol /repo/packages/core/src/", ""),
    ]
    # Note: stripping whitespace doesn't collapse trailing slash, tokens differ at 1 of 6
    # Actually let's use a flag/arg tweak that yields >=0.80 token Jaccard & >=0.85 char sim
    loop_steps = [
        _bash(0, "python /app/train.py --lr 0.01 --epochs 10 --batch 32 --seed 1", "RuntimeError: OOM", exit_code=1),
        _bash(1, "python /app/train.py --lr 0.01 --epochs 10 --batch 16 --seed 1", "RuntimeError: OOM", exit_code=1),
    ]
    lbls = label_trajectory(loop_steps)
    assert "near_duplicate" in lbls[1].labels
    assert lbls[1].is_loop

    # Non-loop: running two different test files should NOT trigger near_duplicate
    normal_steps = [
        _bash(0, "pytest tests/test_alpha.py", "1 passed in 0.10s"),
        _bash(1, "pytest tests/test_beta.py", "1 passed in 0.12s"),
    ]
    nlbls = label_trajectory(normal_steps)
    assert "near_duplicate" not in nlbls[1].labels
    assert not nlbls[1].is_loop


def test_aba_thrash_vs_productive_edit_test():
    # Real A-B-A thrash (unchanged obs on A)
    thrash = [
        _bash(0, "cat /app/a.py", "content A"),
        _bash(1, "cat /app/b.py", "content B"),
        _bash(2, "cat /app/a.py", "content A"),
    ]
    lbls = label_trajectory(thrash)
    assert "aba_thrash" in lbls[2].labels
    assert lbls[2].is_loop

    # Normal pytest -> Edit -> pytest (PASS!) must NOT be labeled aba_thrash
    normal = [
        _bash(0, "pytest", "FAILED 1"),
        make_step(1, tool_name="Edit", raw_args={"file_path": "/app/a.py", "old_string": "x", "new_string": "y"}, obs="ok"),
        _bash(2, "pytest", "PASSED 1"),
    ]
    nlbls = label_trajectory(normal)
    assert "aba_thrash" not in nlbls[2].labels
    assert "productive_repeat" in nlbls[2].labels
    assert not nlbls[2].is_loop


def test_period_k_rotation_up_to_5_and_not_6():
    # Period-3 rotation: A, B, C, A, B, C
    rot3 = [
        _bash(0, "cmd A", "out A"),
        _bash(1, "cmd B", "out B"),
        _bash(2, "cmd C", "out C"),
        _bash(3, "cmd A", "out A"),
        _bash(4, "cmd B", "out B"),
        _bash(5, "cmd C", "out C"),
    ]
    lbls = label_trajectory(rot3)
    for idx in (3, 4, 5):
        assert "period_k_rotation" in lbls[idx].labels
        assert lbls[idx].rotation_k == 3
        assert lbls[idx].primary_label == "period_k_rotation"

    # Period-6 should NOT be labeled period_k_rotation (k <= 5 constraint)
    rot6 = [
        _bash(i, f"cmd {i % 6}", f"out {i % 6}")
        for i in range(12)
    ]
    lbls6 = label_trajectory(rot6)
    assert all("period_k_rotation" not in l.labels for l in lbls6)


def test_error_streak_across_different_calls_broken_by_success():
    steps = [
        _bash(0, "python run_v1.py", "ModuleNotFoundError: No module named 'scipy'", exit_code=1),
        _bash(1, "python run_v2.py", "ModuleNotFoundError: No module named 'scipy'", exit_code=1),
        _bash(2, "pip list", "numpy 2.0", exit_code=0),
        _bash(3, "python run_v2.py", "ModuleNotFoundError: No module named 'scipy'", exit_code=1),
    ]
    lbls = label_trajectory(steps)
    assert "error_streak" not in lbls[0].labels
    assert "error_streak" in lbls[1].labels
    assert not lbls[2].is_loop
    assert "error_streak" not in lbls[3].labels  # streak broken by step 2


def test_redundant_reread_cleared_by_write_and_respects_ranges():
    steps = [
        make_step(0, tool_name="Read", raw_args={"file_path": "/app/foo.py", "offset": 1, "limit": 100}, obs="lines 1..100"),
        make_step(1, tool_name="Read", raw_args={"file_path": "/app/foo.py", "offset": 101, "limit": 100}, obs="lines 101..200"),
        make_step(2, tool_name="Read", raw_args={"file_path": "/app/foo.py", "offset": 1, "limit": 100}, obs="lines 1..100"),
        make_step(3, tool_name="Edit", raw_args={"file_path": "/app/foo.py", "old_string": "a", "new_string": "b"}, obs="ok"),
        make_step(4, tool_name="Read", raw_args={"file_path": "/app/foo.py", "offset": 1, "limit": 100}, obs="lines 1..100 updated"),
    ]
    lbls = label_trajectory(steps)
    assert "redundant_reread" not in lbls[1].labels  # different range
    assert "redundant_reread" in lbls[2].labels      # same range, no write in between
    assert "redundant_reread" not in lbls[4].labels  # write at step 3 cleared read state
