"""Unit tests for canonicalize.py."""

from agent_loops.canonicalize import (
    canonicalize_tool_call,
    classify_file_op,
    collapse_openhands_tb_steps,
    detect_error_signature,
    hash_observation,
    make_step,
)


def test_canonicalize_strips_whitespace_timestamps_and_temp_paths():
    c1 = canonicalize_tool_call(
        "Bash",
        "  python  /tmp/run_9912a/script.py   --since 2026-03-09T12:00:01Z  ",
    )
    c2 = canonicalize_tool_call(
        "bash",
        "python /tmp/run_7711b/script.py --since 2026-04-01T09:15:30Z",
    )
    assert c1 == c2


def test_canonicalize_json_dict_order_invariant():
    c1 = canonicalize_tool_call("Read", {"offset": 1, "file_path": "/app/main.py"})
    c2 = canonicalize_tool_call("read", '{"file_path": "/app/main.py", "offset": 1}')
    assert c1 == c2


def test_hash_observation_strips_pgid_and_duration_noise():
    o1 = (
        "Command: pytest\nError: (none)\nExit Code: 0\n"
        "Process Group PGID: 4151\n1 passed in 0.12s"
    )
    o2 = (
        "Command: pytest\nError: (none)\nExit Code: 0\n"
        "Process Group PGID: 8821\n1 passed in 0.45s"
    )
    assert hash_observation(o1, strip_noise=True) == hash_observation(o2, strip_noise=True)


def test_benign_gemini_error_none_is_not_an_error():
    obs = (
        "Command: ls -la\nDirectory: (root)\n"
        "Output: README.md\nError: (none)\nExit Code: 0\nSignal: 0"
    )
    is_err, sig = detect_error_signature(obs)
    assert is_err is False
    assert sig is None


def test_detect_error_signature_normalizes_noise():
    obs1 = "Exit code 1\nTraceback (most recent call last):\nModuleNotFoundError: No module named 'foo' at 2026-03-01T10:00:00Z"
    obs2 = "Exit code 1\nTraceback (most recent call last):\nModuleNotFoundError: No module named 'foo' at 2026-03-01T10:01:05Z"
    e1, s1 = detect_error_signature(obs1)
    e2, s2 = detect_error_signature(obs2)
    assert e1 and e2
    assert s1 == s2


def test_classify_file_op_reads_writes_and_bash():
    assert classify_file_op("Read", {"file_path": "/app/a.py"}) == ("read", "/app/a.py", "full")
    assert classify_file_op("Read", {"file_path": "/app/a.py", "offset": 100, "limit": 50}) == (
        "read",
        "/app/a.py",
        "100:50",
    )
    assert classify_file_op("Edit", {"file_path": "/app/a.py"}) == ("write", "/app/a.py", None)
    assert classify_file_op("Bash", {"command": "cat /app/a.py"}) == ("read", "/app/a.py", "cat:full")
    assert classify_file_op("Bash", {"command": "echo hi > /app/a.py"}) == ("write_any", None, None)


def test_collapse_openhands_action_obs_pairs():
    raw = [
        {"src": "agent", "msg": "Running command: ls", "tools": [{"fn": "execute_bash", "cmd": "ls"}], "obs": ""},
        {"src": "agent", "msg": "Command `ls` executed with exit code 0.", "tools": [{"fn": "execute_bash", "cmd": "ls"}], "obs": "a.py"},
        {"src": "agent", "msg": "Command `ls` executed with exit code 0.", "tools": [{"fn": "execute_bash", "cmd": "ls"}], "obs": "a.py"},
    ]
    collapsed = collapse_openhands_tb_steps(raw)
    assert len(collapsed) == 2
    assert collapsed[0]["obs"] == "a.py"


def test_rsc_ref_and_coarse_edit_never_collide():
    s1 = make_step(0, tool_name="bash_command", raw_args="$3a", obs="$3b")
    s2 = make_step(1, tool_name="bash_command", raw_args="$3c", obs="$3d")
    assert not s1.args_known and not s1.obs_known
    assert s1.call_hash != s2.call_hash
    assert s1.obs_hash != s2.obs_hash

    e1 = make_step(0, tool_name="Edit", raw_args="/app/ars.R", obs="ok", is_tb_coarse_schema=True)
    e2 = make_step(1, tool_name="Edit", raw_args="/app/ars.R", obs="ok", is_tb_coarse_schema=True)
    assert not e1.args_known
    assert e1.call_hash != e2.call_hash
    assert e1.file_op == "write" and e1.file_path == "/app/ars.R"
