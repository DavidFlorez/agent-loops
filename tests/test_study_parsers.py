"""Unit tests for Longtail-SWE and Sierra tau2 trajectory parsers and shared cohort extraction."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agent_loops.study import analyze_longtail_swe, analyze_sierra_tau2


def test_analyze_longtail_swe_parses_graded_and_zero_step_runs(tmp_path: Path):
    data_dir = tmp_path / "longtail_swe" / "data"
    data_dir.mkdir(parents=True)

    traj_ok = json.dumps(
        {
            "trajectory_id": "t_ok",
            "events": [
                {"type": "prompt", "seq": 0, "messages": [{"role": "user", "content": "Fix bug " * 50}]},
                {"type": "tool_call", "tool_call_id": "c1", "tool": "bash", "input": {"command": "pytest"}},
                {"type": "tool_result", "tool_call_id": "c1", "content": "Exit code 1\nAssertionError: x", "exit_code": 1},
                {"type": "tool_call", "tool_call_id": "c2", "tool": "bash", "input": {"command": "pytest"}},
                {"type": "tool_result", "tool_call_id": "c2", "content": "Exit code 1\nAssertionError: x", "exit_code": 1},
            ],
            "usage": {
                "steps": 2,
                "tokens": {"input": 220, "cached_input": 100, "output": 30, "total": 250},
                "cost_usd": None,
            },
        }
    )
    traj_zero = json.dumps(
        {
            "trajectory_id": "t_zero",
            "events": [{"type": "prompt", "seq": 0, "messages": [{"role": "user", "content": "Fix bug"}]}],
            "usage": {"steps": 0, "tokens": {"input": 0, "cached_input": 0, "output": 0, "total": 0}, "cost_usd": None},
        }
    )

    table = pa.table(
        {
            "trajectory_id": ["t_ok", "t_zero"],
            "instance_id": ["repo__1", "repo__1"],
            "canonical_model": ["kimi-k3", "deepseek-v4-flash"],
            "scaffold": ["swelike", "swelike"],
            "resolved": pa.array([1, None], type=pa.int64()),
            "effective_context_tokens": pa.array([262144, 1048576], type=pa.int64()),
            "trajectory": [traj_ok, traj_zero],
        }
    )
    pq.write_table(table, data_dir / "swelike__test.parquet")

    recs, counts = analyze_longtail_swe(tmp_path)
    assert len(recs) == 2
    by_id = {r.run_id: r for r in recs}

    r_ok = by_id["t_ok"]
    assert r_ok.graded is True
    assert r_ok.succeeded is True
    assert r_ok.n_steps == 2
    assert r_ok.has_any_loop is True
    assert r_ok.context_limit_tokens == 262144
    assert r_ok.cohort == "longtail_swe_swelike"
    assert r_ok.cost_included is True
    assert r_ok.cost_source == "tokens_rebuilt_list_rates"

    r_zero = by_id["t_zero"]
    assert r_zero.graded is False
    assert r_zero.n_steps == 0
    assert r_zero.has_any_loop is False
    assert counts["zero_step_runs"] == 1


def test_analyze_sierra_tau2_separates_alltools_classic_and_filters_user_sim(tmp_path: Path):
    base = tmp_path / "sierra_tau2_public"

    # Submission 1: AllTools banking_knowledge (valid) + airline (valid)
    sub1 = base / "kimi-k3_sierra_2026-08-04" / "trajectories"
    bk_sim = sub1 / "banking_knowledge" / "simulations"
    bk_sim.mkdir(parents=True)
    (sub1 / "banking_knowledge" / "results.json").write_text(
        json.dumps(
            {
                "info": {
                    "retrieval_config": "alltools",
                    "user_info": {"llm": "gpt-5.2"},
                    "agent_info": {"llm": "openrouter/moonshotai/kimi-k3"},
                }
            }
        )
    )
    (bk_sim / "s1.json").write_text(
        json.dumps(
            {
                "id": "s1",
                "task_id": "T001",
                "reward_info": {"reward": 0.0},
                "messages": [
                    {"role": "user", "content": "Help me"},
                    {
                        "role": "assistant",
                        "content": "Checking",
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
                        "cost": 0.0,
                        "tool_calls": [{"id": "tc1", "name": "KB_search_bm25", "arguments": {"query": "fee"}}],
                    },
                    {"role": "tool", "id": "tc1", "content": "No match", "error": False},
                    {
                        "role": "assistant",
                        "content": "Checking again",
                        "usage": {"prompt_tokens": 1200, "completion_tokens": 50},
                        "cost": 0.0,
                        "tool_calls": [{"id": "tc2", "name": "KB_search_bm25", "arguments": {"query": "fee"}}],
                    },
                    {"role": "tool", "id": "tc2", "content": "No match", "error": False},
                    {
                        "role": "assistant",
                        "content": "Done, anything else?",
                        "usage": {"prompt_tokens": 1400, "completion_tokens": 30},
                        "cost": 0.0,
                    },
                    {"role": "user", "content": "No thanks"},
                ],
            }
        )
    )

    # Submission 2: terminal_use banking_knowledge (must get different cohort) + legacy gpt-4.1 classic
    sub2 = base / "gemini-3-flash_sierra_2026-03-02" / "trajectories"
    (sub2 / "banking_knowledge" / "simulations").mkdir(parents=True)
    (sub2 / "banking_knowledge" / "results.json").write_text(
        json.dumps(
            {
                "info": {
                    "retrieval_config": "terminal_use",
                    "user_info": {"llm": "gpt-5.2"},
                    "agent_info": {"llm": "vertex_ai/gemini-3-flash-preview"},
                }
            }
        )
    )
    (sub2 / "banking_knowledge" / "simulations" / "s2.json").write_text(
        json.dumps({"id": "s2", "task_id": "T001", "reward_info": {"reward": 1.0}, "messages": []})
    )

    (sub2 / "airline" / "simulations").mkdir(parents=True)
    (sub2 / "airline" / "results.json").write_text(
        json.dumps({"info": {"user_info": {"llm": "gpt-4.1-2025-04-14"}, "agent_info": {"llm": "x"}}})
    )
    (sub2 / "airline" / "simulations" / "s3.json").write_text(
        json.dumps({"id": "s3", "task_id": "0", "reward_info": {"reward": 1.0}, "messages": []})
    )

    recs, counts = analyze_sierra_tau2(tmp_path)
    assert len(recs) == 2
    assert counts["skipped_non_gpt52_user_sim"] == 1

    r1 = next(r for r in recs if r.model == "kimi-k3")
    assert r1.cohort == "tau2_banking_knowledge_alltools"
    assert r1.n_steps == 3  # 2 tool steps + 1 dialogue step
    assert r1.n_tool_steps == 2
    assert r1.has_any_loop is True
    assert r1.loop_steps == 1
    assert r1.cost_included is True
    assert r1.cost_source == "rebuilt_from_turn_tokens_uncached"
    assert r1.total_cost_usd > 0

    r2 = next(r for r in recs if r.model == "gemini-3-flash")
    assert r2.cohort == "tau2_bk_terminal_use"
    assert r2.n_steps == 0
