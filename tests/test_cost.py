"""Unit tests for cache-aware cost rebuild and exclusion filters."""

from agent_loops.canonicalize import make_step
from agent_loops.cost import (
    detect_compaction_or_subagents,
    rebuild_trajectory_costs,
)


def _step(idx: int, msg_len: int, obs_len: int, tool: str = "Bash"):
    return make_step(
        idx,
        tool_name=tool,
        raw_args={"command": "x" * max(1, msg_len - 10)},
        obs="y" * obs_len,
        msg_chars_override=msg_len,
        obs_chars_override=obs_len,
    )


def test_cache_aware_rebuild_growth_and_rescale():
    # 4 steps, each adding 400 chars (100 tokens) of msg and 800 chars (200 tokens) of obs
    steps = [_step(i, 400, 800) for i in range(4)]
    # Expected raw tokens at CHARS_PER_TOKEN=4:
    # P0 = 1000 chars = 250 tokens
    # t=0: cached=0, new_in=250, out=100, ctx_after=250
    # t=1: cached=350, new_in=200, out=100, ctx_after=550
    # t=2: cached=650, new_in=200, out=100, ctx_after=850
    # t=3: cached=950, new_in=200, out=100
    # total raw_cached = 1950, raw_new_in = 850, raw_out = 400
    # For claude-sonnet-4-5 (in=$3, cached=$0.30, out=$15 per 1M):
    # raw_cost = (1950*0.30 + 850*3.00 + 400*15.00)/1e6 = (585 + 2550 + 6000)/1e6 = 0.009135
    res = rebuild_trajectory_costs(
        steps,
        model_name="claude-sonnet-4-5-20250929@anthropic",
        initial_prompt_chars=1000,
        reported_cost_usd=0.010,  # within [0.5, 2.0]
    )
    assert res.included is True
    assert abs(res.total_cost_usd - 0.010) < 1e-9
    # Step 3 must cost more than step 1 due to cumulative cached prefix
    assert res.step_costs_usd[3] > res.step_costs_usd[1]


def test_rebuild_drops_mismatch_beyond_pre_registered_margin():
    steps = [_step(i, 400, 800) for i in range(4)]
    # Reported cost 10x higher than raw rebuild -> ratio < 0.50 -> dropped
    res = rebuild_trajectory_costs(
        steps,
        model_name="claude-sonnet-4-5-20250929@anthropic",
        initial_prompt_chars=1000,
        reported_cost_usd=0.10,
    )
    assert res.included is False
    assert res.drop_reason == "rebuild_mismatch"


def test_excludes_subagents_and_compaction():
    steps_sub = [make_step(0, tool_name="Task", raw_args={}, obs="done")]
    assert detect_compaction_or_subagents(steps_sub) == "subagents"
    assert detect_compaction_or_subagents([], aux_models_tokens=100) == "subagents"

    steps_cmp = [
        make_step(0, tool_name="Bash", raw_args="ls", obs="ok", assistant_text="This session is being continued from a previous conversation that ran out of context.")
    ]
    assert detect_compaction_or_subagents(steps_cmp) == "compaction"
    assert detect_compaction_or_subagents([], has_multiple_prompts=True) == "compaction"
