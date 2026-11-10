"""Cache-aware per-step cost rebuild, rescaling, and validation filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_loops.canonicalize import NormalizedStep

# Pre-registered mismatch bounds (set a priori before inspecting dataset rebuild errors):
# A trajectory is retained in dollar analysis only if raw_estimated / reported falls in [0.50, 2.00].
REBUILD_MIN_RATIO = 0.50
REBUILD_MAX_RATIO = 2.00

CHARS_PER_TOKEN = 4.0
FALLBACK_EMPTY_PROMPT_TOKENS = 500.0

SUBAGENT_TOOLS: frozenset[str] = frozenset(
    {
        "task",
        "taskoutput",
        "agent",
        "call_llm_batch",
        "spawn_agent",
        "subagent",
        "delegate",
    }
)

COMPACTION_PHRASES: tuple[str, ...] = (
    "ran out of context",
    "continued from a previous conversation",
    "this session is being continued",
    "compacting conversation",
    "history has been compacted",
    "context_compacted",
    "summary of previous conversation",
    "conversation was compressed",
    "previous conversation summary",
)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD per 1M tokens: (input, cached_input, output)."""

    input_per_m: float
    cached_per_m: float
    output_per_m: float


# Standard public API rates ($ / 1M tokens)
MODEL_PRICING: dict[str, ModelPricing] = {
    # Anthropic
    "claude-haiku-4-5-20251001@anthropic": ModelPricing(1.00, 0.10, 5.00),
    "claude-sonnet-4-5-20250929@anthropic": ModelPricing(3.00, 0.30, 15.00),
    "claude-opus-4-1-20250805@anthropic": ModelPricing(15.00, 1.50, 75.00),
    "claude-opus-4-5-20251101@anthropic": ModelPricing(5.00, 0.50, 25.00),
    "claude-opus-4-6@anthropic": ModelPricing(5.00, 0.50, 25.00),
    "claude-opus-4.6@anthropic": ModelPricing(5.00, 0.50, 25.00),
    "claude-opus-5": ModelPricing(5.00, 0.50, 25.00),
    "claude-sonnet-5": ModelPricing(3.00, 0.30, 15.00),
    "claude-fable-5": ModelPricing(3.00, 0.30, 15.00),
    "fable-5": ModelPricing(3.00, 0.30, 15.00),
    # OpenAI
    "gpt-5@openai": ModelPricing(1.25, 0.125, 10.00),
    "gpt-5-codex@openai": ModelPricing(1.25, 0.125, 10.00),
    "gpt-5.1@openai": ModelPricing(1.25, 0.125, 10.00),
    "gpt-5.1-codex@openai": ModelPricing(1.25, 0.125, 10.00),
    "gpt-5.2@openai": ModelPricing(1.75, 0.175, 14.00),
    "gpt-5.2-codex@openai": ModelPricing(1.75, 0.175, 14.00),
    "gpt-5-mini@openai": ModelPricing(0.25, 0.025, 2.00),
    "gpt-5-nano@openai": ModelPricing(0.05, 0.005, 0.40),
    "gpt-5.6-sol": ModelPricing(1.75, 0.175, 14.00),
    "gpt-5.6-luna": ModelPricing(0.40, 0.04, 3.20),
    "composer-2.5": ModelPricing(1.25, 0.125, 10.00),
    # Google Gemini
    "gemini-2.5-flash@gemini": ModelPricing(0.30, 0.075, 2.50),
    "gemini-2.5-pro@gemini": ModelPricing(1.25, 0.3125, 10.00),
    "gemini-3-flash-preview@gemini": ModelPricing(0.50, 0.05, 3.00),
    "gemini-3-pro-preview@gemini": ModelPricing(2.00, 0.20, 12.00),
    "gemini-3.1-pro-preview@google": ModelPricing(2.00, 0.20, 12.00),
    # xAI
    "grok-4-0709@xai": ModelPricing(3.00, 0.75, 15.00),
    "grok-code-fast-1@xai": ModelPricing(0.20, 0.02, 1.50),
    "x-ai/grok-4.5": ModelPricing(3.00, 0.75, 15.00),
    "grok-4.5": ModelPricing(3.00, 0.75, 15.00),
    # DeepSeek
    "deepseek-v3.2@deepseek": ModelPricing(0.28, 0.028, 0.42),
    "deepseek/deepseek-v4-pro": ModelPricing(0.55, 0.055, 2.20),
    # Together AI (prompt caching generally disabled in TB metadata: cache_tokens=0.0)
    "qwen/qwen3-coder-480b-a35b-instruct-fp8@together_ai": ModelPricing(2.00, 2.00, 2.00),
    "moonshotai/kimi-k2-instruct-0905@together_ai": ModelPricing(1.00, 1.00, 3.00),
    "moonshotai/kimi-k2-thinking@together_ai": ModelPricing(1.20, 1.20, 4.00),
    "openai/gpt-oss-120b@together_ai": ModelPricing(0.15, 0.15, 0.60),
    "openai/gpt-oss-20b@together_ai": ModelPricing(0.05, 0.05, 0.20),
    # Fireworks / MiniMax / Z-AI / Qwen / Xiaomi
    "accounts/fireworks/models/glm-4p6@fireworks_ai": ModelPricing(0.55, 0.055, 2.19),
    "accounts/fireworks/models/minimax-m2@fireworks_ai": ModelPricing(0.30, 0.30, 1.20),
    "minimax-m2.1@minimax": ModelPricing(0.30, 0.03, 1.20),
    "minimax-m2.1@openai": ModelPricing(0.30, 0.03, 1.20),
    "minimax-m2.5@minimax": ModelPricing(0.30, 0.03, 1.20),
    "minimax/minimax-m3": ModelPricing(0.30, 0.03, 1.20),
    "glm-5@z-ai": ModelPricing(0.80, 0.16, 2.56),
    "z-ai/glm-5.2": ModelPricing(0.80, 0.16, 2.56),
    "xiaomi/mimo-v2.5-pro": ModelPricing(0.50, 0.05, 2.00),
    "qwen/qwen3.5-35b-a3b": ModelPricing(0.25, 0.025, 1.00),
    "qwen/qwen3.6-27b": ModelPricing(0.25, 0.025, 1.00),
    "qwen/qwen3.6-35b-a3b": ModelPricing(0.25, 0.025, 1.00),
}

DEFAULT_PRICING = ModelPricing(1.50, 0.15, 6.00)


def get_model_pricing(model_name: str, *, disable_cache_discount: bool = False) -> ModelPricing:
    """Lookup model pricing; optionally disable cache discount if harness reports zero cache."""
    key = (model_name or "").strip().lower()
    p = MODEL_PRICING.get(key, DEFAULT_PRICING)
    if disable_cache_discount:
        return ModelPricing(p.input_per_m, p.input_per_m, p.output_per_m)
    return p


@dataclass(slots=True)
class TrajectoryCostRebuild:
    """Result of per-step cache-aware cost rebuild and validation."""

    included: bool
    drop_reason: str | None
    step_costs_usd: list[float]
    total_cost_usd: float
    raw_cost_usd: float
    raw_input_all_tokens: float
    raw_output_tokens: float
    reported_cost_usd: float | None
    mismatch_ratio: float | None


def detect_compaction_or_subagents(
    steps: list[NormalizedStep],
    *,
    extra_texts: list[str] | None = None,
    aux_models_tokens: int = 0,
    has_multiple_prompts: bool = False,
) -> str | None:
    """Return exclusion reason ('subagents' or 'compaction') if detected, else None."""
    if aux_models_tokens and aux_models_tokens > 0:
        return "subagents"
    for s in steps:
        if s.has_tool and s.tool_name in SUBAGENT_TOOLS:
            return "subagents"
    if has_multiple_prompts:
        return "compaction"
    for s in steps:
        txt = (s.assistant_text or "").lower()
        if any(ph in txt for ph in COMPACTION_PHRASES):
            return "compaction"
    for et in extra_texts or ():
        low = et.lower()
        if any(ph in low for ph in COMPACTION_PHRASES):
            return "compaction"
    return None


def rebuild_trajectory_costs(
    steps: list[NormalizedStep],
    *,
    model_name: str,
    initial_prompt_chars: int,
    reported_cost_usd: float | None,
    reported_uncached_input_tokens: float | None = None,
    reported_cached_input_tokens: float | None = None,
    reported_output_tokens: float | None = None,
    reported_total_input_tokens: float | None = None,
    disable_cache_discount: bool = False,
    compaction_or_subagent_reason: str | None = None,
) -> TrajectoryCostRebuild:
    """Rebuild per-step cache-aware costs, validate against reported totals, and rescale."""
    n = len(steps)
    zeros = [0.0] * n

    if compaction_or_subagent_reason is not None:
        return TrajectoryCostRebuild(
            included=False,
            drop_reason=compaction_or_subagent_reason,
            step_costs_usd=zeros,
            total_cost_usd=0.0,
            raw_cost_usd=0.0,
            raw_input_all_tokens=0.0,
            raw_output_tokens=0.0,
            reported_cost_usd=reported_cost_usd,
            mismatch_ratio=None,
        )

    if n == 0:
        return TrajectoryCostRebuild(
            included=False,
            drop_reason="empty_steps",
            step_costs_usd=zeros,
            total_cost_usd=0.0,
            raw_cost_usd=0.0,
            raw_input_all_tokens=0.0,
            raw_output_tokens=0.0,
            reported_cost_usd=reported_cost_usd,
            mismatch_ratio=None,
        )

    pricing = get_model_pricing(model_name, disable_cache_discount=disable_cache_discount)

    # Cumulative prefix model:
    # Step 0 sees P_0 new tokens and 0 cached prefix.
    # Step t>0 sees ctx[t-1] + out[t-1] as cached prefix, and obs[t-1] as new uncached input.
    p0 = (
        initial_prompt_chars / CHARS_PER_TOKEN
        if initial_prompt_chars > 0
        else FALLBACK_EMPTY_PROMPT_TOKENS
    )

    raw_cached = [0.0] * n
    raw_new_in = [0.0] * n
    raw_out = [0.0] * n
    raw_costs = [0.0] * n

    ctx = 0.0
    prev_out = 0.0
    prev_obs = 0.0
    for t, s in enumerate(steps):
        if t == 0:
            cached_t = 0.0
            new_in_t = p0
        else:
            cached_t = ctx + prev_out
            new_in_t = max(1.0, prev_obs)
        out_t = max(1.0, s.msg_chars / CHARS_PER_TOKEN)

        raw_cached[t] = cached_t
        raw_new_in[t] = new_in_t
        raw_out[t] = out_t
        raw_costs[t] = (
            cached_t * pricing.cached_per_m
            + new_in_t * pricing.input_per_m
            + out_t * pricing.output_per_m
        ) / 1e6

        ctx = cached_t + new_in_t
        prev_out = out_t
        prev_obs = s.obs_chars / CHARS_PER_TOKEN

    raw_total_cost = sum(raw_costs)
    raw_total_cached = sum(raw_cached)
    raw_total_new_in = sum(raw_new_in)
    raw_total_in_all = raw_total_cached + raw_total_new_in
    raw_total_out = sum(raw_out)

    # Validate rebuild against reported cost (when available and > 0) or reported input tokens
    has_rep_cost = reported_cost_usd is not None and reported_cost_usd > 0.0
    has_rep_toks = (
        reported_total_input_tokens is not None and reported_total_input_tokens > 0.0
    )

    if not has_rep_cost and not has_rep_toks:
        return TrajectoryCostRebuild(
            included=False,
            drop_reason="missing_reported_cost_and_tokens",
            step_costs_usd=zeros,
            total_cost_usd=0.0,
            raw_cost_usd=raw_total_cost,
            raw_input_all_tokens=raw_total_in_all,
            raw_output_tokens=raw_total_out,
            reported_cost_usd=reported_cost_usd,
            mismatch_ratio=None,
        )

    if has_rep_cost:
        ratio = raw_total_cost / reported_cost_usd if reported_cost_usd else float("inf")
    else:
        ratio = raw_total_in_all / reported_total_input_tokens if reported_total_input_tokens else float("inf")

    if ratio < REBUILD_MIN_RATIO or ratio > REBUILD_MAX_RATIO:
        return TrajectoryCostRebuild(
            included=False,
            drop_reason="rebuild_mismatch",
            step_costs_usd=zeros,
            total_cost_usd=0.0,
            raw_cost_usd=raw_total_cost,
            raw_input_all_tokens=raw_total_in_all,
            raw_output_tokens=raw_total_out,
            reported_cost_usd=reported_cost_usd,
            mismatch_ratio=ratio,
        )

    # Rescale step components to match reported tokens when component totals are known,
    # and match reported cost_usd exactly when reported_cost_usd > 0.
    if (
        reported_uncached_input_tokens is not None
        and reported_cached_input_tokens is not None
        and reported_output_tokens is not None
    ):
        scale_c = (reported_cached_input_tokens / raw_total_cached) if raw_total_cached > 0 else 0.0
        scale_i = (reported_uncached_input_tokens / raw_total_new_in) if raw_total_new_in > 0 else 0.0
        scale_o = (reported_output_tokens / raw_total_out) if raw_total_out > 0 else 0.0
        comp_costs = [
            (
                raw_cached[t] * scale_c * pricing.cached_per_m
                + raw_new_in[t] * scale_i * pricing.input_per_m
                + raw_out[t] * scale_o * pricing.output_per_m
            )
            / 1e6
            for t in range(n)
        ]
        comp_sum = sum(comp_costs)
        if has_rep_cost and comp_sum > 0:
            cscale = reported_cost_usd / comp_sum
            step_costs = [c * cscale for c in comp_costs]
        else:
            step_costs = comp_costs
    else:
        scale = (reported_cost_usd / raw_total_cost) if (has_rep_cost and raw_total_cost > 0) else 1.0
        step_costs = [c * scale for c in raw_costs]

    total_usd = sum(step_costs)
    return TrajectoryCostRebuild(
        included=True,
        drop_reason=None,
        step_costs_usd=step_costs,
        total_cost_usd=total_usd,
        raw_cost_usd=raw_total_cost,
        raw_input_all_tokens=raw_total_in_all,
        raw_output_tokens=raw_total_out,
        reported_cost_usd=reported_cost_usd if has_rep_cost else total_usd,
        mismatch_ratio=ratio,
    )
