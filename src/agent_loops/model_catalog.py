"""Central model metadata catalog across Terminal-Bench, SWE-rebench, Longtail-SWE, and Sierra tau2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Normalized classification and covariate metadata for a model identifier."""

    canonical_name: str
    family: str
    is_open_weight: bool
    release_month: str  # YYYY-MM
    default_serving_route: str  # "first_party" or "third_party"

    @property
    def release_months_since_2025(self) -> int:
        year, month = (int(p) for p in self.release_month.split("-"))
        return (year - 2025) * 12 + (month - 1)


# Explicit closed exceptions that share vendor prefixes with open-weight models
# (e.g., Qwen3.8-Max is proprietary closed weights despite Qwen3.5/3.6/3.8-27B open releases;
# Cursor Composer 2.5 is a proprietary checkpoint despite being built from Kimi K2.5).
_EXPLICIT_MODEL_TABLE: dict[str, ModelMeta] = {
    # Proprietary Qwen Max tiers (Closed)
    "qwen3-8-max": ModelMeta("qwen3-8-max", "qwen", False, "2026-08", "third_party"),
    "qwen3.8-max": ModelMeta("qwen3.8-max", "qwen", False, "2026-08", "third_party"),
    "qwen/qwen3.8-max": ModelMeta("qwen3.8-max", "qwen", False, "2026-08", "third_party"),
    "openrouter/qwen/qwen3.8-max": ModelMeta("qwen3.8-max", "qwen", False, "2026-08", "third_party"),
    # Cursor Composer (Closed)
    "composer-2.5": ModelMeta("composer-2.5", "composer", False, "2026-03", "first_party"),
    # OpenAI Apache-2.0 open-weight models (Open)
    "openai/gpt-oss-120b": ModelMeta("gpt-oss-120b", "gpt_oss", True, "2025-08", "third_party"),
    "openai/gpt-oss-20b": ModelMeta("gpt-oss-20b", "gpt_oss", True, "2025-08", "third_party"),
    "gpt-oss-120b": ModelMeta("gpt-oss-120b", "gpt_oss", True, "2025-08", "third_party"),
    "gpt-oss-20b": ModelMeta("gpt-oss-20b", "gpt_oss", True, "2025-08", "third_party"),
    # Thinking Machines Inkling (Apache-2.0 Open-Weights)
    "inkling": ModelMeta("inkling", "inkling", True, "2026-07", "third_party"),
    "openrouter/thinkingmachines/inkling": ModelMeta("inkling", "inkling", True, "2026-07", "third_party"),
}

# Ordered longest/most-specific pattern matching on stripped lowercase base model ID
_PATTERN_RULES: tuple[tuple[str, ModelMeta], ...] = (
    # Anthropic Claude (Closed)
    ("claude-opus-5.5", ModelMeta("claude-opus-5.5", "claude", False, "2026-09", "first_party")),
    ("claude-opus-5-5", ModelMeta("claude-opus-5.5", "claude", False, "2026-09", "first_party")),
    ("claude-opus-5", ModelMeta("claude-opus-5", "claude", False, "2026-07", "first_party")),
    ("claude-sonnet-5", ModelMeta("claude-sonnet-5", "claude", False, "2026-06", "first_party")),
    ("claude-fable-5", ModelMeta("claude-fable-5", "claude", False, "2026-06", "first_party")),
    ("fable-5", ModelMeta("claude-fable-5", "claude", False, "2026-06", "first_party")),
    ("claude-opus-4-8", ModelMeta("claude-opus-4.8", "claude", False, "2026-05", "first_party")),
    ("claude-opus-4-7", ModelMeta("claude-opus-4.7", "claude", False, "2026-04", "first_party")),
    ("claude-opus-4-6", ModelMeta("claude-opus-4.6", "claude", False, "2026-02", "first_party")),
    ("claude-opus-4.6", ModelMeta("claude-opus-4.6", "claude", False, "2026-02", "first_party")),
    ("claude-opus-4-5", ModelMeta("claude-opus-4.5", "claude", False, "2025-11", "first_party")),
    ("claude-opus-4-1", ModelMeta("claude-opus-4.1", "claude", False, "2025-08", "first_party")),
    ("claude-sonnet-4-5", ModelMeta("claude-sonnet-4.5", "claude", False, "2025-09", "first_party")),
    ("claude-4.5-sonnet", ModelMeta("claude-sonnet-4.5", "claude", False, "2025-09", "first_party")),
    ("claude-haiku-4-5", ModelMeta("claude-haiku-4.5", "claude", False, "2025-10", "first_party")),
    ("claude-3-7-sonnet", ModelMeta("claude-3.7-sonnet", "claude", False, "2025-02", "first_party")),
    ("claude-", ModelMeta("claude-other", "claude", False, "2026-01", "first_party")),
    # Google Gemini (Closed)
    ("gemini-3.1-pro", ModelMeta("gemini-3.1-pro", "gemini", False, "2026-02", "first_party")),
    ("gemini-3-1-pro", ModelMeta("gemini-3.1-pro", "gemini", False, "2026-02", "first_party")),
    ("gemini-3-pro", ModelMeta("gemini-3-pro", "gemini", False, "2025-11", "first_party")),
    ("gemini-3-flash", ModelMeta("gemini-3-flash", "gemini", False, "2025-12", "first_party")),
    ("gemini-2.5-pro", ModelMeta("gemini-2.5-pro", "gemini", False, "2025-06", "first_party")),
    ("gemini-2-5-pro", ModelMeta("gemini-2.5-pro", "gemini", False, "2025-06", "first_party")),
    ("gemini-2.5-flash", ModelMeta("gemini-2.5-flash", "gemini", False, "2025-06", "first_party")),
    ("gemini-", ModelMeta("gemini-other", "gemini", False, "2025-06", "first_party")),
    # OpenAI Proprietary GPT (Closed — gpt-oss handled in _EXPLICIT_MODEL_TABLE)
    ("gpt-6.1-sol", ModelMeta("gpt-6.1-sol", "gpt", False, "2026-09", "first_party")),
    ("gpt-5.6-sol", ModelMeta("gpt-5.6-sol", "gpt", False, "2026-07", "first_party")),
    ("gpt-5-6-sol", ModelMeta("gpt-5.6-sol", "gpt", False, "2026-07", "first_party")),
    ("gpt-5.6-luna", ModelMeta("gpt-5.6-luna", "gpt", False, "2026-07", "first_party")),
    ("gpt-5-5", ModelMeta("gpt-5.5", "gpt", False, "2026-04", "first_party")),
    ("gpt-5-4", ModelMeta("gpt-5.4", "gpt", False, "2026-03", "first_party")),
    ("gpt-5.3-codex", ModelMeta("gpt-5.3-codex", "gpt", False, "2026-02", "first_party")),
    ("gpt-5.2-codex", ModelMeta("gpt-5.2-codex", "gpt", False, "2025-12", "first_party")),
    ("gpt-5.2", ModelMeta("gpt-5.2", "gpt", False, "2025-12", "first_party")),
    ("gpt-5-2-none", ModelMeta("gpt-5.2-none", "gpt", False, "2025-12", "first_party")),
    ("gpt-5-2", ModelMeta("gpt-5.2", "gpt", False, "2025-12", "first_party")),
    ("gpt-5.1-codex-max", ModelMeta("gpt-5.1-codex-max", "gpt", False, "2025-11", "first_party")),
    ("gpt-5.1-codex-mini", ModelMeta("gpt-5.1-codex-mini", "gpt", False, "2025-11", "first_party")),
    ("gpt-5.1-codex", ModelMeta("gpt-5.1-codex", "gpt", False, "2025-11", "first_party")),
    ("gpt-5.1", ModelMeta("gpt-5.1", "gpt", False, "2025-11", "first_party")),
    ("gpt-5-codex", ModelMeta("gpt-5-codex", "gpt", False, "2025-09", "first_party")),
    ("gpt-5-mini", ModelMeta("gpt-5-mini", "gpt", False, "2025-08", "first_party")),
    ("gpt-5-nano", ModelMeta("gpt-5-nano", "gpt", False, "2025-08", "first_party")),
    ("gpt-5", ModelMeta("gpt-5", "gpt", False, "2025-08", "first_party")),
    ("gpt-4.1-mini", ModelMeta("gpt-4.1-mini", "gpt", False, "2025-04", "first_party")),
    ("gpt-4.1", ModelMeta("gpt-4.1", "gpt", False, "2025-04", "first_party")),
    ("o4-mini", ModelMeta("o4-mini", "gpt", False, "2025-04", "first_party")),
    # xAI Grok (Closed)
    ("grok-4.5", ModelMeta("grok-4.5", "grok", False, "2026-07", "first_party")),
    ("grok-4-5", ModelMeta("grok-4.5", "grok", False, "2026-07", "first_party")),
    ("grok-4-2", ModelMeta("grok-4.2", "grok", False, "2026-03", "first_party")),
    ("grok-4-1-fast", ModelMeta("grok-4.1-fast", "grok", False, "2025-11", "first_party")),
    ("grok-4-fast", ModelMeta("grok-4-fast", "grok", False, "2025-09", "first_party")),
    ("grok-code-fast-1", ModelMeta("grok-code-fast-1", "grok", False, "2025-08", "first_party")),
    ("grok-4-0709", ModelMeta("grok-4", "grok", False, "2025-07", "first_party")),
    ("grok-", ModelMeta("grok-other", "grok", False, "2025-07", "first_party")),
    # DeepSeek (Open-Weights)
    ("deepseek-v4-flash", ModelMeta("deepseek-v4-flash", "deepseek", True, "2026-07", "third_party")),
    ("deepseek_v4_flash", ModelMeta("deepseek-v4-flash", "deepseek", True, "2026-07", "third_party")),
    ("deepseek-v4-pro", ModelMeta("deepseek-v4-pro", "deepseek", True, "2026-08", "third_party")),
    ("deepseek-v3.2", ModelMeta("deepseek-v3.2", "deepseek", True, "2025-12", "first_party")),
    # Z.ai / Zhipu GLM (Open-Weights)
    ("glm-5.3-flash", ModelMeta("glm-5.3-flash", "glm", True, "2026-08", "third_party")),
    ("glm_5_3_flash", ModelMeta("glm-5.3-flash", "glm", True, "2026-08", "third_party")),
    ("glm-5.2", ModelMeta("glm-5.2", "glm", True, "2026-06", "third_party")),
    ("glm-5-2", ModelMeta("glm-5.2", "glm", True, "2026-06", "third_party")),
    ("glm-5-think", ModelMeta("glm-5", "glm", True, "2026-02", "third_party")),
    ("glm-5", ModelMeta("glm-5", "glm", True, "2026-02", "first_party")),
    ("glm-4.7", ModelMeta("glm-4.7", "glm", True, "2025-12", "first_party")),
    ("glm-4p6", ModelMeta("glm-4.6", "glm", True, "2025-09", "third_party")),
    # Moonshot Kimi (Open-Weights)
    ("kimi-k3", ModelMeta("kimi-k3", "kimi", True, "2026-07", "third_party")),
    ("kimi-k2.5", ModelMeta("kimi-k2.5", "kimi", True, "2026-01", "first_party")),
    ("kimi-k2-thinking", ModelMeta("kimi-k2-thinking", "kimi", True, "2025-11", "third_party")),
    ("kimi-k2-instruct-0905", ModelMeta("kimi-k2-0905", "kimi", True, "2025-09", "third_party")),
    # MiniMax (Open-Weights)
    ("minimax-m3", ModelMeta("minimax-m3", "minimax", True, "2026-05", "third_party")),
    ("minimax-m2.5", ModelMeta("minimax-m2.5", "minimax", True, "2026-02", "first_party")),
    ("minimax-m2.1", ModelMeta("minimax-m2.1", "minimax", True, "2025-12", "first_party")),
    ("minimax-m2", ModelMeta("minimax-m2", "minimax", True, "2025-10", "third_party")),
    # Qwen Open-Weights (note: Qwen3.8-Max handled in _EXPLICIT_MODEL_TABLE above)
    ("qwen3.8-27b", ModelMeta("qwen3.8-27b", "qwen", True, "2026-08", "third_party")),
    ("qwen_3_8_27b", ModelMeta("qwen3.8-27b", "qwen", True, "2026-08", "third_party")),
    ("qwen3.6-35b-a3b", ModelMeta("qwen3.6-35b-a3b", "qwen", True, "2026-04", "third_party")),
    ("qwen3.6-27b", ModelMeta("qwen3.6-27b", "qwen", True, "2026-04", "third_party")),
    ("qwen_3_6_27b", ModelMeta("qwen3.6-27b", "qwen", True, "2026-04", "third_party")),
    ("qwen3.5-397b-a17b", ModelMeta("qwen3.5-397b-a17b", "qwen", True, "2026-02", "third_party")),
    ("qwen3.5-35b-a3b", ModelMeta("qwen3.5-35b-a3b", "qwen", True, "2026-02", "third_party")),
    ("qwen3-coder-480b", ModelMeta("qwen3-coder-480b", "qwen", True, "2025-07", "third_party")),
    # Xiaomi MiMo (Open-Weights)
    ("mimo-v2.5-pro", ModelMeta("mimo-v2.5-pro", "mimo", True, "2026-04", "third_party")),
)


def _strip_provider_prefix_suffix(model: str) -> tuple[str, str]:
    """Return (core_lower_without_provider_suffix, provider_suffix_lower)."""
    raw = (model or "").strip().lower()
    if not raw:
        raise ValueError("Empty model identifier")
    if "@" in raw:
        left, suffix = raw.rsplit("@", 1)
    else:
        left, suffix = raw, ""
    return left, suffix


def get_model_meta(model: str) -> ModelMeta:
    """Resolve ModelMeta for any model identifier; raises ValueError if unknown."""
    left, _suffix = _strip_provider_prefix_suffix(model)
    tail = left.split("/")[-1]
    for key in (left, tail):
        if key in _EXPLICIT_MODEL_TABLE:
            return _EXPLICIT_MODEL_TABLE[key]
    for pat, meta in _PATTERN_RULES:
        if pat in left:
            return meta
    raise ValueError(f"Unclassified model identifier in model_catalog: {model!r}")


def is_open_weight_model(model: str) -> bool:
    """Return True if model has open/downloadable weights, False if proprietary/closed."""
    return get_model_meta(model).is_open_weight


def get_serving_route(
    dataset: str,
    model: str,
    *,
    litellm_llm: str | None = None,
) -> str:
    """Return 'first_party' or 'third_party' serving route grounded in dataset telemetry.

    - Terminal-Bench encodes '@provider' directly in model names ('together_ai', 'fireworks_ai',
      'modelscope', plus 'minimax-m2.1@openai' are third-party; native APIs are first-party).
    - SWE-rebench 'swelike' routes all 13 models through SWE-rebench's shared inference gateway;
      we preserve metadata default but flag SWE gateway uniformity in regression notes.
    - Longtail-SWE uses official Anthropic/OpenAI APIs for Closed models and hosted/Nebius/OpenRouter
      endpoints for Open models.
    - Sierra tau2 records exact LiteLLM 'agent_info.llm': 'openrouter/*' and self-hosted
      'openai/glm-5-fp8' / 'openai/qwen3.5-397b' are third_party; 'anthropic/*', 'claude-*',
      'vertex_ai/gemini-*', 'openai/responses/*', 'gpt-*', 'xai/*' are first_party.
    """
    meta = get_model_meta(model)
    ds = dataset.lower()
    raw = (model or "").strip().lower()

    if ds == "terminalbench":
        left, suffix = _strip_provider_prefix_suffix(raw)
        if suffix in {"together_ai", "fireworks_ai", "modelscope"}:
            return "third_party"
        if suffix == "openai" and "minimax" in left:
            return "third_party"
        if suffix in {"anthropic", "openai", "gemini", "google", "xai", "deepseek", "z-ai", "minimax", "kimi"}:
            return "first_party"
        return meta.default_serving_route

    if ds.startswith("tau2") or ds.startswith("sierra"):
        ll = (litellm_llm or raw).strip().lower()
        if ll.startswith("openrouter/") or "fp8" in ll or ll.startswith("openai/qwen"):
            return "third_party"
        if ll.startswith(("anthropic/", "claude-", "vertex_ai/", "openai/responses/", "gpt-", "xai/")):
            return "first_party"
        return meta.default_serving_route

    return meta.default_serving_route


# Documented scaffold / configuration step caps by cohort
COHORT_STEP_CAPS: dict[str, int | None] = {
    "tb_terminus_2": 200,  # Note: some TB tasks/configs ran >200 turns; observed up to 535
    "tb_openhands": None,  # Observed up to 3,006 raw events
    "tb_mini_swe_agent": None,  # Observed up to 796 steps
    "swe_rebench_swelike": 500,  # Observed up to 492 steps in swelike
    "longtail_swe_swelike": 500,  # Observed max_steps=501 (500 turns + prompt/final)
    "tau2_classic": 200,  # results.json info.max_steps=200 (message turns), max_errors=10
    "tau2_banking_knowledge_alltools": 200,  # info.max_steps=200, max_errors=10
}
