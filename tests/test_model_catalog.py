"""Tests for centralized model metadata, open/closed splits, family mapping, and serving routes."""

import pytest

from agent_loops.model_catalog import (
    get_model_meta,
    get_serving_route,
    is_open_weight_model,
)


def test_longtail_and_sierra_open_vs_closed_classifications():
    # Open-Weights
    for m in (
        "kimi-k3",
        "moonshotai/Kimi-K3",
        "deepseek-v4-flash",
        "deepseek_v4_flash",
        "deepseek/deepseek-v4-flash-0731",
        "glm-5.3-flash",
        "zai-org/GLM-5.3-Flash",
        "glm-5-2",
        "glm-5-think",
        "qwen3.8-27b",
        "Qwen/Qwen3.8-27B",
        "qwen3.6-27b",
        "qwen_3_6_27b",
        "qwen3.5-397b-a17b-think",
        "inkling",
        "openrouter/thinkingmachines/inkling",
    ):
        assert is_open_weight_model(m) is True, m

    # Closed proprietary — including Qwen 3.8 Max exception
    for m in (
        "qwen3-8-max",
        "openrouter/qwen/qwen3.8-max",
        "claude-opus-5.5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-fable-5",
        "gpt-6.1-sol",
        "gpt-5-6-sol",
        "gpt-5-5",
        "gpt-5-4",
        "gpt-5-2-none",
        "grok-4-5",
        "grok-4-2",
        "grok-4-1-fast",
        "grok-4-fast",
        "gemini-3-1-pro-preview",
        "gemini-2-5-pro",
    ):
        assert is_open_weight_model(m) is False, m


def test_family_resolution_and_dates():
    assert get_model_meta("gemini-3-1-pro-preview").family == "gemini"
    assert get_model_meta("qwen3-8-max").family == "qwen"
    assert get_model_meta("qwen3-8-max").is_open_weight is False
    assert get_model_meta("inkling").family == "inkling"
    assert get_model_meta("openai/gpt-oss-120b@together_ai").family == "gpt_oss"
    assert get_model_meta("claude-opus-5.5").release_month == "2026-09"
    assert get_model_meta("openai/gpt-oss-20b@together_ai").release_months_since_2025 == 7


def test_serving_routes_by_dataset():
    assert get_serving_route("terminalbench", "glm-5@z-ai") == "first_party"
    assert get_serving_route("terminalbench", "minimax-m2.1@openai") == "third_party"
    assert get_serving_route("terminalbench", "minimax-m2.1@minimax") == "first_party"
    assert get_serving_route("terminalbench", "openai/gpt-oss-120b@together_ai") == "third_party"

    assert (
        get_serving_route("tau2_banking_knowledge", "kimi-k3", litellm_llm="openrouter/moonshotai/kimi-k3")
        == "third_party"
    )
    assert (
        get_serving_route("tau2_banking_knowledge", "qwen3-8-max", litellm_llm="openrouter/qwen/qwen3.8-max")
        == "third_party"
    )
    assert (
        get_serving_route(
            "tau2_banking_knowledge",
            "gemini-3-1-pro-preview",
            litellm_llm="vertex_ai/gemini-3.1-pro-preview",
        )
        == "first_party"
    )
    assert (
        get_serving_route("tau2_classic", "glm-5-think", litellm_llm="openai/glm-5-fp8")
        == "third_party"
    )


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model_meta("unseen/imaginary-model-999")
