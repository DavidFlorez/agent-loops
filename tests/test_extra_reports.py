"""Tests for Gemini and Open-vs-Closed classification and report helpers."""

import pytest

from agent_loops.extra_reports import _gemini_generation, is_open_weight_model


def test_open_weight_vs_closed_classification():
    # Open weights (including Apache-2.0 gpt-oss)
    assert is_open_weight_model("openai/gpt-oss-120b@together_ai") is True
    assert is_open_weight_model("openai/gpt-oss-20b@together_ai") is True
    assert is_open_weight_model("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@together_ai") is True
    assert is_open_weight_model("Qwen/Qwen3.6-35B-A3B") is True
    assert is_open_weight_model("minimax/minimax-m3") is True
    assert is_open_weight_model("z-ai/glm-5.2") is True
    assert is_open_weight_model("deepseek/deepseek-v4-pro") is True
    assert is_open_weight_model("xiaomi/mimo-v2.5-pro") is True
    assert is_open_weight_model("accounts/fireworks/models/glm-4p6@fireworks_ai") is True

    # Closed proprietary models
    assert is_open_weight_model("gemini-2.5-flash@gemini") is False
    assert is_open_weight_model("gemini-3.1-pro-preview@Google") is False
    assert is_open_weight_model("claude-opus-4-6@anthropic") is False
    assert is_open_weight_model("claude-fable-5") is False
    assert is_open_weight_model("gpt-5.3-codex@openai") is False
    assert is_open_weight_model("gpt-5.6-sol") is False
    assert is_open_weight_model("x-ai/grok-4.5") is False
    assert is_open_weight_model("composer-2.5") is False


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        is_open_weight_model("totally-unknown-vendor/mystery-model")


def test_gemini_generation_buckets():
    assert _gemini_generation("gemini-2.5-flash@gemini") == "gemini-2.5"
    assert _gemini_generation("gemini-3-flash-preview@Google") == "gemini-3.0"
    assert _gemini_generation("gemini-3-pro-preview@gemini") == "gemini-3.0"
    assert _gemini_generation("gemini-3.1-pro-preview@Google") == "gemini-3.1"
