from __future__ import annotations

from unittest.mock import MagicMock, patch

from manga_pipeline.pipeline.step4_translate import (
    ClaudeTranslator,
    _build_system_prompt,
    _extract_json,
)


def _mock_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_build_system_prompt_contains_glossary_and_style():
    s = _build_system_prompt("사이타마=사이타마", "natural tone")
    assert "사이타마=사이타마" in s
    assert "natural tone" in s


def test_extract_json_plain():
    assert _extract_json('{"translations":[]}') == {"translations": []}


def test_extract_json_with_prose_around():
    txt = 'Here is your JSON:\n{"translations":[{"id":0,"ko":"안녕"}]}\nDone.'
    out = _extract_json(txt)
    assert out is not None
    assert out["translations"][0]["ko"] == "안녕"


def test_extract_json_invalid_returns_none():
    assert _extract_json("not json at all") is None


def test_translate_batch_parses_json_and_uses_cache_control():
    with patch("anthropic.Anthropic") as anth_cls:
        client = MagicMock()
        anth_cls.return_value = client
        client.messages.create.return_value = _mock_response(
            '{"translations":[{"id":0,"ko":"안녕하세요"},{"id":1,"ko":"감사합니다"}]}'
        )

        t = ClaudeTranslator(api_key="sk-test", model="claude-sonnet-4-6")
        out = t.translate_batch(["こんにちは", "ありがとう"], glossary="g", style="s")

        assert out == ["안녕하세요", "감사합니다"]
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "g" in kwargs["system"][0]["text"]
        assert "s" in kwargs["system"][0]["text"]


def test_translate_batch_falls_back_per_line_on_missing_id():
    call_count = {"n": 0}

    def fake_create(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _mock_response('{"translations":[{"id":0,"ko":"안녕"}]}')
        return _mock_response('{"translations":[{"id":0,"ko":"감사"}]}')

    with patch("anthropic.Anthropic") as anth_cls:
        client = MagicMock()
        anth_cls.return_value = client
        client.messages.create.side_effect = fake_create

        t = ClaudeTranslator(api_key="sk-test")
        out = t.translate_batch(["こんにちは", "ありがとう"])

        assert out == ["안녕", "감사"]
        assert call_count["n"] == 2


def test_translate_batch_empty_input():
    with patch("anthropic.Anthropic"):
        t = ClaudeTranslator(api_key="sk-test")
        assert t.translate_batch([]) == []
