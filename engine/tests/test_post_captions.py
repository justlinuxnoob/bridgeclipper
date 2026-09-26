"""Offline tests for per-platform post captions."""

import asyncio

from clip_engine.config import Settings
from clip_engine.services import post_captions as pc
from clip_engine.services.claude_code import ClaudeCodeError
from clip_engine.services.intelligence_planner import ClipPlanSegment
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptWord


def transcript():
    words = [TranscriptWord(w, i * 1000, i * 1000 + 800) for i, w in enumerate(
        "Astronauts fly a tiny plush toy to show when they reach zero gravity .".split())]
    return [TranscriptSegment(0, words[-1].end_time_ms, " ".join(w.word for w in words), None, words)]


def clips():
    return [ClipPlanSegment(0, 6000, 0.9, summary="Zero G Toy"), ClipPlanSegment(6000, 13000, 0.8, summary="Weightless")]


def entry(index, **overrides):
    value = {
        "index": index,
        "tiktok": {"caption": "  Why astronauts   bring a plush toy to space ", "hashtags": ["#space", "NASA", "zero gravity", "#fyp!", "extra"]},
        "youtube": {"title": "The <Plush> Toy That Tells Astronauts They're in Space", "description": "A tiny toy floats.", "tags": ["#NASA", "nasa", "space"]},
        "instagram": {"caption": "The toy starts floating: that's zero G.", "hashtags": ["space"]},
    }
    value.update(overrides)
    return value


def test_prompt_has_each_clip_transcript_and_title():
    prompt = pc.build_prompt(clips(), transcript(), "How to Know You're in Space")
    assert "Source video: How to Know You're in Space" in prompt
    assert "### Clip 0\nOn-screen title: Zero G Toy\nLength: 6 seconds\nTranscript: Astronauts fly a tiny plush toy" in prompt
    assert "### Clip 1" in prompt and "reach zero gravity" in prompt.split("### Clip 1")[1]


def test_normalize_enforces_platform_limits():
    result = pc.normalize(entry(0))
    assert result["tiktok"]["caption"] == "Why astronauts bring a plush toy to space"
    # Punctuation and spaces stripped, duplicates dropped, at most three.
    assert result["tiktok"]["hashtags"] == ["#space", "#NASA", "#zerogravity"]
    assert result["youtube"]["title"] == "The Plush Toy That Tells Astronauts They're in Space"
    assert result["youtube"]["tags"] == ["NASA", "space"]
    assert pc.normalize(entry(0, tiktok={"caption": "", "hashtags": []}, youtube={"title": "", "description": "", "tags": []},
                              instagram={"caption": "", "hashtags": []})) is None
    long = pc.normalize(entry(0, youtube={"title": "T" * 300, "description": "é" * 5000, "tags": []}))
    assert len(long["youtube"]["title"]) == 100 and len(long["youtube"]["description"].encode()) <= 4800


def test_claude_code_writes_all_clips_and_costs_nothing(monkeypatch):
    calls = []

    async def fake_run(system, user, schema, **kwargs):
        calls.append(kwargs)
        return {"clips": [entry(1), entry(0), entry(7)]}, {}

    monkeypatch.setattr(pc, "run_claude_json", fake_run)
    settings = Settings(_env_file=None, planner_backend="claude_code")
    captions, cost = asyncio.run(pc.write_post_captions(settings, clips(), transcript(), "Title"))
    assert [c is not None for c in captions] == [True, True]
    assert calls[0]["effort"] == "low" and calls[0]["model"] == "opus"
    assert cost == {"provider": "claude_code", "model": "claude-code/opus", "cost": 0.0}
    text = pc.post_text(captions[0])
    assert text.startswith("=== TikTok ===\nWhy astronauts bring a plush toy to space\n\n#space #NASA #zerogravity")
    assert "=== YouTube Shorts ===\nTitle: The Plush Toy" in text and "=== Instagram Reels ===" in text


def test_invalid_output_retries_once_then_gives_up_quietly(monkeypatch):
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(1)
        raise ClaudeCodeError("bad", "invalid_output")

    monkeypatch.setattr(pc, "run_claude_json", fake_run)
    settings = Settings(_env_file=None, planner_backend="claude_code")
    captions, cost = asyncio.run(pc.write_post_captions(settings, clips(), transcript(), None))
    assert len(calls) == 2 and captions == [None, None] and cost is None


def test_login_failure_is_not_retried_and_never_raises(monkeypatch):
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(1)
        raise ClaudeCodeError("Not logged in", "not_logged_in")

    monkeypatch.setattr(pc, "run_claude_json", fake_run)
    captions, _ = asyncio.run(pc.write_post_captions(Settings(_env_file=None, planner_backend="claude_code"), clips(), transcript(), None))
    assert len(calls) == 1 and captions == [None, None]


def test_skipped_without_a_backend_or_transcript(monkeypatch):
    async def boom(*args, **kwargs):
        raise AssertionError("must not call")

    monkeypatch.setattr(pc, "run_claude_json", boom)
    monkeypatch.setattr(pc, "_via_openrouter", boom)
    no_key = Settings(_env_file=None, planner_backend="openrouter", openrouter_api_key="")
    assert asyncio.run(pc.write_post_captions(no_key, clips(), transcript(), None)) == ([None, None], None)
    claude = Settings(_env_file=None, planner_backend="claude_code")
    assert asyncio.run(pc.write_post_captions(claude, clips(), [], None)) == ([None, None], None)
