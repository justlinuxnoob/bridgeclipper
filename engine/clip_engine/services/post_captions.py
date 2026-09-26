"""
Per-platform post text (TikTok, YouTube Shorts, Instagram Reels) for each planned clip.

Runs alongside rendering, from the clip transcript. Uses Claude Code when it
is the planner backend (subscription, cost 0), otherwise a small OpenRouter
model when a key is set. Failure never fails the job: clips just ship
without post text.
"""

import json
import logging
import re
from typing import Any, Optional

import httpx

from clip_engine.services.claude_code import ClaudeCodeError, run_claude_json, schema_errors

logger = logging.getLogger(__name__)

OPENROUTER_CAPTION_MODEL = "openai/gpt-4.1-mini"
MAX_CLIP_TRANSCRIPT_CHARS = 4000
MAX_HASHTAGS = 3
MAX_YOUTUBE_TAGS = 8

_TEXT = {"type": "string"}
_STRINGS = {"type": "array", "items": {"type": "string"}}
_SOCIAL = {
    "type": "object",
    "properties": {"caption": _TEXT, "hashtags": _STRINGS},
    "required": ["caption", "hashtags"],
    "additionalProperties": False,
}
POST_CAPTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "tiktok": _SOCIAL,
                    "youtube": {
                        "type": "object",
                        "properties": {"title": _TEXT, "description": _TEXT, "tags": _STRINGS},
                        "required": ["title", "description", "tags"],
                        "additionalProperties": False,
                    },
                    "instagram": _SOCIAL,
                },
                "required": ["index", "tiktok", "youtube", "instagram"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["clips"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You write the post text that goes with short vertical video clips on TikTok, YouTube Shorts and Instagram Reels.

Rules for every platform:
- Write in the same language as the clip transcript.
- Be specific and accurate to what is said in the clip. Never invent facts, names, numbers or promises.
- Lead with the hook: the most interesting, concrete point of the clip.
- No clickbait the clip does not deliver, no "wait for it", no generic calls like "follow for more".
- Hashtags: relevant topic tags only, never #fyp, #foryou, #viral, #explore or similar; give them without spaces.

TikTok: a caption of 1-2 short sentences (about 80-150 characters), conversational, the payoff or question first. Up to 3 hashtags.
YouTube Shorts: a title of 40-70 characters (never over 100) with the main topic words near the start, no hashtags in the title. A description of 1-3 sentences that adds context. Up to 5 backend tags (topic variants, names).
Instagram Reels: the key point within the first 125 characters, then optionally one more sentence of context (about 100-300 characters in total). Up to 3 hashtags.

Return ONLY JSON matching the schema, one entry per clip, using each clip's index."""


def _clip_transcript(segments: list, start_ms: int, end_ms: int) -> str:
    words = [
        word.word
        for segment in segments
        for word in (segment.words or [])
        if word.start_time_ms >= start_ms and word.end_time_ms <= end_ms
    ]
    return " ".join(words)[:MAX_CLIP_TRANSCRIPT_CHARS]


def build_prompt(clips: list, transcript_segments: list, video_title: Optional[str]) -> str:
    parts = [f"Source video: {video_title}"] if video_title else []
    for index, clip in enumerate(clips):
        text = _clip_transcript(transcript_segments, clip.start_time_ms, clip.end_time_ms)
        seconds = (clip.end_time_ms - clip.start_time_ms) / 1000
        parts.append(
            f"### Clip {index}\nOn-screen title: {clip.summary or '(none)'}\nLength: {seconds:.0f} seconds\n"
            f"Transcript: {text or '(no speech)'}"
        )
    parts.append(f"Write post text for all {len(clips)} clips (index 0 to {len(clips) - 1}).")
    return "\n\n".join(parts)


def _clean(text: Any, limit: int) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()[:limit].strip()


def _hashtags(values: Any, limit: int = MAX_HASHTAGS) -> list[str]:
    tags: list[str] = []
    for value in values or []:
        tag = re.sub(r"[^\w]", "", str(value), flags=re.UNICODE)
        if tag and f"#{tag}".lower() not in (t.lower() for t in tags):
            tags.append(f"#{tag[:50]}")
        if len(tags) == limit:
            break
    return tags


def normalize(entry: dict) -> Optional[dict]:
    """Enforce platform limits; returns None when nothing usable is left."""
    tiktok, youtube, instagram = entry.get("tiktok") or {}, entry.get("youtube") or {}, entry.get("instagram") or {}
    title = _clean(youtube.get("title"), 100).replace("<", "").replace(">", "")
    tags: list[str] = []
    for tag in youtube.get("tags") or []:
        tag = _clean(tag, 100).lstrip("#")
        if tag and tag.lower() not in (t.lower() for t in tags) and sum(len(t) + 1 for t in tags) + len(tag) <= 450:
            tags.append(tag)
        if len(tags) == MAX_YOUTUBE_TAGS:
            break
    description = str(youtube.get("description") or "").strip()
    while len(description.encode()) > 4800:
        description = description[:-50]
    result = {
        "tiktok": {"caption": _clean(tiktok.get("caption"), 2000), "hashtags": _hashtags(tiktok.get("hashtags"))},
        "youtube": {"title": title, "description": description.replace("<", "").replace(">", ""), "tags": tags},
        "instagram": {"caption": _clean(instagram.get("caption"), 2000), "hashtags": _hashtags(instagram.get("hashtags"))},
    }
    if not (result["tiktok"]["caption"] or result["youtube"]["title"] or result["instagram"]["caption"]):
        return None
    return result


def post_text(captions: dict) -> str:
    """Human-readable copy of all platforms, written next to the clip."""
    def social(block: dict) -> str:
        return "\n\n".join(filter(None, [block["caption"], " ".join(block["hashtags"])]))

    youtube = captions["youtube"]
    return (
        f"=== TikTok ===\n{social(captions['tiktok'])}\n\n"
        f"=== YouTube Shorts ===\nTitle: {youtube['title']}\n\n{youtube['description']}\n\n"
        f"Tags: {', '.join(youtube['tags'])}\n\n"
        f"=== Instagram Reels ===\n{social(captions['instagram'])}\n"
    )


async def _via_openrouter(settings, system: str, user: str) -> tuple[Any, dict]:
    from clip_engine.services.openrouter import chat_completion, json_schema_format, message_text

    payload = {
        "model": OPENROUTER_CAPTION_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 6000,
        "response_format": json_schema_format("post_captions", POST_CAPTIONS_SCHEMA),
        "provider": {"require_parameters": True},
    }
    async with httpx.AsyncClient(
        base_url=settings.openrouter_base_url,
        timeout=httpx.Timeout(180.0, connect=30.0),
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
    ) as client:
        body, usage = await chat_completion(client, payload)
    content, _ = message_text(body)
    value = json.loads(content or "")
    if schema_errors(value, POST_CAPTIONS_SCHEMA):
        raise ValueError("Post captions did not match the schema")
    return value, {"provider": "openrouter", "model": OPENROUTER_CAPTION_MODEL, "cost": usage.get("cost") or 0.0}


async def write_post_captions(settings, clips: list, transcript_segments: list, video_title: Optional[str]) -> tuple[list[Optional[dict]], Optional[dict]]:
    """Returns (captions per clip in plan order, cost info), or Nones when unavailable."""
    empty: list[Optional[dict]] = [None] * len(clips)
    if not clips or not transcript_segments:
        return empty, None
    user = build_prompt(clips, transcript_segments, video_title)
    try:
        if settings.planner_backend == "claude_code":
            value, cost = None, {"provider": "claude_code", "model": f"claude-code/{settings.claude_code_model}", "cost": 0.0}
            for attempt in range(2):
                try:
                    value, _ = await run_claude_json(
                        SYSTEM_PROMPT, user, POST_CAPTIONS_SCHEMA,
                        model=settings.claude_code_model, effort="low",
                        timeout_seconds=min(settings.claude_code_timeout_seconds, 300.0),
                        cli_path=settings.claude_code_cli,
                    )
                    break
                except ClaudeCodeError as e:
                    if e.reason != "invalid_output" or attempt == 1:
                        raise
        elif settings.openrouter_api_key:
            value, cost = await _via_openrouter(settings, SYSTEM_PROMPT, user)
        else:
            return empty, None
    except Exception as e:  # Post text is optional; never fail the run over it.
        logger.warning("Post captions unavailable: %s", type(e).__name__)
        return empty, None

    captions = list(empty)
    for entry in value.get("clips", []):
        index = entry.get("index")
        if isinstance(index, int) and 0 <= index < len(clips) and captions[index] is None:
            captions[index] = normalize(entry)
    logger.info("Post captions written for %d of %d clips", sum(c is not None for c in captions), len(clips))
    return captions, cost
