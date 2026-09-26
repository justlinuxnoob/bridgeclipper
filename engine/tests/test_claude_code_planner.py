"""
Offline tests for the Claude Code planner backend. The `claude` CLI is replaced
by a fake subprocess, so no login or network is needed.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from clip_engine.config import Settings
from clip_engine.error_policy import safe_failure_code, safe_job_error_text, safe_processing_error
from clip_engine.services import claude_code
from clip_engine.services.claude_code import ClaudeCodeError, find_claude_cli, run_claude_json, schema_errors
from clip_engine.services.intelligence_planner import (
    CLIP_PLAN_SCHEMA,
    ClaudeCodePlanningError,
    IntelligencePlannerService,
    VisionFrame,
)
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptWord, TranscriptionResult


def make_transcript(seconds: int = 300) -> TranscriptionResult:
    segments = []
    for start in range(0, seconds, 5):
        words = [
            TranscriptWord(word=f"word{start}", start_time_ms=start * 1000, end_time_ms=start * 1000 + 2000),
            TranscriptWord(word="end.", start_time_ms=start * 1000 + 2000, end_time_ms=start * 1000 + 4500),
        ]
        segments.append(TranscriptSegment(
            start_time_ms=start * 1000, end_time_ms=start * 1000 + 4500, text=f"word{start} end.", words=words,
        ))
    return TranscriptionResult(segments=segments, full_text="", duration_seconds=seconds)


def clip(start, end, scores=(8, 8, 8, 8, 8), summary="Great Title Here"):
    return {
        "start_time": start, "end_time": end, "summary": summary,
        "scores": dict(zip(("hook", "standalone", "arc", "quotability", "ending"), scores)),
        "tags": ["tag"],
    }


def make_planner(key="", **overrides) -> IntelligencePlannerService:
    planner = IntelligencePlannerService()
    planner.settings = Settings(_env_file=None, openrouter_api_key=key, planner_backend="claude_code", **overrides)
    return planner


def cli_body(plan=None, *, result=None, is_error=False, status=None):
    body = {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": result if result is not None else json.dumps(plan),
        "usage": {"input_tokens": 1200, "cache_read_input_tokens": 300, "output_tokens": 400},
        "modelUsage": {"claude-opus-5-5": {}},
        "total_cost_usd": 0.42,
    }
    if plan is not None and result is None:
        body["structured_output"] = plan
    if status is not None:
        body["api_error_status"] = status
    return body


def valid_plan(*clips):
    return {"insights": "Podcast/Interview: sharp opinions", "clips": [
        {**c, "emphasis": ["sharp"]} for c in clips or [clip(10, 40)]
    ]}


class FakeProcess:
    def __init__(self, stdout: bytes, hang: bool = False):
        self._stdout = stdout
        self._hang = hang
        self.returncode = None
        self.killed = False
        self.stdin_text = None

    async def communicate(self, data):
        self.stdin_text = data.decode()
        if self._hang:
            await asyncio.Event().wait()
        self.returncode = 0
        return self._stdout, b""

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """Queue CLI responses; records the argv, cwd, env and stdin of every call."""
    cli = tmp_path / "bin" / "claude"
    cli.parent.mkdir()
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    responses, calls = [], []

    async def fake_exec(program, *args, cwd=None, env=None, **kwargs):
        item = responses.pop(0)
        process = item if isinstance(item, FakeProcess) else FakeProcess(json.dumps(item).encode())
        calls.append({
            "program": program, "args": list(args), "cwd": cwd, "env": env,
            "cwd_entries": os.listdir(cwd), "process": process,
        })
        return process

    monkeypatch.setattr(claude_code.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    real_sleep = asyncio.sleep
    monkeypatch.setattr("clip_engine.services.intelligence_planner.asyncio.sleep", lambda _: real_sleep(0))
    return SimpleNamespace(path=str(cli), responses=responses, calls=calls)


def plan(planner, **kwargs):
    return asyncio.run(planner.plan_clips(
        transcript_result=kwargs.pop("transcript", make_transcript(300)),
        video_metadata=SimpleNamespace(duration_seconds=300),
        max_clips=3,
        auto_clip_count=False,
        min_duration_seconds=15,
        max_duration_seconds=60,
        **kwargs,
    ))


class TestCliInvocation:
    def test_runs_tool_less_opus_in_empty_dir_with_login_env_and_no_api_keys(self, fake_cli):
        fake_cli.responses.append(cli_body(valid_plan()))
        planner = make_planner(claude_code_cli=fake_cli.path)
        result = plan(planner)

        call = fake_cli.calls[0]
        args = call["args"]
        assert call["program"] == fake_cli.path
        assert args[:3] == ["-p", "--output-format", "json"]
        assert args[args.index("--model") + 1] == "opus"
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--setting-sources") + 1] == ""
        assert {"--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"} <= set(args)
        assert json.loads(args[args.index("--json-schema") + 1]) == CLIP_PLAN_SCHEMA
        assert "AI-Clipping-Agent" in args[args.index("--system-prompt") + 1]
        assert "--bare" not in args  # --bare would ignore the subscription login
        # Fresh empty temp dir, so no project CLAUDE.md is picked up; removed afterwards.
        assert call["cwd_entries"] == [] and "bridgeclip-claude-" in call["cwd"]
        assert not os.path.exists(call["cwd"])
        env = call["env"]
        assert env["HOME"] == os.environ["HOME"]
        assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(fake_cli.path)
        assert "OPENROUTER_API_KEY" not in env and "ANTHROPIC_API_KEY" not in env
        # Prompt goes over stdin: transcript plus the JSON-only instruction.
        stdin = call["process"].stdin_text
        assert "Here is the transcript of the video" in stdin
        assert "Return ONLY a JSON object that matches this JSON Schema" in stdin

        # The planner's plan went through the normal parser (the end snaps to a sentence end).
        assert [s.start_time_ms for s in result.segments] == [10_000]
        assert result.segments[0].end_time_ms >= 40_000
        costs = result.api_costs
        assert (costs.provider, costs.model, costs.estimated_cost_usd, costs.attempts) == ("claude_code", "claude-opus-5-5", 0.0, 1)
        assert costs.prompt_tokens == 1500 and costs.completion_tokens == 400

    def test_frames_are_not_sent_when_there_is_a_transcript(self, fake_cli, tmp_path):
        fake_cli.responses.append(cli_body(valid_plan()))
        frame = tmp_path / "f.jpg"
        frame.write_bytes(b"j" * 1200)
        planner = make_planner(key="sk-or", claude_code_cli=fake_cli.path)
        plan(planner, frames=[VisionFrame(t * 1000, str(frame), 512, 288) for t in (10, 20, 30)])
        stdin = fake_cli.calls[0]["process"].stdin_text
        assert "base64" not in stdin and "sample frames" not in stdin

    def test_longform_uses_longform_schema(self, fake_cli):
        planner = make_planner(claude_code_cli=fake_cli.path)
        planner._current_longform = True
        fake_cli.responses.append(cli_body(result="{}"))
        with pytest.raises(ClaudeCodePlanningError):
            asyncio.run(planner._call_claude_code([
                {"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "u"}]},
            ]))
        schema = json.loads(fake_cli.calls[0]["args"][fake_cli.calls[0]["args"].index("--json-schema") + 1])
        assert "chapters" in schema["properties"]["clips"]["items"]["required"]


class TestRetriesAndErrors:
    def test_invalid_json_is_retried_once_then_succeeds(self, fake_cli):
        fake_cli.responses += [cli_body(result="Sure! Here are some clips."), cli_body(valid_plan())]
        result = plan(make_planner(claude_code_cli=fake_cli.path))
        assert len(fake_cli.calls) == 2
        assert result.api_costs.attempts == 2 and result.api_costs.estimated_cost_usd == 0.0

    def test_schema_mismatch_twice_fails_with_clear_error(self, fake_cli):
        bad = {"clips": [{"start_time": 10}]}
        fake_cli.responses += [cli_body(result=json.dumps(bad)), cli_body(result=json.dumps(bad))]
        with pytest.raises(ClaudeCodePlanningError) as failure:
            plan(make_planner(claude_code_cli=fake_cli.path))
        assert len(fake_cli.calls) == 2
        assert failure.value.reason == "invalid_output"
        assert safe_processing_error(failure.value) == "Claude Code returned an invalid clip plan"

    def test_fenced_json_in_result_text_is_accepted(self, fake_cli):
        fake_cli.responses.append(cli_body(result=f"```json\n{json.dumps(valid_plan())}\n```"))
        assert len(plan(make_planner(claude_code_cli=fake_cli.path)).segments) == 1

    @pytest.mark.parametrize("text, status, reason", [
        ("Not logged in · Please run /login", None, "not_logged_in"),
        ("API Error: 401 authentication_error", 401, "not_logged_in"),
        ("You've hit your usage limit · resets 7pm", None, "usage_limit"),
        ("Something unexpected", None, "failed"),
    ])
    def test_cli_errors_fail_without_retry(self, fake_cli, text, status, reason):
        fake_cli.responses.append(cli_body(result=text, is_error=True, status=status))
        with pytest.raises(ClaudeCodePlanningError) as failure:
            plan(make_planner(claude_code_cli=fake_cli.path))
        assert len(fake_cli.calls) == 1
        assert failure.value.reason == reason and not failure.value.retryable
        public = safe_processing_error(failure.value)
        assert public.startswith("Claude Code") and safe_job_error_text(public) == public
        assert safe_failure_code(failure.value) == f"planning.claude_code.{reason}"

    def test_timeout_kills_the_cli(self, fake_cli):
        fake_cli.responses.append(FakeProcess(b"", hang=True))
        planner = make_planner(claude_code_cli=fake_cli.path, claude_code_timeout_seconds=0.05)
        with pytest.raises(ClaudeCodePlanningError) as failure:
            plan(planner)
        assert failure.value.reason == "timeout"
        assert fake_cli.calls[0]["process"].killed

    def test_missing_cli(self, tmp_path):
        planner = make_planner(claude_code_cli=str(tmp_path / "nope"))
        with pytest.raises(ClaudeCodePlanningError) as failure:
            plan(planner)
        assert failure.value.reason == "cli_missing"


class TestVisualOnlyFallback:
    def _frames(self, tmp_path):
        frames = []
        for i, second in enumerate((10, 20, 30, 40)):
            path = tmp_path / f"{i}.jpg"
            path.write_bytes(b"j" * 1200)
            frames.append(VisionFrame(second * 1000, str(path), 512, 288))
        return frames

    def test_silent_video_without_openrouter_key_fails_clearly(self, fake_cli, tmp_path):
        with pytest.raises(ClaudeCodePlanningError) as failure:
            plan(make_planner(claude_code_cli=fake_cli.path), transcript=TranscriptionResult(segments=[], full_text=""),
                 frames=self._frames(tmp_path))
        assert failure.value.reason == "needs_speech"
        assert "OpenRouter key" in safe_processing_error(failure.value)
        assert fake_cli.calls == []

    def test_silent_video_with_key_uses_openrouter_vision(self, fake_cli, tmp_path, monkeypatch):
        planner = make_planner(key="sk-or", claude_code_cli=fake_cli.path)
        seen = []

        async def fake_openrouter(**kwargs):
            seen.append(kwargs)
            body = json.dumps({"insights": "Visual", "clips": [{"start_time": 10, "end_time": 40, "summary": "Visible Result"}]})
            return ({"choices": [{"message": {"content": body}, "finish_reason": "stop"}]},
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01})

        monkeypatch.setattr(planner, "_call_openrouter", fake_openrouter)
        result = plan(planner, transcript=TranscriptionResult(segments=[], full_text=""), frames=self._frames(tmp_path))
        assert len(seen) == 1 and fake_cli.calls == []
        assert result.api_costs.provider == "openrouter"


class TestSettingsAndHelpers:
    def test_layout_vision_needs_openrouter_key(self):
        assert Settings(_env_file=None, openrouter_api_key="").layout_vision_enabled is False
        assert Settings(_env_file=None, openrouter_api_key="k").layout_vision_enabled is True

    def test_planner_backend_is_validated(self):
        assert Settings(_env_file=None).planner_backend == "openrouter"
        with pytest.raises(ValueError):
            Settings(_env_file=None, planner_backend="gpt")

    def test_desktop_bundle_is_found_newest_first(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        for version in ("2.1.9", "2.1.280", "2.1.30"):
            exe = tmp_path / ".config" / "Claude" / "claude-code" / version / "claude"
            exe.parent.mkdir(parents=True)
            exe.write_text("")
            exe.chmod(0o755)
        assert find_claude_cli().endswith("/2.1.280/claude")

    def test_schema_errors(self):
        assert schema_errors(valid_plan(), CLIP_PLAN_SCHEMA) == []
        errors = schema_errors({"clips": [{"start_time": True, "extra": 1}]}, CLIP_PLAN_SCHEMA)
        assert "$: missing 'insights'" in errors
        assert "$.clips[0]: unexpected 'extra'" in errors
        assert "$.clips[0].start_time: expected number" in errors

    def test_long_system_prompt_moves_to_stdin(self, fake_cli):
        fake_cli.responses.append(cli_body({"ok": True}))
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        value, _ = asyncio.run(run_claude_json("S" * 200_000, "U", schema, model="opus", effort="none",
                                               timeout_seconds=5, cli_path=fake_cli.path))
        args = fake_cli.calls[0]["args"]
        assert value == {"ok": True}
        assert len(args[args.index("--system-prompt") + 1]) < 100
        assert args[args.index("--effort") + 1] == "low"
        assert fake_cli.calls[0]["process"].stdin_text.startswith("SSS")

    def test_error_reason_is_a_fixed_code(self):
        assert ClaudeCodePlanningError("x", reason="made-up").reason == "failed"
        assert isinstance(ClaudeCodeError("x", "timeout"), Exception)
