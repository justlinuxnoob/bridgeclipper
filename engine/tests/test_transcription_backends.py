"""
Offline tests for the Groq and local whisper.cpp transcription backends.
Groq is served by an httpx mock transport; whisper.cpp by a fake function.
"""

import asyncio
import json

import httpx
import pytest

from clip_engine.config import Settings
from clip_engine.error_policy import safe_failure_code, safe_job_error_text, safe_processing_error
from clip_engine.services import local_whisper
from clip_engine.services import transcription_service as stt
from clip_engine.services.transcription_service import TranscriptionError, TranscriptionService


def make_service(**overrides) -> TranscriptionService:
    service = TranscriptionService()
    service.settings = Settings(_env_file=None, **overrides)
    return service


def groq_body(words, language="English"):
    return {
        "text": " ".join(w[0] for w in words),
        "language": language,
        "words": [{"word": w, "start": s, "end": e} for w, s, e in words],
        "x_groq": {"id": "req_1"},
    }


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF" + b"\0" * 64)
    return str(path)


@pytest.fixture
def no_media(monkeypatch):
    """Durations come from the test; chunk extraction is a no-op."""
    durations = {}
    monkeypatch.setattr(TranscriptionService, "_audio_duration", staticmethod(lambda path: durations.get(path, durations.get("*", 60.0))))
    monkeypatch.setattr(TranscriptionService, "_extract_chunk", staticmethod(lambda *args: None))
    monkeypatch.setattr(stt.asyncio, "sleep", lambda _: _instant())
    return durations


async def _instant():
    return None


def mock_groq(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


class TestGroq:
    def test_sends_multipart_word_timestamps_and_costs_nothing(self, monkeypatch, wav, no_media):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=groq_body([("Labas", 0.2, 0.6), ("rytas.", 0.6, 1.1)], "Lithuanian"))

        mock_groq(monkeypatch, handler)
        service = make_service(transcription_backend="groq", groq_api_key="gsk_test", openrouter_api_key="")
        result = asyncio.run(service.transcribe_audio(wav, keyterms=["BridgeClip"]))

        request = seen[0]
        body = request.content.decode("latin-1")
        assert str(request.url) == stt.GROQ_TRANSCRIPTION_URL
        assert request.headers["authorization"] == "Bearer gsk_test"
        assert 'name="model"\r\n\r\nwhisper-large-v3-turbo' in body
        assert 'name="response_format"\r\n\r\nverbose_json' in body
        assert body.count('name="timestamp_granularities[]"') == 2
        assert "Expected vocabulary: BridgeClip" in body
        assert 'name="language"' not in body  # auto-detect
        assert [w.word for s in result.segments for w in s.words] == ["Labas", "rytas."]
        assert result.segments[0].words[0].start_time_ms == 200
        assert (result.provider, result.language) == ("groq", "Lithuanian")
        assert result.api_costs.estimated_cost_usd == 0.0 and not result.api_costs.cost_incomplete
        assert result.api_costs.model == "groq/whisper-large-v3-turbo"

    def test_missing_or_rejected_key(self, monkeypatch, wav, no_media):
        with pytest.raises(TranscriptionError) as missing:
            asyncio.run(make_service(transcription_backend="groq").transcribe_audio(wav))
        assert missing.value.reason == "groq_auth"

        mock_groq(monkeypatch, lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}))
        with pytest.raises(TranscriptionError) as rejected:
            asyncio.run(make_service(transcription_backend="groq", groq_api_key="gsk_bad").transcribe_audio(wav))
        assert rejected.value.reason == "groq_auth"
        assert safe_processing_error(rejected.value) == "Groq rejected the transcription key"
        assert safe_failure_code(rejected.value) == "transcription.groq_auth"

    def test_free_limit_falls_back_to_openrouter_whisper(self, monkeypatch, wav, no_media):
        models = []

        async def fake_request(self, path, language, keyterms, model=None):
            models.append(model)
            if model.startswith("groq/"):
                raise stt.TranscriptionProviderError("rate_limit", 429, 3600)
            return {"text": "Hi.", "words": [{"word": "Hi.", "start": 0.1, "end": 0.4}], "usage": {"cost": 0.0002}}

        monkeypatch.setattr(TranscriptionService, "_request_transcript", fake_request)
        service = make_service(transcription_backend="groq", groq_api_key="gsk", openrouter_api_key="sk-or")
        result = asyncio.run(service.transcribe_audio(wav))
        # Retry-After is an hour, so no retry on Groq: straight to OpenRouter Whisper Turbo.
        assert models == ["groq/whisper-large-v3-turbo", stt.BUDGET_TRANSCRIPTION_MODEL]
        assert result.api_costs.model == stt.BUDGET_TRANSCRIPTION_MODEL
        assert result.api_costs.estimated_cost_usd == 0.0002

    def test_free_limit_without_openrouter_key_fails_clearly(self, monkeypatch, wav, no_media):
        async def fake_request(self, path, language, keyterms, model=None):
            raise stt.TranscriptionProviderError("rate_limit", 429, 3600)

        monkeypatch.setattr(TranscriptionService, "_request_transcript", fake_request)
        with pytest.raises(TranscriptionError) as failure:
            asyncio.run(make_service(transcription_backend="groq", groq_api_key="gsk").transcribe_audio(wav))
        assert failure.value.reason == "groq_limit"
        message = safe_processing_error(failure.value)
        assert message == "Groq free transcription limit reached" and safe_job_error_text(message) == message

    def test_short_rate_limit_is_retried_on_groq(self, monkeypatch, wav, no_media):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "2"}, json={})
            return httpx.Response(200, json=groq_body([("Hello.", 0.0, 0.5)]))

        mock_groq(monkeypatch, handler)
        result = asyncio.run(make_service(transcription_backend="groq", groq_api_key="gsk").transcribe_audio(wav))
        assert len(calls) == 2 and result.segments[0].text == "Hello."


class TestLocalWhisper:
    def test_chunks_keep_source_timeline_and_pin_detected_language(self, monkeypatch, wav, no_media):
        # 650 s of audio -> chunks [0,301], [299,601], [599,650] with 1 s overlap.
        no_media[wav] = 650.0
        no_media["*"] = 302.0
        calls = []

        def fake_chunk(path, duration, *, language, prompt, cli_path, models_dir, model, threads):
            calls.append({"language": language, "model": model, "prompt": prompt})
            index = len(calls) - 1
            # Chunk-relative times: one word at 0.5 s (overlap, owned by the previous
            # chunk except for the first) and one at 10 s.
            words = [{"word": f"early{index}", "start": 0.3, "end": 0.7},
                     {"word": f"word{index}.", "start": 10.0, "end": 10.5}]
            return {"text": "", "language": "lt", "words": words}

        monkeypatch.setattr(local_whisper, "transcribe_chunk", fake_chunk)
        service = make_service(transcription_backend="local", local_whisper_model="small")
        result = asyncio.run(service.transcribe_audio(wav, keyterms=["Vilnius"], timeline_offset_seconds=100.0))

        assert [c["model"] for c in calls] == ["small"] * 3
        assert [c["language"] for c in calls] == [None, "lt", "lt"]
        assert calls[0]["prompt"] == "Expected vocabulary: Vilnius"
        words = [(w.word, w.start_time_ms) for s in result.segments for w in s.words]
        # Chunk starts are 0, 299 and 599 s, plus the 100 s timeline offset. Words at
        # 0.3-0.7 s in chunks 2 and 3 fall inside the previous chunk's core and are dropped.
        assert words == [
            ("early0", 100_300), ("word0.", 110_000),
            ("word1.", 409_000),
            ("word2.", 709_000),
        ]
        assert (result.provider, result.language) == ("local", "lt")
        assert result.api_costs.estimated_cost_usd == 0.0 and result.api_costs.model == "local/small"

    def test_missing_install_is_a_clear_error(self, wav, no_media, tmp_path):
        service = make_service(transcription_backend="local", local_whisper_cli=str(tmp_path / "missing"))
        with pytest.raises(TranscriptionError) as failure:
            asyncio.run(service.transcribe_audio(wav))
        assert failure.value.reason == "local_unavailable"
        assert safe_processing_error(failure.value) == "Local whisper.cpp is not set up"

    def test_missing_model_is_a_clear_error(self, wav, no_media, tmp_path):
        cli = tmp_path / "whisper-cli"
        cli.write_text("#!/bin/sh\n")
        cli.chmod(0o755)
        service = make_service(transcription_backend="local", local_whisper_cli=str(cli),
                               local_whisper_models_dir=str(tmp_path))
        with pytest.raises(TranscriptionError) as failure:
            asyncio.run(service.transcribe_audio(wav))
        assert failure.value.reason == "local_unavailable"

    def test_parse_whisper_json(self):
        body = {"result": {"language": "en"}, "transcription": [
            {"offsets": {"from": 0, "to": 400}, "text": " Hello"},
            {"offsets": {"from": 400, "to": 420}, "text": ","},
            {"offsets": {"from": 500, "to": 900}, "text": " world."},
            {"offsets": {"from": 1000, "to": 3000}, "text": " [BLANK_AUDIO]"},
            {"offsets": {"from": 3000, "to": 99000}, "text": " Ačiū"},
        ]}
        parsed = local_whisper.parse_whisper_json(body, duration=5.0)
        assert parsed["language"] == "en"
        assert parsed["words"] == [
            {"word": "Hello,", "start": 0.0, "end": 0.42},
            {"word": "world.", "start": 0.5, "end": 0.9},
            {"word": "Ačiū", "start": 3.0, "end": 5.0},  # clamped to the chunk
        ]

    def test_real_cli_command_line(self, monkeypatch, tmp_path):
        cli = tmp_path / "whisper-cli"
        cli.write_text("#!/bin/sh\n")
        cli.chmod(0o755)
        (tmp_path / "ggml-small.bin").write_bytes(b"x")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            output = cmd[cmd.index("-of") + 1]
            with open(output + ".json", "w") as f:
                json.dump({"result": {"language": "en"}, "transcription": [{"offsets": {"from": 0, "to": 300}, "text": " Hi."}]}, f)
            return local_whisper.subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(local_whisper.subprocess, "run", fake_run)
        parsed = local_whisper.transcribe_chunk(
            "a.wav", 1.0, language=None, prompt=None, cli_path=str(cli),
            models_dir=str(tmp_path), model="small", threads=0,
        )
        cmd = seen["cmd"]
        assert cmd[cmd.index("-m") + 1] == str(tmp_path / "ggml-small.bin")
        assert cmd[cmd.index("-l") + 1] == "auto"
        assert {"-ml", "-sow", "-oj", "-np"} <= set(cmd)
        assert parsed["words"] == [{"word": "Hi.", "start": 0.0, "end": 0.3}]


def test_openrouter_backend_still_needs_openrouter_key(wav, no_media):
    with pytest.raises(TranscriptionError) as failure:
        asyncio.run(make_service(transcription_backend="openrouter", openrouter_api_key="").transcribe_audio(wav))
    assert failure.value.reason == "auth"
