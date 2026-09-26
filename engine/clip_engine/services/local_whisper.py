"""
Local transcription with whisper.cpp's `whisper-cli`.

Used when TRANSCRIPTION_BACKEND=local. Runs one audio chunk at a time with
one-word segments (`-ml 1 -sow`), so each JSON segment carries a whole word
with its timing, then returns an OpenAI-style verbose_json body that the
transcription service parses exactly like a provider response.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Non-speech markers whisper emits, e.g. [BLANK_AUDIO], (music), [Laughter].
_NON_SPEECH = re.compile(r"^\s*[\[(].*[\])]\s*$")
_WORD_CHARS = re.compile(r"\w", re.UNICODE)

# whisper.cpp's --prompt holds at most n_text_ctx/2 tokens; stay well inside.
_MAX_PROMPT_CHARS = 600


class LocalWhisperError(Exception):
    """`reason` is local_unavailable (not installed) or local_failed."""

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


def _default_root() -> str:
    return os.path.join(os.path.expanduser("~"), "Projects", "whisper.cpp")


def find_whisper_cli(configured: str = "") -> Optional[str]:
    """Configured path, then a ~/Projects/whisper.cpp build, then PATH."""
    if configured:
        path = os.path.expanduser(configured)
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    root = _default_root()
    # A GPU build wins when present; the CPU build is the portable default.
    for build in ("build-vulkan", "build-cuda", "build-cpu", "build"):
        candidate = os.path.join(root, build, "bin", "whisper-cli")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("whisper-cli")


def model_path(models_dir: str, model: str) -> str:
    return os.path.join(os.path.expanduser(models_dir) or os.path.join(_default_root(), "models"), f"ggml-{model}.bin")


def default_threads() -> int:
    """One thread per physical core: hyperthreads slow whisper.cpp down."""
    return max(1, (os.cpu_count() or 2) // 2)


def transcribe_chunk(
    audio_path: str,
    duration: float,
    *,
    language: Optional[str],
    prompt: Optional[str],
    cli_path: str,
    models_dir: str,
    model: str,
    threads: int,
) -> dict:
    """Transcribe one WAV file; returns {"text", "language", "words": [{word, start, end}]}."""
    cli = find_whisper_cli(cli_path)
    if not cli:
        raise LocalWhisperError("whisper.cpp whisper-cli not found", "local_unavailable")
    weights = model_path(models_dir, model)
    if not os.path.isfile(weights):
        raise LocalWhisperError(f"whisper.cpp model not found: ggml-{model}.bin", "local_unavailable")

    with tempfile.TemporaryDirectory(prefix="bridgeclip-whisper-") as work:
        output = os.path.join(work, "out")
        cmd = [
            cli, "-m", weights, "-f", audio_path,
            "-l", language or "auto",
            "-t", str(threads or default_threads()),
            "-ml", "1", "-sow",  # one word per segment, split on word boundaries
            "-oj", "-of", output, "-np",
        ]
        if prompt:
            cmd += ["--prompt", prompt[:_MAX_PROMPT_CHARS]]
        # Generous: a slow CPU runs large models at roughly 1.5x real time.
        timeout = max(600.0, duration * 5)
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise LocalWhisperError(f"whisper.cpp did not finish within {timeout:.0f}s", "local_failed") from None
        except OSError as e:
            raise LocalWhisperError(f"whisper.cpp could not start: {e}", "local_unavailable") from None
        if result.returncode != 0 or not os.path.isfile(output + ".json"):
            detail = result.stderr.decode("utf-8", "replace").strip()[-300:]
            logger.error("whisper.cpp exited %s: %s", result.returncode, detail)
            raise LocalWhisperError("whisper.cpp transcription failed", "local_failed")
        with open(output + ".json", encoding="utf-8", errors="replace") as f:
            body = json.load(f)
    return parse_whisper_json(body, duration)


def parse_whisper_json(body: dict, duration: float) -> dict:
    """Map whisper-cli `-ml 1 -sow -oj` output to verbose_json words (seconds)."""
    words: list[dict] = []
    for segment in body.get("transcription") or []:
        text = str(segment.get("text") or "").strip()
        offsets = segment.get("offsets") or {}
        start, end = offsets.get("from"), offsets.get("to")
        if not text or _NON_SPEECH.match(text) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        start_s = max(0.0, start / 1000)
        end_s = min(max(start_s, end / 1000), duration)
        start_s = min(start_s, end_s)
        if not _WORD_CHARS.search(text):
            # Stray punctuation split off a word: attach it to the previous word.
            if words:
                words[-1]["word"] += text
                words[-1]["end"] = max(words[-1]["end"], end_s)
            continue
        words.append({"word": text, "start": round(start_s, 3), "end": round(end_s, 3)})
    language = (body.get("result") or {}).get("language")
    return {
        "text": " ".join(w["word"] for w in words),
        "language": language if isinstance(language, str) else None,
        "words": words,
    }
