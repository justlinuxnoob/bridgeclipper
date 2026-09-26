"""
Run the local Claude Code CLI (`claude -p`) as a one-shot, tool-less JSON model.

Used by the clip planner when PLANNER_BACKEND=claude_code, so planning runs on
the user's Claude subscription instead of per-token OpenRouter billing. Each
call starts a fresh session in an empty temporary directory with no tools, no
settings files and no MCP servers, so nothing from the user's projects leaks in.
"""

import asyncio
import glob
import json
import logging
import os
import re
import shutil
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Passed through so the CLI finds its login (~/.claude) and can reach the API
# on the user's connection. API keys are deliberately not passed: an
# ANTHROPIC_API_KEY would switch the CLI from the subscription to API billing.
_PASSTHROUGH_ENV = (
    "HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR",
    "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
    "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
)

# The CLI accepts these for --effort; planner efforts below "low" map to "low".
_CLI_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Linux caps a single argv string at 128 KiB; longer system prompts go on stdin.
_MAX_ARG_BYTES = 100_000


class ClaudeCodeError(Exception):
    """A failed Claude Code call. `reason` is a fixed code, safe to show."""

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


def find_claude_cli(configured: str = "") -> Optional[str]:
    """Resolve the `claude` executable: configured path, PATH, then install locations."""
    if configured:
        path = os.path.expanduser(configured)
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    found = shutil.which("claude")
    if found:
        return found
    home = os.path.expanduser("~")
    for candidate in (
        os.path.join(home, ".local", "bin", "claude"),
        os.path.join(home, ".claude", "local", "claude"),
        os.path.join(home, ".npm-global", "bin", "claude"),
        "/usr/local/bin/claude",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    # The copy bundled with the Claude desktop app, newest version first.
    bundled = glob.glob(os.path.join(home, ".config", "Claude", "claude-code", "*", "claude"))
    bundled.sort(key=lambda p: _version_key(os.path.basename(os.path.dirname(p))), reverse=True)
    return next((p for p in bundled if os.access(p, os.X_OK)), None)


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _cli_env(cli_path: str) -> dict[str, str]:
    env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if os.environ.get(key)}
    env.setdefault("HOME", os.path.expanduser("~"))
    cli_dir = os.path.dirname(cli_path)
    env["PATH"] = os.pathsep.join(filter(None, [cli_dir, env.get("PATH", "/usr/bin:/bin")]))
    env["DISABLE_AUTOUPDATER"] = "1"
    return env


def cli_effort(planner_effort: str) -> str:
    return planner_effort if planner_effort in _CLI_EFFORTS else "low"


def _classify_failure(text: str, status: Any) -> str:
    lowered = text.lower()
    if status in (401, 403) or any(marker in lowered for marker in (
        "not logged in", "/login", "invalid api key", "authentication", "oauth token",
    )):
        return "not_logged_in"
    if status == 429 or re.search(r"usage limit|rate limit|limit reached|hit your .*limit|out of (extra )?usage", lowered):
        return "usage_limit"
    return "failed"


def extract_json(text: str) -> Any:
    """Parse a JSON value from model text, tolerating code fences or stray prose."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if fenced:
        stripped = fenced.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise
        return json.loads(match.group())


_JSON_TYPES: dict[str, Any] = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "number": (int, float), "integer": int,
}


def schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate against the JSON Schema subset the planner schemas use."""
    expected = schema.get("type")
    if expected:
        python_type = _JSON_TYPES[expected]
        is_bool = isinstance(value, bool)
        if not isinstance(value, python_type) or (is_bool and expected in ("number", "integer")):
            return [f"{path}: expected {expected}"]
    errors: list[str] = []
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        errors += [f"{path}: missing '{key}'" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected '{key}'" for key in value if key not in properties]
        for key, sub in properties.items():
            if key in value:
                errors += schema_errors(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errors


async def run_claude_json(
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    *,
    model: str,
    effort: str,
    timeout_seconds: float,
    cli_path: str = "",
) -> tuple[Any, dict[str, Any]]:
    """Ask Claude Code for one JSON value matching `schema`.

    Returns (value, usage) where usage has input_tokens, output_tokens and the
    model that answered. Raises ClaudeCodeError with reason cli_missing,
    not_logged_in, usage_limit, timeout, invalid_output or failed.
    """
    cli = find_claude_cli(cli_path)
    if not cli:
        raise ClaudeCodeError("Claude Code CLI not found", "cli_missing")

    args = [
        "-p",
        "--output-format", "json",
        "--model", model,
        "--effort", cli_effort(effort),
        # No built-in tools, settings files, MCP servers, skills or saved session.
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--json-schema", json.dumps(schema, separators=(",", ":")),
    ]
    stdin_text = user_prompt
    if len(system_prompt.encode()) <= _MAX_ARG_BYTES:
        args += ["--system-prompt", system_prompt]
    else:
        args += ["--system-prompt", "Follow the instructions at the start of the user message."]
        stdin_text = f"{system_prompt}\n\n---\n\n{user_prompt}"

    with tempfile.TemporaryDirectory(prefix="bridgeclip-claude-") as workdir:
        process = await asyncio.create_subprocess_exec(
            cli, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=_cli_env(cli),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_text.encode()), timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise ClaudeCodeError(f"Claude Code did not answer within {timeout_seconds:.0f}s", "timeout") from None
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    try:
        body = json.loads(stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        detail = stderr.decode("utf-8", "replace").strip()[-300:]
        logger.error("Claude Code exited %s without a JSON result: %s", process.returncode, detail)
        raise ClaudeCodeError("Claude Code returned no result", _classify_failure(detail, None)) from None

    if body.get("is_error") or body.get("subtype") not in (None, "success"):
        detail = str(body.get("result") or body.get("subtype") or "")[:300]
        reason = _classify_failure(detail, body.get("api_error_status"))
        logger.error("Claude Code call failed (%s): %s", reason, detail)
        raise ClaudeCodeError(detail or "Claude Code call failed", reason)

    value = body.get("structured_output")
    if value is None:
        try:
            value = extract_json(str(body.get("result") or ""))
        except json.JSONDecodeError:
            raise ClaudeCodeError("Claude Code did not return JSON", "invalid_output") from None
    errors = schema_errors(value, schema)
    if errors:
        logger.warning("Claude Code JSON failed schema validation: %s", "; ".join(errors[:5]))
        raise ClaudeCodeError(f"Claude Code JSON did not match the schema ({errors[0]})", "invalid_output")

    usage = body.get("usage") or {}
    return value, {
        "input_tokens": int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "model": next(iter(body.get("modelUsage") or {}), None),
    }
