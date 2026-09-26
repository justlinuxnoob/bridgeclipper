# Local Claude fork (`local-claude` branch)

This branch lets BridgeClip run with no per-token AI billing:

| Step | Backend | Cost |
| --- | --- | --- |
| Clip planning | Claude Code CLI (`claude -p`) on your Claude subscription | $0 |
| Transcription | Groq free tier, Whisper Large V3 Turbo (~8 h of audio a day) | $0 |
| Transcription fallback | OpenRouter Whisper Turbo, only if an OpenRouter key is saved | ~$0.01 per audio hour |
| Post captions (TikTok, YouTube Shorts, Instagram) | Claude Code, written while clips render | $0 |
| Framing | Local face tracking (layout vision needs an OpenRouter key) | $0 |

Choose backends in **Settings → AI engines**. Keys are only required for the backends you pick.

Each clip gets ready-to-post text per platform: in the Library (**Post captions** button with copy buttons),
in `job_output.json`, and as `clip_NN.captions.txt` next to the video.

New machine? See [CACHYOS-SETUP.md](CACHYOS-SETUP.md).

## Setup (Linux)

1. Install Claude Code and log in once: run `claude`, type `/login`, then `/exit`.
2. Create a Groq key at <https://console.groq.com/keys> and paste it into **Settings → API keys → Groq**.
3. Set **Settings → AI engines** to *Clip planner: Claude Code (subscription)* and *Transcription: Groq (free)*.

## Run

```bash
cd ~/Projects/bridgeclip
./scripts/dev-linux.sh
```

`scripts/dev-linux.sh` is `npm run dev` with inherited Electron variables cleared. Launched from a terminal
inside another Electron app (such as the Claude desktop app), `ELECTRON_FORCE_IS_PACKAGED` otherwise leaves
a black window.

## How Claude Code planning works

`engine/clip_engine/services/claude_code.py` runs, from an empty temporary directory:

```
claude -p --output-format json --model opus --effort <planner effort> --tools "" --setting-sources "" \
  --strict-mcp-config --disable-slash-commands --no-session-persistence --json-schema <clip plan schema> \
  --system-prompt <planner system prompt>      # transcript and instructions arrive on stdin
```

Only HOME, PATH and locale/proxy variables are passed, never API keys: an `ANTHROPIC_API_KEY` would switch
the CLI from your subscription to API billing. The JSON is schema-checked and retried once if invalid.
`--bare` is not used because it ignores the subscription login. Planning is text-only; silent videos use
OpenRouter vision planning when an OpenRouter key is saved.

## Local whisper.cpp (optional)

*Transcription: Local (whisper.cpp)* runs `whisper-cli` from `~/Projects/whisper.cpp`
(`build-vulkan`, `build-cuda`, `build-cpu` or `build`, first found) with models in
`~/Projects/whisper.cpp/models`. On a ThinkPad T480 (i5-8350U, CPU) Large V3 Turbo q5_0 ran at about
**89 minutes per hour of audio**, slower than real time, so it's best on a desktop GPU. For an AMD card, build
with `-DGGML_VULKAN=ON` (needs `vulkan-headers`, `glslc` and `spirv-headers-devel` on Fedora).
