# Setting up on CachyOS (desktop)

This is everything needed to run this fork on a fresh CachyOS (Arch-based) install. Take one of two routes:
**A. paste the prompt below into Claude Code** and let it do the work, or **B. run the commands yourself**.

What you need at hand: your GitHub login, your Claude (Max) login, and a Groq API key from
<https://console.groq.com/keys> (make a new one per computer if you like).

## A. Let Claude Code do it

1. Install Claude Code and log in:
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   claude
   ```
   In Claude, type `/login` and sign in with your Claude account. Stay in that session.
2. Paste this prompt:

   > Set up my BridgeClip fork on this CachyOS PC. Clone the private repo `justlinuxnoob/bridgeclipper`
   > into ~/Projects/bridgeclipper with `gh` (log me in with `gh auth login` first if needed), then follow
   > `docs/CACHYOS-SETUP.md` section B exactly: install the pacman packages, Python 3.12 and Node 22,
   > create engine/.venv, run the hash-locked pip install and `npm ci`, run `npm run typecheck` and
   > `npm test`, then start the app with `./scripts/dev-linux.sh`. Use sudo only for pacman/paru, and tell
   > me before you run anything else as root. Check Settings → System check is green. Don't install
   > GPU drivers or whisper.cpp unless I ask. When done, tell me to paste my Groq key in
   > Settings → API keys → Groq and pick Claude Code + Groq under Settings → AI engines.

## B. Manual steps

```bash
# 1. System packages. CachyOS's ffmpeg already has libx264 and libass (the `ass` filter).
sudo pacman -S --needed git github-cli base-devel ffmpeg nodejs-lts-jod npm

# 2. Python 3.12 (the engine's lockfile targets 3.12; CachyOS's default python is newer).
paru -S python312            # or: yay -S python312

# 3. Claude Code (skip if you did step A.1) and GitHub
curl -fsSL https://claude.ai/install.sh | bash
claude                       # type /login, sign in, then /exit
gh auth login

# 4. The app
mkdir -p ~/Projects && cd ~/Projects
gh repo clone justlinuxnoob/bridgeclipper
cd bridgeclipper
python3.12 -m venv engine/.venv
engine/.venv/bin/pip install --require-hashes -r engine/requirements.lock
npm ci

# 5. Checks, then run
npm run typecheck && npm test
./scripts/dev-linux.sh
```

Check that `node --version` prints v22. If `nodejs-lts-jod` isn't available, install Node 22 with `fnm` or `nvm` instead.

### In the app

1. **Settings → System check**: every row should be green.
2. **Settings → API keys → Groq**: paste your Groq key. It's stored in the system keychain, so a new PC needs it once.
   If saving says secure storage is unavailable, make sure a keyring such as `gnome-keyring` or KWallet is running.
3. **Settings → AI engines**: *Clip planner: Claude Code (subscription)* and *Transcription: Groq (free)*.
4. Optional: an OpenRouter key adds the paid fallback (Whisper past Groq's free limit, silent-video planning,
   layout vision).

Each clip in the Library then has a **Post captions** button with TikTok, YouTube Shorts and Instagram text.

## Optional: free local transcription on the RX 5700 (Vulkan)

ROCm doesn't support the RX 5700 (RDNA1), but whisper.cpp's Vulkan backend does.

```bash
sudo pacman -S --needed vulkan-radeon vulkan-headers shaderc spirv-headers cmake
git clone --depth 1 https://github.com/ggml-org/whisper.cpp ~/Projects/whisper.cpp
cd ~/Projects/whisper.cpp
cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON && cmake --build build-vulkan -j --target whisper-cli
bash models/download-ggml-model.sh large-v3-turbo-q5_0
```

Then pick *Transcription: Local (whisper.cpp)* in Settings. The app finds `build-vulkan` automatically.

## Notes

- Rendering uses the CPU (libx264), so a faster desktop CPU speeds it up directly. GPU encoding
  (VAAPI on the RX 5700) would need changes to `engine/clip_engine/services/rendering_service.py`.
- If YouTube downloads start failing, install a JavaScript runtime for yt-dlp: `sudo pacman -S deno`.
- `./scripts/dev-linux.sh` is `npm run dev` with inherited Electron variables cleared. Use it when starting from a
  terminal inside the Claude desktop app, where plain `npm run dev` shows a black window.
