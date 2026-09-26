#!/usr/bin/env bash
# Run the dev app from a shell that may have inherited Electron variables from a
# parent Electron app (for example a terminal opened inside the Claude desktop app).
# ELECTRON_FORCE_IS_PACKAGED makes the dev app load out/renderer instead of the
# Vite dev server, which shows an empty black window.
set -euo pipefail
cd "$(dirname "$0")/.."
exec env -u ELECTRON_FORCE_IS_PACKAGED -u ELECTRON_RUN_AS_NODE -u ELECTRON_PA_APP_NAME \
  npm run dev -- "$@"
