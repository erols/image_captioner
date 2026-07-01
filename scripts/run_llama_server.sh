#!/usr/bin/env bash
# Launches llama-server against a given GGUF model (+ optional mmproj for
# vision). Sets the gfx1151 ROCm "Preview" override required on Strix Halo.
set -euo pipefail

MODEL_PATH="${1:?usage: run_llama_server.sh MODEL_GGUF [MMPROJ_GGUF]}"
MMPROJ_PATH="${2:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.local/src/llama.cpp}"
PORT="${LLAMA_SERVER_PORT:-8080}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"

ARGS=(-m "$MODEL_PATH" --port "$PORT" -ngl 999)
if [ -n "$MMPROJ_PATH" ]; then
  ARGS+=(--mmproj "$MMPROJ_PATH")
fi

exec "$LLAMA_CPP_DIR/build/bin/llama-server" "${ARGS[@]}"
