#!/usr/bin/env bash
# Clones (or updates) and builds llama.cpp's llama-server with the ROCm/HIP
# backend. Do NOT use the Vulkan/RADV backend here — it has documented
# mmproj (vision encoder) correctness and crash bugs on AMD GPUs.
set -euo pipefail

REPO_DIR="${LLAMA_CPP_DIR:-$HOME/.local/src/llama.cpp}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull --ff-only

cmake -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j"$(nproc)"

echo "Built: $REPO_DIR/build/bin/llama-server"
