#!/usr/bin/env bash
set -euo pipefail

build_dir=$(mktemp -d -t editalive-kernel.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST=$(python - <<'PYTHON'
import torch

major, minor = torch.cuda.get_device_capability()
print("9.0a" if (major, minor) == (9, 0) else f"{major}.{minor}")
PYTHON
)
  export TORCH_CUDA_ARCH_LIST
fi

python -m pip install scikit-build-core cmake ninja wheel
git init -q "$build_dir"
git -C "$build_dir" remote add origin https://github.com/hao-ai-lab/FastVideo.git
git -C "$build_dir" fetch --depth 1 origin b1d89eba1f177f2b096192c9b1e7ceb7096bea99
git -C "$build_dir" checkout -q --detach FETCH_HEAD
git -C "$build_dir" submodule update --init --recursive --depth 1 \
  fastvideo-kernel/include/tk fastvideo-kernel/include/cutlass

# PyTorch 2.6 needs evaluated annotations for custom operators.
python - "$build_dir/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
assert text.count("from __future__ import annotations\n") == 1
path.write_text(text.replace("from __future__ import annotations\n", ""))
PY

CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
  python -m pip install --no-build-isolation "$build_dir/fastvideo-kernel"
