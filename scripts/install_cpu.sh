#!/usr/bin/env bash
# Install the locked environment with the CPU-only build of PyTorch (no CUDA libraries).
# PyPI's Linux torch wheels pull in about 3 GB of NVIDIA packages that CI and laptops
# without a GPU never use. Afterwards run commands with `uv run --no-sync`.
set -euo pipefail

skip=$(python3 - <<'PY'
import tomllib
lock = tomllib.load(open("uv.lock", "rb"))
names = [p["name"] for p in lock["package"]]
gpu = [n for n in names if n == "torch" or n.startswith(("nvidia-", "cuda-")) or n == "triton"]
print(" ".join(f"--no-install-package {n}" for n in gpu))
PY
)
torch_version=$(python3 -c 'import tomllib; print(next(p["version"] for p in tomllib.load(open("uv.lock","rb"))["package"] if p["name"]=="torch"))')

# shellcheck disable=SC2086
uv sync --frozen --all-extras $skip
uv pip install "torch==${torch_version}" --index-url https://download.pytorch.org/whl/cpu
