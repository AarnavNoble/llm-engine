#!/usr/bin/env bash
# Download an open-weights model into models/<name>. Requires `pip install huggingface_hub`.
set -euo pipefail
REPO="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
NAME="$(basename "$REPO")"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models/$NAME"
mkdir -p "$DEST"
python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer.model"])
print("downloaded", repo, "->", dest)
PY
