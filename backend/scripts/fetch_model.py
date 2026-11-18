"""Download the served int8 MiniLM (backend layout) from the Hugging Face Hub, pinned to a commit.

    python backend/scripts/fetch_model.py backend/model/minilm      # CI and the Dockerfile both call this

The repo is public: no token is needed or used. REVISION is the single place that says which model version the deployed API and
CI run; to ship a new model, upload it, then change REVISION here (the golden-score tests in CI will tell you if the scanner and
the model disagree). Needs only `huggingface_hub`.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "namya06/maskwall-injection-minilm"
REVISION = "78085d047839aef62e6bc4bad01dad4b4e9cf370"
FILES = ["model.int8.onnx", "tokenizer.json", "model_meta.json"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fetch_model.py <destination folder>")
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO, revision=REVISION, local_dir=str(dest), allow_patterns=FILES)
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    missing = [f for f in FILES if not (dest / f).exists()]
    if missing:
        raise SystemExit(f"download incomplete, missing: {missing}")
    for f in FILES:
        print(f"{f}: {(dest / f).stat().st_size:,} bytes")
    print(f"model {REPO}@{REVISION[:10]} -> {dest}")


if __name__ == "__main__":
    main()
