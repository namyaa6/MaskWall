"""Create/update the Hugging Face Docker Space that serves the firewall API.

    python scripts/deploy_space.py --dry-run     # list what would be uploaded
    python scripts/deploy_space.py               # create the Space (if needed) and upload

Uploads ONLY what the Docker build needs: Dockerfile, requirements.lock.txt, app/ and the Space README (YAML header).
No tests, data, local model files or .env. Secrets are never handled here: set them in the Space settings
(Settings > Variables and secrets). Uses the login from `hf auth login`.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import shutil

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SPACE_NAME = "maskwall-api"


def stage(dest: Path) -> list[str]:
    shutil.copy2(BACKEND / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(BACKEND / "requirements.lock.txt", dest / "requirements.lock.txt")
    shutil.copy2(BACKEND / "space" / "README.md", dest / "README.md")
    shutil.copytree(BACKEND / "app", dest / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return sorted(str(p.relative_to(dest)).replace("\\", "/") for p in dest.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    repo = f"{api.whoami()['name']}/{SPACE_NAME}"
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        print(f"{repo}: {len(files)} files")
        for f in files:
            print("  ", f)
        if args.dry_run:
            return
        api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
        info = api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="space", commit_message="Deploy backend")
        print("uploaded:", info.commit_url)
        print(f"space page: https://huggingface.co/spaces/{repo}   (build logs: Logs button on that page)")


if __name__ == "__main__":
    main()
