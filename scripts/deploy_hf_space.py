"""Deploy this repo to a Hugging Face Space (Docker SDK).

    python scripts/deploy_hf_space.py --token hf_xxx --space ArminShirzad/tutor-rag

What it does:
  1. creates the Space (docker SDK) if it does not exist
  2. uploads the source, prepending the YAML frontmatter that Spaces requires
     to README.md -- the GitHub README stays untouched
  3. sets GEMINI_API_KEY as a *secret* (never committed, never in the image)

The Space then builds the Dockerfile, which bakes the models and the index at
build time so the container starts ready to serve.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FRONTMATTER = """---
title: tutor-rag
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Grounded RAG over course material with hybrid retrieval and reranking
---

"""

INCLUDE = ["app", "ui", "data/corpus", "scripts", "tests"]
INCLUDE_FILES = ["Dockerfile", "requirements.txt", "requirements-dev.txt",
                 "LICENSE", ".env.example"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token")
    p.add_argument("--space", required=True, help="e.g. username/tutor-rag")
    p.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY"))
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if not args.token:
        print("Need a write token: --token hf_xxx  (https://huggingface.co/settings/tokens)")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    who = api.whoami()
    print(f"authenticated as {who['name']}")

    # 1. create the Space
    try:
        api.create_repo(repo_id=args.space, repo_type="space", space_sdk="docker",
                        private=args.private, exist_ok=True)
        print(f"space ready: https://huggingface.co/spaces/{args.space}")
    except Exception as exc:
        print(f"could not create space: {exc}")
        return 1

    # 2. assemble the upload folder
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "space"
        staging.mkdir()

        import shutil

        for d in INCLUDE:
            src = ROOT / d
            if src.exists():
                shutil.copytree(src, staging / d,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in INCLUDE_FILES:
            if (ROOT / f).exists():
                shutil.copy2(ROOT / f, staging / f)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        (staging / "README.md").write_text(FRONTMATTER + readme, encoding="utf-8")

        print("uploading ...")
        api.upload_folder(folder_path=str(staging), repo_id=args.space,
                          repo_type="space", commit_message="deploy tutor-rag")

    # 3. secrets -- set as a Space secret, never baked into the image
    if args.gemini_key:
        api.add_space_secret(repo_id=args.space, key="GEMINI_API_KEY", value=args.gemini_key)
        print("set GEMINI_API_KEY as a space secret")
    else:
        print("no Gemini key given -- the Space will run in extractive mode")

    print(f"\nbuilding: https://huggingface.co/spaces/{args.space}")
    print("first build takes ~8-12 minutes (torch + model weights).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
