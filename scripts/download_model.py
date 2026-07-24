"""Download a registered Qwen2.5 snapshot into ./models_cache.

Usage:
    python scripts/download_model.py                       # default 0.5B
    python scripts/download_model.py Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from shrike.config import MODEL_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        choices=sorted(MODEL_REGISTRY),
        help="HF repo id of a registered Qwen2.5 model",
    )
    args = parser.parse_args()

    local_dir = f"models_cache/{args.repo_id.split('/')[-1].lower()}"
    path = snapshot_download(
        args.repo_id,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"],
    )
    print("Downloaded to", path)


if __name__ == "__main__":
    main()
