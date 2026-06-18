from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="google-bert/bert-base-multilingual-cased")
    parser.add_argument("--output", default="models/bert-base-multilingual-cased")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    snapshot_download(repo_id=args.repo, local_dir=output)
    print(f"Model downloaded to: {output}")
