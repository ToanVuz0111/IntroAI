from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal", default="artifacts/full_smoke/evaluation.csv")
    parser.add_argument("--random", default="artifacts/paper_random/evaluation.csv")
    parser.add_argument("--output", default="artifacts/split_comparison.csv")
    args = parser.parse_args()
    temporal = pd.read_csv(args.temporal).assign(split_strategy="temporal")
    random = pd.read_csv(args.random).assign(split_strategy="paper_random")
    output = pd.concat([temporal, random], ignore_index=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(output.to_string(index=False))
