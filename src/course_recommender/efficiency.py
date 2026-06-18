from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from .config import project_path
from .recommendation import load_bundle, recommend
from .utils import dump_json


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def profile_efficiency(config: dict, sessions: int = 100) -> dict:
    artifacts = project_path(config, config["artifacts_dir"])
    bundle = load_bundle(config)
    users = bundle["user_ids"][: min(sessions, len(bundle["user_ids"]))]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    recommend(config, users[0], 10, translate_output=False)
    latencies = []
    for user_id in users:
        started = time.perf_counter()
        recommend(config, user_id, 10, translate_output=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)

    bert = _read_json(artifacts / "bert_finetuning_summary.json")
    pairwise = _read_json(artifacts / "training_summary.json")
    ppo = _read_json(artifacts / "ppo_training_summary.json")
    actor_parameters = (
        sum(item.numel() for item in bundle["catalog_actor"].parameters())
        if bundle["catalog_actor"] is not None
        else 0
    )
    critic_parameters = int(pairwise.get("critic_parameters", 0))
    parameter_rows = [
        {
            "component": "BERT",
            "total_parameters": int(bert.get("total_parameters", 0)),
            "trainable_parameters": int(bert.get("trainable_parameters", 0)),
        },
        {
            "component": "Catalog Actor",
            "total_parameters": actor_parameters,
            "trainable_parameters": actor_parameters,
        },
        {
            "component": "Critic",
            "total_parameters": critic_parameters,
            "trainable_parameters": critic_parameters,
        },
    ]
    pd.DataFrame(parameter_rows).to_csv(artifacts / "model_parameters.csv", index=False)
    pd.DataFrame({"latency_seconds": latencies}).to_csv(
        artifacts / "inference_latency.csv",
        index=False,
    )
    summary = {
        "sessions": len(latencies),
        "mean_inference_seconds": float(np.mean(latencies)),
        "p50_inference_seconds": float(np.percentile(latencies, 50)),
        "p95_inference_seconds": float(np.percentile(latencies, 95)),
        "peak_inference_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        ),
        "bert_training_seconds": bert.get("training_seconds"),
        "pairwise_training_seconds": pairwise.get("training_seconds"),
        "ppo_training_seconds": ppo.get("training_seconds"),
        "total_parameters": sum(row["total_parameters"] for row in parameter_rows),
        "trainable_parameters": sum(row["trainable_parameters"] for row in parameter_rows),
    }
    dump_json(artifacts / "efficiency_summary.json", summary)
    return summary
