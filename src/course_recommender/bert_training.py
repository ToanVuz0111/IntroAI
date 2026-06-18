from __future__ import annotations

import random
import time

import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import project_path
from .utils import dump_json, ensure_dir, set_seed


class CourseTripletDataset(Dataset):
    def __init__(self, triplets: list[tuple[int, int, int]]) -> None:
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        return self.triplets[index]


def build_triplets(train: pd.DataFrame, course_index: dict[str, int], seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    all_indices = list(course_index.values())
    triplets = []
    for _, group in train.sort_values("timestamp").groupby("user_id"):
        positives = [course_index[str(value)] for value in group["course_id"] if str(value) in course_index]
        positives = list(dict.fromkeys(positives))
        if len(positives) < 2:
            continue
        positive_set = set(positives)
        for anchor, positive in zip(positives, positives[1:]):
            candidates = [value for value in all_indices if value not in positive_set]
            if candidates:
                triplets.append((anchor, positive, rng.choice(candidates)))
    return triplets


def fine_tune_bert(config: dict) -> dict:
    set_seed(config.get("seed", 42))
    artifacts = project_path(config, config["artifacts_dir"])
    model_cfg = config["embedding"]
    tune_cfg = config["bert_finetuning"]
    source = project_path(config, model_cfg["local_model_dir"])
    output = ensure_dir(project_path(config, model_cfg["checkpoint_dir"]))
    courses = pd.read_csv(artifacts / "courses.csv", dtype={"course_id": str})
    train = pd.read_csv(artifacts / "train.csv", dtype={"user_id": str, "course_id": str})
    texts = courses["text"].fillna("").astype(str).tolist()
    course_index = {value: index for index, value in enumerate(courses["course_id"].astype(str))}
    triplets = build_triplets(train, course_index, config.get("seed", 42))
    max_triplets = tune_cfg.get("max_triplets")
    if max_triplets:
        triplets = triplets[: int(max_triplets)]
    if not triplets:
        raise RuntimeError("BERT fine-tuning needs users with at least two training courses.")

    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, use_fast=False,
    )
    model = AutoModel.from_pretrained(source, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.layer[-int(tune_cfg.get("unfreeze_last_layers", 4)):]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if getattr(model, "pooler", None) is not None:
        for parameter in model.pooler.parameters():
            parameter.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        CourseTripletDataset(triplets),
        batch_size=int(tune_cfg.get("batch_size", 8)),
        shuffle=True,
    )
    parameters = [item for item in model.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(tune_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(tune_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(tune_cfg.get("epochs", 3))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(len(loader) * epochs * 0.1), 1),
        num_training_steps=max(len(loader) * epochs, 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def encode(indices: torch.Tensor) -> torch.Tensor:
        batch_text = [texts[int(index)] for index in indices]
        tokens = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=int(model_cfg.get("max_length", 128)),
            return_tensors="pt",
        ).to(device)
        return model(**tokens).last_hidden_state[:, 0]

    losses = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                anchor_vector = encode(anchor)
                positive_vector = encode(positive)
                negative_vector = encode(negative)
                positive_similarity = functional.cosine_similarity(anchor_vector, positive_vector)
                negative_similarity = functional.cosine_similarity(anchor_vector, negative_vector)
                loss = functional.relu(
                    float(tune_cfg.get("margin", 0.2)) - positive_similarity + negative_similarity
                ).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach())
        losses.append(total / max(len(loader), 1))

    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    summary = {
        "source_model": str(source),
        "checkpoint": str(output),
        "triplets": len(triplets),
        "epochs": epochs,
        "losses": losses,
        "total_parameters": sum(item.numel() for item in model.parameters()),
        "trainable_parameters": sum(item.numel() for item in model.parameters() if item.requires_grad),
        "device": str(device),
        "training_seconds": time.perf_counter() - started,
        "seconds_per_epoch": (time.perf_counter() - started) / max(epochs, 1),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    dump_json(artifacts / "bert_finetuning_summary.json", summary)
    return summary
