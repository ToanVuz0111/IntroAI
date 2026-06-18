from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import project_path
from .utils import (
    anonymize_course_id,
    anonymize_user_id,
    clean_text,
    dump_json,
    ensure_dir,
    read_jsonl,
    read_pairs,
)


COURSE_COLUMNS = [
    "course_id", "source_course_id", "title", "description", "tags", "category", "difficulty",
    "source_dataset", "total_videos", "text",
]
INTERACTION_COLUMNS = [
    "interaction_id", "user_id", "course_id", "timestamp", "rating", "clicked",
    "video_views", "completion_rate", "quiz_score", "engagement_time",
    "access_frequency", "session_duration", "device_type", "interaction_type",
    "source_dataset", "is_derived", "reward",
]


def _course_video_counts(root: Path) -> Counter:
    return Counter(course_id for course_id, _ in read_pairs(root / "relations/course-video.json"))


def _course_tags(root: Path, limit: int = 30) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for course_id, concept_id in read_pairs(root / "relations/course-concept.json"):
        if len(values[course_id]) < limit:
            values[course_id].append(clean_text(concept_id.replace("K_", "").replace("_", " "), 100))
    return values


def normalize_difficulty(value: object) -> tuple[str, float | None]:
    text = clean_text(value, 100).lower()
    mapping = {
        "beginner": 0.0,
        "introductory": 0.0,
        "intermediate": 0.5,
        "advanced": 1.0,
    }
    for label, score in mapping.items():
        if label in text:
            return label, score
    return "unknown", None


def export_courses(root: Path, output: Path, course_salt: str) -> int:
    counts = _course_video_counts(root)
    tags = _course_tags(root)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COURSE_COLUMNS)
        writer.writeheader()
        for item in read_jsonl(root / "entities/course.json"):
            source_course_id = str(item["id"])
            course_id = anonymize_course_id(source_course_id, course_salt)
            title = clean_text(item.get("name"), 300)
            description = clean_text(item.get("about"), 2500)
            tag_text = ", ".join(tags.get(source_course_id, []))
            difficulty, _ = normalize_difficulty(item.get("difficulty"))
            writer.writerow({
                "course_id": course_id,
                "source_course_id": source_course_id,
                "title": title,
                "description": description,
                "tags": tag_text,
                "category": tag_text.split(",")[0] if tag_text else "unknown",
                "difficulty": difficulty,
                "source_dataset": "MOOCCube",
                "total_videos": counts.get(source_course_id, 0),
                "text": clean_text(" ".join([title, description, tag_text]), 4000),
            })
            count += 1
    return count


def export_users_and_interactions(
    root: Path,
    users_output: Path,
    interactions_output: Path,
    max_users: int | None,
    salt: str,
    course_salt: str,
) -> tuple[int, int]:
    selected: dict[str, str] = {}
    enrollments: list[tuple[str, str, str]] = []
    with users_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "age", "gender", "education_level", "source_dataset"],
        )
        writer.writeheader()
        for index, item in enumerate(read_jsonl(root / "entities/user.json")):
            if max_users is not None and index >= max_users:
                break
            original_id = str(item["id"])
            user_id = anonymize_user_id(original_id, salt)
            selected[original_id] = user_id
            writer.writerow({
                "user_id": user_id,
                "age": "",
                "gender": "unknown",
                "education_level": "unknown",
                "source_dataset": "MOOCCube",
            })
            courses = item.get("course_order") or []
            times = item.get("enroll_time") or []
            for position, course_id in enumerate(courses):
                timestamp = str(times[position]) if position < len(times) else ""
                enrollments.append((
                    user_id,
                    anonymize_course_id(course_id, course_salt),
                    timestamp,
                ))

    with interactions_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERACTION_COLUMNS)
        writer.writeheader()
        for index, (user_id, course_id, timestamp) in enumerate(enrollments):
            # MOOCCube enrollment data lacks completion/quiz/engagement. These explicit
            # neutral proxies are marked derived and documented.
            completion = 0.5
            quiz = 0.5
            engagement = 0.5
            reward = 0.4 * completion + 0.4 * quiz + 0.2 * engagement
            writer.writerow({
                "interaction_id": f"i_{index}",
                "user_id": user_id,
                "course_id": course_id,
                "timestamp": timestamp,
                "rating": "",
                "clicked": 1,
                "video_views": 0,
                "completion_rate": completion,
                "quiz_score": quiz,
                "engagement_time": engagement,
                "access_frequency": 1,
                "session_duration": 0,
                "device_type": "unknown",
                "interaction_type": "enrollment",
                "source_dataset": "MOOCCube",
                "is_derived": True,
                "reward": reward,
            })
    return len(selected), len(enrollments)


def _aggregate_video_activity(events: list[dict], video_counts: Counter) -> list[dict]:
    courses: dict[str, dict] = {}
    for event in events:
        course_id = str(event.get("course_id") or "")
        video_id = str(event.get("video_id") or "")
        if not course_id or not video_id:
            continue
        item = courses.setdefault(course_id, {
            "videos": set(),
            "watching_count": 0.0,
            "watch_time": 0.0,
            "progress_time": 0.0,
            "duration": 0.0,
            "timestamps": [],
        })
        if video_id not in item["videos"]:
            item["videos"].add(video_id)
            item["duration"] += max(float(event.get("video_duration") or 0.0), 0.0)
        item["watching_count"] += max(float(event.get("watching_count") or 0.0), 0.0)
        item["watch_time"] += max(float(event.get("local_watching_time") or 0.0), 0.0)
        item["progress_time"] += max(float(event.get("video_progress_time") or 0.0), 0.0)
        timestamp = event.get("local_start_time") or event.get("local_end_time")
        if timestamp:
            item["timestamps"].append(str(timestamp))

    rows = []
    for course_id, item in courses.items():
        total_videos = max(int(video_counts.get(course_id, 0)), 1)
        completion = min(len(item["videos"]) / total_videos, 1.0)
        # Public MOOCCube has no quiz score. Progress coverage is an explicit proxy.
        quiz_proxy = min(item["progress_time"] / max(item["duration"], 1.0), 1.0)
        engagement = min(item["watch_time"] / max(item["duration"], 1.0), 1.0)
        reward = 0.4 * completion + 0.4 * quiz_proxy + 0.2 * engagement
        rows.append({
            "course_id": course_id,
            "timestamp": min(item["timestamps"]) if item["timestamps"] else "",
            "video_views": len(item["videos"]),
            "completion_rate": completion,
            "quiz_score": quiz_proxy,
            "engagement_time": engagement,
            "access_frequency": item["watching_count"],
            "session_duration": item["watch_time"],
            "reward": reward,
        })
    return rows


def export_video_activity(
    root: Path,
    users_output: Path,
    interactions_output: Path,
    max_users: int | None,
    salt: str,
    course_salt: str,
) -> tuple[int, int]:
    video_counts = _course_video_counts(root)
    user_count = 0
    interaction_count = 0
    with (
        users_output.open("w", encoding="utf-8", newline="") as users_handle,
        interactions_output.open("w", encoding="utf-8", newline="") as interactions_handle,
    ):
        users_writer = csv.DictWriter(
            users_handle,
            fieldnames=["user_id", "age", "gender", "education_level", "source_dataset"],
        )
        interactions_writer = csv.DictWriter(interactions_handle, fieldnames=INTERACTION_COLUMNS)
        users_writer.writeheader()
        interactions_writer.writeheader()
        for item in read_jsonl(root / "additional_information/user_video_act.json"):
            activity_rows = _aggregate_video_activity(item.get("activity") or [], video_counts)
            if not activity_rows:
                continue
            user_id = anonymize_user_id(item.get("id"), salt)
            users_writer.writerow({
                "user_id": user_id,
                "age": "",
                "gender": "unknown",
                "education_level": "unknown",
                "source_dataset": "MOOCCube",
            })
            for activity in activity_rows:
                interactions_writer.writerow({
                    "interaction_id": f"i_{interaction_count}",
                    "user_id": user_id,
                    "course_id": anonymize_course_id(activity["course_id"], course_salt),
                    "timestamp": activity["timestamp"],
                    "rating": "",
                    "clicked": 1,
                    "video_views": activity["video_views"],
                    "completion_rate": activity["completion_rate"],
                    "quiz_score": activity["quiz_score"],
                    "engagement_time": activity["engagement_time"],
                    "access_frequency": activity["access_frequency"],
                    "session_duration": activity["session_duration"],
                    "device_type": "unknown",
                    "interaction_type": "video_activity",
                    "source_dataset": "MOOCCube",
                    "is_derived": True,
                    "reward": activity["reward"],
                })
                interaction_count += 1
            user_count += 1
            if max_users is not None and user_count >= max_users:
                break
    return user_count, interaction_count


def prepare_mooccube(config: dict) -> dict:
    data_cfg = config["data"]
    root = project_path(config, data_cfg["mooccube_dir"])
    output = ensure_dir(project_path(config, data_cfg["processed_dir"]))
    course_salt = data_cfg.get("course_hash_salt", "introai-course")
    course_count = export_courses(root, output / "courses.csv", course_salt)
    if data_cfg.get("interaction_source") == "video_activity":
        user_count, interaction_count = export_video_activity(
            root,
            output / "users.csv",
            output / "interactions.csv",
            data_cfg.get("max_users"),
            data_cfg.get("user_hash_salt", "introai"),
            course_salt,
        )
    else:
        user_count, interaction_count = export_users_and_interactions(
            root,
            output / "users.csv",
            output / "interactions.csv",
            data_cfg.get("max_users"),
            data_cfg.get("user_hash_salt", "introai"),
            course_salt,
        )
    summary = {
        "source": "MOOCCube",
        "courses": course_count,
        "users": user_count,
        "interactions": interaction_count,
        "derived_feedback": True,
        "interaction_source": data_cfg.get("interaction_source", "enrollment"),
    }
    dump_json(output / "dataset_summary.json", summary)
    return summary


def split_interactions(df: pd.DataFrame, strategy: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cleaned = df.drop_duplicates(subset=["user_id", "course_id", "timestamp"]).copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True)
    if strategy == "paper_random":
        from sklearn.model_selection import train_test_split

        frequency = cleaned.groupby("user_id")["course_id"].transform("count")
        frequency_bin = pd.qcut(
            frequency.rank(method="first"),
            q=min(4, max(frequency.nunique(), 1)),
            labels=False,
            duplicates="drop",
        ).fillna(0).astype(int)
        difficulty = cleaned.get(
            "difficulty",
            pd.Series("unknown", index=cleaned.index),
        ).fillna("unknown").astype(str)
        strata = difficulty + "_" + frequency_bin.astype(str)
        counts = strata.value_counts()
        strata = strata.where(strata.map(counts) >= 2, "other")
        try:
            train, remainder = train_test_split(
                cleaned,
                test_size=0.30,
                random_state=seed,
                stratify=strata,
            )
            remainder_strata = strata.loc[remainder.index]
            remainder_counts = remainder_strata.value_counts()
            remainder_strata = remainder_strata.where(
                remainder_strata.map(remainder_counts) >= 2,
                "other",
            )
            validation, test = train_test_split(
                remainder,
                test_size=0.50,
                random_state=seed,
                stratify=remainder_strata,
            )
        except ValueError:
            shuffled = cleaned.sample(frac=1.0, random_state=seed)
            a, b = int(len(shuffled) * 0.70), int(len(shuffled) * 0.85)
            train, validation, test = shuffled.iloc[:a], shuffled.iloc[a:b], shuffled.iloc[b:]
        return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)

    train, val, test = [], [], []
    for _, group in cleaned.sort_values("timestamp").groupby("user_id", sort=False):
        n = len(group)
        train_end = max(1, math.ceil(n * 0.70))
        val_end = max(train_end, math.ceil(n * 0.85))
        train.append(group.iloc[:train_end])
        if val_end > train_end:
            val.append(group.iloc[train_end:val_end])
        if n > val_end:
            test.append(group.iloc[val_end:])
    empty = cleaned.iloc[:0]
    return (
        pd.concat(train, ignore_index=True) if train else empty,
        pd.concat(val, ignore_index=True) if val else empty,
        pd.concat(test, ignore_index=True) if test else empty,
    )


def build_splits(config: dict) -> dict:
    processed = project_path(config, config["data"]["processed_dir"])
    artifacts = ensure_dir(project_path(config, config["artifacts_dir"]))
    courses = pd.read_csv(processed / "courses.csv", dtype={"course_id": str})
    users = pd.read_csv(processed / "users.csv", dtype={"user_id": str})
    interactions = pd.read_csv(
        processed / "interactions.csv",
        dtype={"user_id": str, "course_id": str},
    )
    courses["difficulty"] = courses["difficulty"].fillna("unknown").astype(str).str.lower()
    difficulty_mapping = {"beginner": 0.0, "introductory": 0.0, "intermediate": 0.5, "advanced": 1.0}
    courses["difficulty_score"] = courses["difficulty"].map(difficulty_mapping)
    valid_courses = set(courses["course_id"])
    interactions = interactions[interactions["course_id"].isin(valid_courses)]
    interactions = interactions.merge(
        courses[["course_id", "difficulty", "difficulty_score"]],
        on="course_id",
        how="left",
    )
    train, val, test = split_interactions(
        interactions,
        config["data"].get("split_strategy", "temporal"),
        config.get("seed", 42),
    )
    numeric_columns = [
        "rating", "video_views", "completion_rate", "quiz_score",
        "engagement_time", "access_frequency", "session_duration",
        "difficulty_score", "reward",
    ]
    imputation = {}
    scaling = {}
    for column in numeric_columns:
        for frame in (train, val, test):
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        mean = float(train[column].mean()) if train[column].notna().any() else 0.0
        imputation[column] = mean
        for frame in (train, val, test):
            frame[column] = frame[column].fillna(mean)

    outlier_cfg = config["data"].get("outlier_filter", {})
    removed = {"train": 0, "validation": 0, "test": 0}
    if outlier_cfg.get("enabled", True):
        iqr_factor = float(outlier_cfg.get("iqr_factor", 3.0))
        bounds = {}
        for column in ("access_frequency", "session_duration"):
            q1, q3 = train[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            bounds[column] = (float(q1 - iqr_factor * iqr), float(q3 + iqr_factor * iqr))
        frames = [train, val, test]
        names = ["train", "validation", "test"]
        filtered = []
        for name, frame in zip(names, frames):
            keep = pd.Series(True, index=frame.index)
            for column, (lower, upper) in bounds.items():
                keep &= frame[column].between(lower, upper)
            if frame["rating"].notna().any():
                low_engagement = train["engagement_time"].quantile(0.10)
                keep &= ~((frame["rating"] >= 5.0) & (frame["engagement_time"] <= low_engagement))
            removed[name] = int((~keep).sum())
            filtered.append(frame.loc[keep].reset_index(drop=True))
        train, val, test = filtered
    for column in numeric_columns:
        minimum = float(train[column].min())
        maximum = float(train[column].max())
        scaling[column] = {"min": minimum, "max": maximum}
        if column in {"video_views", "access_frequency", "session_duration", "difficulty_score"}:
            denominator = maximum - minimum if maximum > minimum else 1.0
            for frame in (train, val, test):
                frame[column] = ((frame[column] - minimum) / denominator).clip(0.0, 1.0)
    courses.to_csv(artifacts / "courses.csv", index=False, encoding="utf-8")
    users.to_csv(artifacts / "users.csv", index=False, encoding="utf-8")
    train.to_csv(artifacts / "train.csv", index=False, encoding="utf-8")
    val.to_csv(artifacts / "val.csv", index=False, encoding="utf-8")
    test.to_csv(artifacts / "test.csv", index=False, encoding="utf-8")
    meta = {
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "split_strategy": config["data"].get("split_strategy", "temporal"),
        "mean_imputation_train_only": imputation,
        "minmax_scaling_train_only": scaling,
        "outliers_removed": removed,
        "course_ids_hashed": True,
    }
    dump_json(artifacts / "split_summary.json", meta)
    return meta
