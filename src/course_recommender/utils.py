from __future__ import annotations

import hashlib
import html
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_pairs(path: str | Path) -> Iterable[tuple[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def repair_mojibake(value: object) -> str:
    """Repair the common UTF-8-as-Latin-1/CP1252 corruption in MOOCCube."""
    if value is None:
        return ""
    text = str(value)
    try:
        raw = bytearray()
        for character in text:
            codepoint = ord(character)
            if codepoint <= 255:
                raw.append(codepoint)
            else:
                raw.extend(character.encode("cp1252"))
        repaired = bytes(raw).decode("utf-8")
        if repaired != text:
            text = repaired
    except (UnicodeEncodeError, UnicodeDecodeError, ValueError):
        pass
    return text


def clean_text(value: object, max_chars: int = 3000) -> str:
    text = repair_mojibake(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def anonymize_user_id(user_id: object, salt: str) -> str:
    raw = f"{salt}:{user_id}".encode("utf-8")
    return "u_" + hashlib.sha256(raw).hexdigest()[:20]


def anonymize_course_id(course_id: object, salt: str) -> str:
    raw = f"{salt}:{course_id}".encode("utf-8")
    return "c_" + hashlib.sha256(raw).hexdigest()[:20]


def dump_json(path: str | Path, value: object) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))
