from __future__ import annotations

import re


GLOSSARY = {
    "C语言程序设计（下）": "C Programming (Part 2)",
    "测试系统集成技术": "Test System Integration Technology",
    "C++语言程序设计进阶": "Advanced C++ Programming",
    "“做中学”Java程序设计": "Learning by Doing: Java Programming",
    "大学计算机基础": "University Computer Fundamentals",
    "语言程序设计": "Programming",
    "程序设计": "Programming",
    "系统集成": "System Integration",
    "进阶": "Advanced",
    "春": "Spring",
    "下": "Part 2",
    "自然灾害": "Natural Disasters",
    "自主模式": "Self-Paced Mode",
    "大学": "University",
    "研究生": "Graduate",
    "学位论文答辩": "Thesis Defense",
    "计算机": "Computer",
    "科学": "Science",
    "技术": "Technology",
    "数学": "Mathematics",
    "物理": "Physics",
    "化学": "Chemistry",
    "经济": "Economics",
    "管理": "Management",
    "教育": "Education",
    "历史": "History",
    "艺术": "Arts",
    "课程": "Course",
    "导论": "Introduction",
    "基础": "Fundamentals",
    "无先修要求": "No prerequisites",
    "无": "None",
}


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


class TopKTranslator:
    """Translate only ranked output. Transformer use is optional and lazy."""

    def __init__(self, backend: str = "offline_glossary", model_name: str = "Helsinki-NLP/opus-mt-zh-en") -> None:
        self.backend = backend
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, local_files_only=True,
            ).to(self._device).eval()
        return self._tokenizer, self._model

    def translate(self, text: object, max_chars: int = 500) -> str:
        source = " ".join(str(text or "").split())[:max_chars]
        if not source or not contains_cjk(source):
            return source
        if self.backend == "transformers":
            try:
                import torch

                tokenizer, model = self._load_model()
                tokens = tokenizer(
                    source,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(self._device)
                with torch.no_grad():
                    generated = model.generate(**tokens, max_new_tokens=192, num_beams=4)
                return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            except Exception:
                pass
        translated = source
        matched = []
        for chinese, english in sorted(GLOSSARY.items(), key=lambda item: len(item[0]), reverse=True):
            if chinese in translated:
                matched.append(english)
            translated = translated.replace(chinese, english)
        if contains_cjk(translated):
            topics = ", ".join(dict.fromkeys(matched)) or "general course"
            return (
                f"Chinese-language course about {topics}. "
                "[Install/download the configured MarianMT model for a full translation.]"
            )
        return translated
