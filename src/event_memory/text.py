from __future__ import annotations

import re


_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_STOP_PHRASES = [
    "使用者",
    "現在",
    "目前",
    "主要",
    "有沒有",
    "是否",
    "提過",
    "哪一",
    "哪裡",
    "什麼",
    "還在",
    "原本",
    "三月時",
]
_STOP_CHARS = set("的了在有沒不想問哪一個嗎呢啊是")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    for phrase in _STOP_PHRASES:
        text = text.replace(phrase, "")
    return text


def tokenize(text: str) -> set[str]:
    text = normalize_text(text)
    tokens = set(_ASCII_WORD.findall(text))
    cjk_chars = [ch for ch in text if _CJK.match(ch) and ch not in _STOP_CHARS]
    tokens.update(cjk_chars)
    tokens.update("".join(cjk_chars[i : i + 2]) for i in range(max(len(cjk_chars) - 1, 0)))
    return {token for token in tokens if token}


def overlap_score(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(len(left_tokens), len(right_tokens))
