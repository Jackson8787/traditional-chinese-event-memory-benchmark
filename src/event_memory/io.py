from __future__ import annotations

import json
from pathlib import Path
from dataclasses import fields
from typing import Iterable, TypeVar

from .schema import DialogueTurn, QAItem

T = TypeVar("T")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_turns(path: str | Path) -> list[DialogueTurn]:
    return [DialogueTurn(**row) for row in read_jsonl(path)]


def load_qa(path: str | Path) -> list[QAItem]:
    allowed_fields = {field.name for field in fields(QAItem)}
    return [QAItem(**{key: value for key, value in row.items() if key in allowed_fields}) for row in read_jsonl(path)]


def turns_by_id(turns: Iterable[DialogueTurn]) -> dict[str, DialogueTurn]:
    return {turn.turn_id: turn for turn in turns}
