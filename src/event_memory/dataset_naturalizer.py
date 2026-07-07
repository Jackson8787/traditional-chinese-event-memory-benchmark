from __future__ import annotations

from pathlib import Path

from .io import read_jsonl, write_jsonl


COPY_FILES = [
    "personas.jsonl",
    "gold_events.jsonl",
    "gold_update_relations.jsonl",
    "qa.jsonl",
]


def naturalize_dataset(source_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    turns = read_jsonl(source / "dialogues.jsonl")
    rewritten_turns = []
    for turn in turns:
        row = dict(turn)
        row["text"] = naturalize_turn_text(str(turn["text"]))
        rewritten_turns.append(row)

    write_jsonl(output / "dialogues.jsonl", rewritten_turns)
    for filename in COPY_FILES:
        source_path = source / filename
        if source_path.exists():
            with (output / filename).open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(source_path.read_text(encoding="utf-8"))
    _write_readme(output, source)

    return {
        "dialogue_turns": len(rewritten_turns),
        "texts_changed": sum(1 for old, new in zip(turns, rewritten_turns) if old["text"] != new["text"]),
    }


def _write_readme(output: Path, source: Path) -> None:
    readme = "\n".join(
        [
            "# v0 Naturalized Seed Benchmark",
            "",
            "Naturalized Traditional Chinese dialogue version of the seed benchmark.",
            "",
            "## Source",
            "",
            f"- Source dataset: `{source}`",
            "- `dialogues.jsonl` has rewritten user-facing text.",
            "- `personas.jsonl`, `gold_events.jsonl`, `gold_update_relations.jsonl`, and `qa.jsonl` are preserved from the source dataset.",
            "- Turn ids and evidence ids are unchanged, so existing gold labels remain valid.",
            "",
            "## Construction",
            "",
            "```powershell",
            '$env:PYTHONPATH="src"',
            r"C:\Users\o1000\anaconda3\envs\fuckyou\python.exe -m event_memory.cli naturalize-dataset --dataset-dir data\v0 --output-dir data\v0_naturalized",
            "```",
            "",
        ]
    )
    with (output / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)


def naturalize_turn_text(text: str) -> str:
    content = text.strip().rstrip("。")

    if content.startswith("我最近在") and "，先把時間排起來" in content:
        topic = content.removeprefix("我最近在").split("，", maxsplit=1)[0]
        return f"最近我把「{topic}」排進日程，想先把準備節奏抓穩。"

    if content.startswith("如果要安排地點，我一開始比較想選"):
        place = content.removeprefix("如果要安排地點，我一開始比較想選")
        return f"地點的話，我原本會比較傾向{place}。"

    if content.startswith("我其實一直") and "，這點應該不會變" in content:
        preference = content.removeprefix("我其實一直").split("，", maxsplit=1)[0]
        return f"說到偏好，我一直是{preference}，短期內應該不會變。"

    if "，那段時間不要排重要事情" in content:
        constraint = content.split("，", maxsplit=1)[0]
        return f"固定限制先記一下：{constraint}，所以那段不要排重要事情。"

    if "，很多安排都會一起確認" in content:
        relationship = content.split("，", maxsplit=1)[0]
        return f"很多安排我都會先確認，因為{relationship}。"

    if content.startswith("我已經") and "，接下來可以進下一步" in content:
        completed = content.removeprefix("我已經").split("，", maxsplit=1)[0]
        return f"{completed}這件事我已經處理完了，接著可以往下一步走。"

    if content.startswith("我後來不想繼續") and "，改成" in content:
        old_plan = content.removeprefix("我後來不想繼續").split("，", maxsplit=1)[0]
        new_plan = content.split("，改成", maxsplit=1)[1]
        return f"後來想了一下，我不打算繼續{old_plan}，現在改成{new_plan}。"

    if content.startswith("不是") and "，後來覺得" in content and "比較適合" in content:
        old_place = content.removeprefix("不是").split("，", maxsplit=1)[0]
        new_place = content.split("，後來覺得", maxsplit=1)[1].split("比較適合", maxsplit=1)[0]
        return f"地點要更新一下，不是{old_place}；後來覺得{new_place}比較適合。"

    if content.startswith("我還是") and "，只是想把方向收斂一點" in content:
        preference = content.removeprefix("我還是").split("，", maxsplit=1)[0]
        return f"偏好沒有變，我還是{preference}，只是想再把方向收斂一點。"

    if content.startswith("我希望") and "，這是下一個時間點" in content:
        future = content.removeprefix("我希望").split("，", maxsplit=1)[0]
        return f"接下來我希望先完成{future}，這會是下一個時間點。"

    return text
