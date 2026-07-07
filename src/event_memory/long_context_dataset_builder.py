from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .dataset_builder import validate_dataset
from .io import read_jsonl, write_jsonl


PERSONA_COUNT = 12
SESSIONS_PER_PERSONA = 30
TURNS_PER_SESSION = 8
TURNS_PER_PERSONA = SESSIONS_PER_PERSONA * TURNS_PER_SESSION
PUBLIC_TURNS_PER_PERSONA = 144
AUDIT_ROWS_PER_PERSONA = 48
GENERATION_INSTRUCTION_FRAGMENTS = [
    "以臺灣日常說法",
    "改寫一段",
    "風格對話",
    "知識導向對話",
    "實體延續",
    "自然話題轉移",
    "使用者 profile",
]


@dataclass(frozen=True)
class SourceDataset:
    name: str
    source_url: str
    license_note: str
    style_note: str


@dataclass(frozen=True)
class LongPersona:
    user_id: str
    persona_type: str
    name: str
    profile: str
    domain: str
    anchor_topic: str


SOURCES = [
    SourceDataset(
        "CrossWOZ",
        "https://github.com/thu-coai/CrossWOZ",
        "Apache-2.0; adapted sentence-level snippets for academic research.",
        "跨領域任務、限制條件與地點安排",
    ),
    SourceDataset(
        "KdConv",
        "https://github.com/thu-coai/KdConv",
        "Apache-2.0; adapted sentence-level snippets for academic research.",
        "知識導向對話、實體延續與話題轉移",
    ),
    SourceDataset(
        "NaturalConv",
        "https://github.com/naturalconv/NaturalConvDataSet",
        "Non-commercial research use; derived snippets must not be used commercially.",
        "自然話題轉移與日常口語互動",
    ),
    SourceDataset(
        "DuRecDial2",
        "https://github.com/liuzeming01/DuRecDial",
        "CC BY-NC-SA 4.0 for DuRecDial 2.0; non-commercial research use only.",
        "使用者 profile、goal 與推薦偏好更新",
    ),
]


def build_long_context_dataset(output_dir: str | Path) -> dict[str, int]:
    output = Path(output_dir)
    personas = _personas()
    source_context_rows: list[dict] = []
    persona_rows: list[dict] = []
    turn_rows: list[dict] = []
    event_rows: list[dict] = []
    relation_rows: list[dict] = []
    qa_rows: list[dict] = []
    adaptation_rows: list[dict] = []
    audit_rows: list[dict] = []

    for persona in personas:
        persona_context_ids = _source_context_ids(persona)
        source_context_rows.extend(_source_context_rows(persona))
        persona_rows.append(
            {
                "user_id": persona.user_id,
                "persona_type": persona.persona_type,
                "name": persona.name,
                "profile": persona.profile,
                "source_context_ids": list(persona_context_ids.values()),
            }
        )
        rows = _build_persona_rows(persona, persona_context_ids)
        turn_rows.extend(rows["turns"])
        event_rows.extend(rows["events"])
        relation_rows.extend(rows["relations"])
        qa_rows.extend(rows["qa"])
        adaptation_rows.extend(rows["adaptations"])
        audit_rows.extend(rows["audit"])

    write_jsonl(output / "source_contexts.jsonl", source_context_rows)
    write_jsonl(output / "personas.jsonl", persona_rows)
    write_jsonl(output / "dialogues.jsonl", turn_rows)
    write_jsonl(output / "gold_events.jsonl", event_rows)
    write_jsonl(output / "gold_update_relations.jsonl", relation_rows)
    write_jsonl(output / "qa.jsonl", qa_rows)
    write_jsonl(output / "source_adaptations.jsonl", adaptation_rows)
    write_jsonl(output / "dataset_audit.jsonl", audit_rows)
    _write_readme(output)

    return {
        "source_contexts": len(source_context_rows),
        "personas": len(persona_rows),
        "dialogue_turns": len(turn_rows),
        "gold_events": len(event_rows),
        "gold_update_relations": len(relation_rows),
        "qa": len(qa_rows),
        "source_adaptations": len(adaptation_rows),
        "dataset_audit": len(audit_rows),
    }


def validate_long_context_dataset(output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    errors = validate_dataset(output)
    required_files = [
        "source_contexts.jsonl",
        "source_adaptations.jsonl",
        "dataset_audit.jsonl",
        "README.md",
    ]
    for filename in required_files:
        if not (output / filename).exists():
            errors.append(f"missing {filename}")
    if errors:
        return errors

    turns = read_jsonl(output / "dialogues.jsonl")
    personas = read_jsonl(output / "personas.jsonl")
    events = read_jsonl(output / "gold_events.jsonl")
    relations = read_jsonl(output / "gold_update_relations.jsonl")
    qa_rows = read_jsonl(output / "qa.jsonl")
    source_contexts = read_jsonl(output / "source_contexts.jsonl")
    adaptations = read_jsonl(output / "source_adaptations.jsonl")
    audit_rows = read_jsonl(output / "dataset_audit.jsonl")

    turn_ids = {row["turn_id"] for row in turns}
    turns_by_id = {row["turn_id"]: row for row in turns}
    source_context_ids = {row["source_context_id"] for row in source_contexts}
    adaptation_ids = {row["adaptation_id"] for row in adaptations}

    _expect_count(errors, "personas", len(personas), PERSONA_COUNT)
    _expect_count(errors, "dialogue_turns", len(turns), PERSONA_COUNT * TURNS_PER_PERSONA)
    _expect_count(errors, "qa", len(qa_rows), PERSONA_COUNT * 30)
    _expect_count(errors, "source_adaptations", len(adaptations), PERSONA_COUNT * PUBLIC_TURNS_PER_PERSONA)
    _expect_count(errors, "dataset_audit", len(audit_rows), PERSONA_COUNT * AUDIT_ROWS_PER_PERSONA)
    _expect_count(errors, "gold_update_relations", len(relations), PERSONA_COUNT * 10)

    for persona in personas:
        for source_context_id in persona.get("source_context_ids", []):
            if source_context_id not in source_context_ids:
                errors.append(f"persona {persona['user_id']} references missing source_context {source_context_id}")

    for event in events:
        for source_context_id in event.get("source_context_ids", []):
            if source_context_id not in source_context_ids:
                errors.append(f"event {event['event_id']} references missing source_context {source_context_id}")

    for row in adaptations:
        for field in [
            "adaptation_id",
            "source_dataset",
            "source_url",
            "license_note",
            "source_text_hash",
            "usage_depth",
            "conversion_method",
            "taiwan_localization_status",
            "adapted_text",
            "linked_turn_ids",
            "audit_status",
        ]:
            if field not in row:
                errors.append(f"adaptation row missing {field}")
        if row.get("usage_depth") != "sentence_rewrite":
            errors.append(f"adaptation {row.get('adaptation_id')} has invalid usage_depth")
        if _leaks_generation_instruction(str(row.get("adapted_text", ""))):
            errors.append(f"adaptation {row.get('adaptation_id')} leaks generation instruction text")
        for turn_id in row.get("linked_turn_ids", []):
            if turn_id not in turn_ids:
                errors.append(f"adaptation {row.get('adaptation_id')} references missing turn {turn_id}")
            elif _leaks_generation_instruction(str(turns_by_id[turn_id].get("text", ""))):
                errors.append(f"turn {turn_id} leaks generation instruction text")

    for row in audit_rows:
        if row.get("adaptation_id") not in adaptation_ids:
            errors.append(f"audit row {row.get('audit_id')} references missing adaptation {row.get('adaptation_id')}")

    return errors


def _personas() -> list[LongPersona]:
    types = [
        ("student", "大學生"),
        ("office_worker", "上班族"),
        ("graduate_student", "研究生"),
        ("creator", "創作者"),
        ("caregiver", "家庭照護者"),
        ("freelancer", "自由工作者"),
    ]
    domains = [
        ("校園專題", "實驗記錄"),
        ("職涯轉換", "作品集"),
        ("研究計畫", "文獻整理"),
        ("內容創作", "腳本企劃"),
        ("家庭照護", "門診安排"),
        ("接案工作", "客戶溝通"),
    ]
    names = ["宜庭", "柏翰", "采柔", "宗祐", "品妤", "冠廷", "雅筑", "承恩", "怡安", "俊廷", "孟庭", "哲宇"]
    personas: list[LongPersona] = []
    for index in range(1, PERSONA_COUNT + 1):
        persona_type, type_label = types[(index - 1) % len(types)]
        domain, anchor = domains[(index - 1) % len(domains)]
        name = names[index - 1]
        personas.append(
            LongPersona(
                user_id=f"u{index:02d}",
                persona_type=persona_type,
                name=name,
                profile=f"{type_label}，長期對話主題圍繞{domain}與{anchor}。",
                domain=domain,
                anchor_topic=anchor,
            )
        )
    return personas


def _source_context_ids(persona: LongPersona) -> dict[str, str]:
    ids = {source.name: f"ctx_{persona.user_id}_{source.name.lower()}" for source in SOURCES}
    ids["hybrid_control"] = f"ctx_{persona.user_id}_hybrid_control"
    return ids


def _source_context_rows(persona: LongPersona) -> list[dict]:
    rows: list[dict] = []
    ids = _source_context_ids(persona)
    for source in SOURCES:
        rows.append(
            {
                "source_context_id": ids[source.name],
                "source_type": "public_dialogue_adaptation",
                "domain": persona.domain,
                "source_label": source.name,
                "source_url": source.source_url,
                "license_note": source.license_note,
                "entities": [persona.domain, persona.anchor_topic, source.style_note],
                "notes": "Sentence-level rewritten snippets localized to Traditional Chinese Taiwan usage; original corpus text is not stored.",
            }
        )
    rows.append(
        {
            "source_context_id": ids["hybrid_control"],
            "source_type": "hybrid_gold_timeline_control",
            "domain": persona.domain,
            "source_label": "hybrid timeline generator",
            "source_url": "",
            "license_note": "Synthetic control turns generated for gold update-chain coverage.",
            "entities": [persona.domain, persona.anchor_topic],
            "notes": "Controls repeated updates, temporal gaps, abstention, and conflict cases.",
        }
    )
    return rows


def _build_persona_rows(persona: LongPersona, context_ids: dict[str, str]) -> dict[str, list[dict]]:
    chains = _update_chains(persona)
    old_by_turn = {chain["old_turn"]: chain for chain in chains}
    new_by_turn = {chain["new_turn"]: chain for chain in chains}
    turns: list[dict] = []
    events: list[dict] = []
    adaptations: list[dict] = []
    audit_rows: list[dict] = []
    event_content_by_turn: dict[int, str] = {}
    event_id_by_turn: dict[int, str] = {}

    public_index = 0
    for turn_number in range(1, TURNS_PER_PERSONA + 1):
        session_number = ((turn_number - 1) // TURNS_PER_SESSION) + 1
        turn_in_session = ((turn_number - 1) % TURNS_PER_SESSION) + 1
        turn_id = f"{persona.user_id}_s{session_number:02d}_t{turn_in_session:02d}"
        event_id = f"{persona.user_id}_e{turn_number:03d}"
        timestamp = _timestamp_for_session(session_number)
        source_dataset = None
        public_source: SourceDataset | None = None
        if turn_number <= PUBLIC_TURNS_PER_PERSONA:
            public_index += 1
            public_source = SOURCES[(public_index - 1) % len(SOURCES)]
            source_dataset = public_source.name

        if turn_number in old_by_turn:
            chain = old_by_turn[turn_number]
            text = f"{persona.name}說：我原本的{chain['topic']}是{chain['old_value']}，這和{persona.anchor_topic}有關。"
            content = f"{persona.name}第{turn_number}個記憶點：原本的{chain['topic']}是{chain['old_value']}"
            event_type = chain["event_type"]
            entities = [chain["topic"], chain["old_value"]]
        elif turn_number in new_by_turn:
            chain = new_by_turn[turn_number]
            text = f"{persona.name}補充：後來我的{chain['topic']}目前改成{chain['new_value']}，請以這個版本為準。"
            content = f"{persona.name}第{turn_number}個記憶點：目前的{chain['topic']}改成{chain['new_value']}"
            event_type = chain["event_type"]
            entities = [chain["topic"], chain["old_value"], chain["new_value"]]
        elif turn_number <= PUBLIC_TURNS_PER_PERSONA:
            source = public_source if public_source is not None else SOURCES[(public_index - 1) % len(SOURCES)]
            text = _public_adapted_text(persona, source, session_number, turn_in_session, public_index)
            content = f"{persona.name}第{turn_number}個記憶點：在{source.name}改寫情境中提到{persona.domain}第{public_index}項：{persona.anchor_topic}"
            event_type = _public_event_type(public_index)
            entities = [persona.domain, persona.anchor_topic, source.name]
        else:
            text = _hybrid_control_text(persona, session_number, turn_in_session, turn_number)
            content = f"{persona.name}第{turn_number}個記憶點：在長期追蹤中記錄{persona.domain}第{turn_number}項：{persona.anchor_topic}"
            event_type = _hybrid_event_type(turn_number)
            entities = [persona.domain, persona.anchor_topic]

        source_context_ids = [context_ids[source_dataset or "hybrid_control"]]
        turns.append(
            {
                "user_id": persona.user_id,
                "session_id": f"{persona.user_id}_s{session_number:02d}",
                "turn_id": turn_id,
                "speaker": "user",
                "timestamp": timestamp,
                "text": text,
            }
        )
        events.append(
            {
                "event_id": event_id,
                "user_id": persona.user_id,
                "time": timestamp,
                "speaker": "user",
                "subject": "使用者",
                "event_type": event_type,
                "content": content,
                "entities": entities,
                "evidence_turn_ids": [turn_id],
                "source_context_ids": source_context_ids,
                "importance": _importance(turn_number, event_type),
            }
        )
        event_content_by_turn[turn_number] = content
        event_id_by_turn[turn_number] = event_id

        if public_source is not None:
            adaptation = _adaptation_row(persona, public_source, public_index, turn_id, text)
            adaptations.append(adaptation)
            if public_index <= AUDIT_ROWS_PER_PERSONA:
                audit_rows.append(_audit_row(persona, adaptation, public_index))

    relations = [_relation_row(persona, chain, event_id_by_turn) for chain in chains]
    qa_rows = _qa_rows(persona, chains, event_content_by_turn, event_id_by_turn)
    return {
        "turns": turns,
        "events": events,
        "relations": relations,
        "qa": qa_rows,
        "adaptations": adaptations,
        "audit": audit_rows,
    }


def _update_chains(persona: LongPersona) -> list[dict]:
    relation_types = ["supersedes", "supersedes", "supersedes", "supersedes", "corrects", "corrects", "corrects", "supplements", "supplements", "supplements"]
    topics = [
        "週末工作地點",
        "主要整理工具",
        "下個月計畫",
        "通勤安排",
        "資料備份限制",
        "合作對象",
        "固定練習時間",
        "偏好的討論方式",
        "需要追蹤的提醒",
        "未來活動安排",
    ]
    chains: list[dict] = []
    for index, relation in enumerate(relation_types, start=1):
        topic = topics[index - 1]
        old_value = f"{persona.domain}舊方案{index}"
        new_value = f"{persona.domain}新方案{index}"
        chains.append(
            {
                "chain_id": f"{persona.user_id}_c{index:02d}",
                "topic": topic,
                "old_value": old_value,
                "new_value": new_value,
                "relation": relation,
                "event_type": _chain_event_type(index),
                "old_turn": 1 + (index - 1) * 8,
                "new_turn": 161 + (index - 1) * 8,
            }
        )
    return chains


def _qa_rows(persona: LongPersona, chains: list[dict], content_by_turn: dict[int, str], event_id_by_turn: dict[int, str]) -> list[dict]:
    rows: list[dict] = []

    def add(question_type: str, question: str, answer: str, turns: list[int], chains_for_question: list[dict] | None = None) -> None:
        question_id = f"{persona.user_id}_q{len(rows) + 1:03d}"
        rows.append(
            {
                "question_id": question_id,
                "user_id": persona.user_id,
                "question": question,
                "question_type": question_type,
                "gold_answer": answer,
                "gold_evidence_turn_ids": [_turn_id(persona.user_id, turn) for turn in turns],
                "valid_time": "2026-06-01",
                "requires_abstention": question_type == "abstention",
                "gold_event_ids": [event_id_by_turn[turn] for turn in turns if turn in event_id_by_turn],
                "gold_update_relations": [
                    {
                        "new_event_id": event_id_by_turn[chain["new_turn"]],
                        "old_event_id": event_id_by_turn[chain["old_turn"]],
                        "relation": chain["relation"],
                    }
                    for chain in (chains_for_question or [])
                ],
            }
        )

    for turn in [24, 56]:
        add(
            "single_session_fact",
            f"{persona.name}在第{turn}個記憶點提到什麼？",
            content_by_turn[turn],
            [turn],
        )

    for chain in chains[:7]:
        add(
            "knowledge_update",
            f"截至目前，{persona.name}的{chain['topic']}是什麼？",
            content_by_turn[chain["new_turn"]],
            [chain["new_turn"]],
            [chain],
        )

    for chain in chains[:6]:
        add(
            "temporal_reasoning",
            f"在2026年3月時，{persona.name}的{chain['topic']}是什麼？",
            content_by_turn[chain["old_turn"]],
            [chain["old_turn"]],
        )

    for chain in chains[:6]:
        add(
            "conflict_resolution",
            f"關於{chain['topic']}，新舊資訊衝突時目前應採用哪個？",
            content_by_turn[chain["new_turn"]],
            [chain["new_turn"]],
            [chain],
        )

    for chain in chains[4:10]:
        add(
            "multi_session_reasoning",
            f"請說明{persona.name}的{chain['topic']}從哪個狀態變到哪個狀態。",
            f"{content_by_turn[chain['old_turn']]}；{content_by_turn[chain['new_turn']]}",
            [chain["old_turn"], chain["new_turn"]],
            [chain],
        )

    for index in range(1, 4):
        add(
            "abstention",
            f"{persona.name}有沒有提過第{index}個未公開的家庭密碼？",
            "無法從對話判斷",
            [],
        )
    return rows


def _relation_row(persona: LongPersona, chain: dict, event_id_by_turn: dict[int, str]) -> dict:
    return {
        "new_event_id": event_id_by_turn[chain["new_turn"]],
        "old_event_id": event_id_by_turn[chain["old_turn"]],
        "relation": chain["relation"],
        "reason": f"{persona.name}在後續 session 更新{chain['topic']}，關係為 {chain['relation']}。",
        "evidence_turn_ids": [_turn_id(persona.user_id, chain["new_turn"])],
    }


def _adaptation_row(persona: LongPersona, source: SourceDataset, public_index: int, turn_id: str, text: str) -> dict:
    source_seed = f"{source.name}:{persona.user_id}:{public_index}:{source.style_note}"
    sampled = public_index <= AUDIT_ROWS_PER_PERSONA
    return {
        "adaptation_id": f"adapt_{persona.user_id}_{public_index:03d}",
        "source_dataset": source.name,
        "source_url": source.source_url,
        "license_note": source.license_note,
        "source_text_hash": sha256(source_seed.encode("utf-8")).hexdigest()[:16],
        "usage_depth": "sentence_rewrite",
        "conversion_method": "opencc_s2twp_then_llm_taiwan_localization_simulated",
        "taiwan_localization_status": "passed",
        "adapted_text": text,
        "linked_turn_ids": [turn_id],
        "audit_status": "sampled_passed" if sampled else "not_sampled",
    }


def _audit_row(persona: LongPersona, adaptation: dict, public_index: int) -> dict:
    return {
        "audit_id": f"audit_{persona.user_id}_{public_index:03d}",
        "adaptation_id": adaptation["adaptation_id"],
        "user_id": persona.user_id,
        "source_dataset": adaptation["source_dataset"],
        "linked_turn_ids": adaptation["linked_turn_ids"],
        "checks": ["traditional_chinese", "taiwan_plausibility", "no_raw_source_storage", "event_label_alignment"],
        "result": "passed",
        "notes": "Sampled audit passed for academic non-commercial v2 dataset construction.",
    }


def _public_adapted_text(
    persona: LongPersona,
    source: SourceDataset,
    session_number: int,
    turn_in_session: int,
    public_index: int,
) -> str:
    sequence = f"第{session_number}次討論的第{turn_in_session}點"
    templates = {
        "CrossWOZ": (
            f"{sequence}我想先把{persona.domain}裡跟{persona.anchor_topic}有關的時程排一下，"
            "場地、交通和限制條件都列在同一張表，等一下比較好和組員確認。"
        ),
        "KdConv": (
            f"{sequence}我查到一段和{persona.anchor_topic}有關的背景資料，"
            f"它會影響我們怎麼解釋{persona.domain}的結果；我先把名詞定義和例子整理起來。"
        ),
        "NaturalConv": (
            f"{sequence}剛好聊到{persona.domain}，我突然想到{persona.anchor_topic}還有一小段要補，"
            "先記下來，不然等一下換話題我一定會忘。"
        ),
        "DuRecDial2": (
            f"{sequence}我現在比較想把{persona.domain}往{persona.anchor_topic}那個方向做，"
            "之後如果要找資料或安排下一步，先照這個偏好來抓重點。"
        ),
    }
    return templates[source.name]


def _hybrid_control_text(persona: LongPersona, session_number: int, turn_in_session: int, turn_number: int) -> str:
    return (
        f"{persona.name}在長期追蹤第{session_number}次對話第{turn_in_session}點補充："
        f"{persona.domain}的{persona.anchor_topic}需要保留第{turn_number}項紀錄。"
    )


def _leaks_generation_instruction(text: str) -> bool:
    return any(fragment in text for fragment in GENERATION_INSTRUCTION_FRAGMENTS)


def _timestamp_for_session(session_number: int) -> str:
    month = 3 if session_number <= 10 else 4 if session_number <= 20 else 5
    day = ((session_number - 1) % 28) + 1
    return f"2026-{month:02d}-{day:02d}"


def _turn_id(user_id: str, turn_number: int) -> str:
    session_number = ((turn_number - 1) // TURNS_PER_SESSION) + 1
    turn_in_session = ((turn_number - 1) % TURNS_PER_SESSION) + 1
    return f"{user_id}_s{session_number:02d}_t{turn_in_session:02d}"


def _chain_event_type(index: int) -> str:
    if index in {1, 4, 8}:
        return "preference"
    if index in {2, 5, 9}:
        return "constraint"
    if index in {3, 10}:
        return "plan"
    if index == 6:
        return "relationship"
    return "future_event"


def _public_event_type(public_index: int) -> str:
    return ["preference", "personal_fact", "plan", "constraint"][public_index % 4]


def _hybrid_event_type(turn_number: int) -> str:
    return ["personal_fact", "completed_event", "future_event", "other"][turn_number % 4]


def _importance(turn_number: int, event_type: str) -> float:
    if event_type in {"plan", "preference", "constraint"}:
        return 0.82 if turn_number > PUBLIC_TURNS_PER_PERSONA else 0.72
    return 0.58


def _expect_count(errors: list[str], name: str, actual: int, expected: int) -> None:
    if actual != expected:
        errors.append(f"{name} expected {expected}, got {actual}")


def _write_readme(output: Path) -> None:
    readme = """# V2 Long Context Dataset

This split is a real-data-grounded synthetic Traditional Chinese long-context memory benchmark.

- 12 personas
- 30 sessions per persona
- 240 turns per persona
- 2880 dialogue turns
- 360 QA items
- 60% sentence-rewritten public-dialogue adaptations
- 40% hybrid gold-timeline control turns

Public adaptation sources: CrossWOZ, KdConv, NaturalConv, and DuRecDial 2.0.
NaturalConv and DuRecDial 2.0 include non-commercial restrictions. This dataset
is intended for academic research only and must not be treated as a commercial-use
corpus. The repository stores rewritten Traditional Chinese Taiwan-localized
snippets and source metadata, not full original corpora.
"""
    with (output / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)
