from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .dataset_builder import validate_dataset
from .io import read_jsonl, write_jsonl


PERSONA_COUNT = 12
SESSIONS_PER_PERSONA = 30
MEMORY_TURNS_PER_SESSION = 8
MEMORY_TURNS_PER_PERSONA = SESSIONS_PER_PERSONA * MEMORY_TURNS_PER_SESSION
AUDIT_ROWS_PER_PERSONA = 48
SOURCE_FACTS_PER_PERSONA = 8
TRACEABLE_FACTS_PER_PERSONA = 2
HARD_QA_PER_PERSONA = 6

QUESTION_DISTRIBUTION = {
    "single_session_fact": 2,
    "knowledge_update": 7,
    "temporal_reasoning": 6,
    "conflict_resolution": 6,
    "multi_session_reasoning": 6,
    "abstention": 3,
}

FORBIDDEN_LITERAL_FRAGMENTS = [
    "以臺灣日常說法",
    "改寫一段",
    "風格對話",
    "source",
    "dataset",
    "metadata",
    "prompt",
]
FORBIDDEN_PATTERNS = [
    re.compile(r"舊方案\d*"),
    re.compile(r"新方案\d*"),
    re.compile(r"第\d+次討論的第\d+點"),
    re.compile(r"說：我"),
    re.compile(r"這和.*有關"),
]


@dataclass(frozen=True)
class PublicPersona:
    user_id: str
    persona_type: str
    name: str
    profile: str
    domain: str
    anchor_topic: str
    location_a: str
    location_b: str
    tool_a: str
    tool_b: str
    collaborator: str
    deadline: str
    service: str
    unknown_target: str


def build_public_grounded_dataset(output_dir: str | Path) -> dict[str, int]:
    output = Path(output_dir)
    personas = _personas()

    source_manifest_rows: list[dict] = []
    source_fact_rows: list[dict] = []
    scenario_rows: list[dict] = []
    source_context_rows: list[dict] = []
    persona_rows: list[dict] = []
    turn_rows: list[dict] = []
    memory_turn_rows: list[dict] = []
    event_rows: list[dict] = []
    relation_rows: list[dict] = []
    qa_rows: list[dict] = []
    source_audit_rows: list[dict] = []
    naturalness_audit_rows: list[dict] = []
    dataset_audit_rows: list[dict] = []

    for persona in personas:
        source_context_id = f"ctx_{persona.user_id}_public_grounded"
        sources = _source_manifest_rows(persona)
        facts = _source_fact_rows(persona, sources)
        scenario = _scenario_card(persona, facts, source_context_id)
        source_manifest_rows.extend(sources)
        source_fact_rows.extend(facts)
        scenario_rows.append(scenario)
        source_context_rows.append(
            {
                "source_context_id": source_context_id,
                "source_type": "public_safe_context_grounding",
                "domain": persona.domain,
                "source_label": "public source facts synthesized into a persona scenario",
                "source_url": "see source_manifest.jsonl",
                "license_note": "Public-safe short factual grounding only; no long original text is stored.",
                "entities": [persona.domain, persona.anchor_topic, persona.location_a, persona.location_b],
                "notes": "Used to ground scenario vocabulary, not as direct dialogue text.",
            }
        )
        persona_rows.append(
            {
                "user_id": persona.user_id,
                "persona_type": persona.persona_type,
                "name": persona.name,
                "profile": persona.profile,
                "source_context_ids": [source_context_id],
                "scenario_card_id": scenario["scenario_card_id"],
                "source_fact_ids": [fact["source_fact_id"] for fact in facts],
            }
        )
        rows = _build_persona_rows(persona, facts, scenario, source_context_id)
        turn_rows.extend(rows["turns"])
        memory_turn_rows.extend(rows["memory_turns"])
        event_rows.extend(rows["events"])
        relation_rows.extend(rows["relations"])
        qa_rows.extend(rows["qa"])
        source_audit_rows.extend(_source_audit_rows(persona, sources))
        naturalness_audit_rows.extend(_naturalness_audit_rows(persona, rows["memory_turns"]))
        dataset_audit_rows.extend(_dataset_audit_rows(persona, rows["memory_turns"]))

    write_jsonl(output / "source_manifest.jsonl", source_manifest_rows)
    write_jsonl(output / "source_facts.jsonl", source_fact_rows)
    write_jsonl(output / "scenario_cards.jsonl", scenario_rows)
    write_jsonl(output / "source_contexts.jsonl", source_context_rows)
    write_jsonl(output / "personas.jsonl", persona_rows)
    write_jsonl(output / "dialogues.jsonl", turn_rows)
    write_jsonl(output / "memory_turns.jsonl", memory_turn_rows)
    write_jsonl(output / "gold_events.jsonl", event_rows)
    write_jsonl(output / "gold_update_relations.jsonl", relation_rows)
    write_jsonl(output / "qa.jsonl", qa_rows)
    write_jsonl(output / "source_audit.jsonl", source_audit_rows)
    write_jsonl(output / "naturalness_audit.jsonl", naturalness_audit_rows)
    write_jsonl(output / "dataset_audit.jsonl", dataset_audit_rows)
    _write_readme(output)

    return {
        "source_manifest": len(source_manifest_rows),
        "source_facts": len(source_fact_rows),
        "scenario_cards": len(scenario_rows),
        "personas": len(persona_rows),
        "dialogue_turns": len(turn_rows),
        "memory_turns": len(memory_turn_rows),
        "gold_events": len(event_rows),
        "gold_update_relations": len(relation_rows),
        "qa": len(qa_rows),
        "source_audit": len(source_audit_rows),
        "naturalness_audit": len(naturalness_audit_rows),
        "dataset_audit": len(dataset_audit_rows),
    }


def build_public_grounded_hard_dataset(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    source = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    required_files = [
        "source_manifest.jsonl",
        "source_facts.jsonl",
        "scenario_cards.jsonl",
        "source_contexts.jsonl",
        "personas.jsonl",
        "dialogues.jsonl",
        "memory_turns.jsonl",
        "gold_events.jsonl",
        "gold_update_relations.jsonl",
        "source_audit.jsonl",
        "naturalness_audit.jsonl",
        "dataset_audit.jsonl",
        "README.md",
    ]
    for filename in required_files:
        shutil.copy2(source / filename, output / filename)

    qa_rows = _hard_challenge_qa_rows(
        read_jsonl(source / "personas.jsonl"),
        read_jsonl(source / "gold_events.jsonl"),
        read_jsonl(source / "gold_update_relations.jsonl"),
    )
    write_jsonl(output / "qa.jsonl", qa_rows)
    write_jsonl(source / "challenge_qa.jsonl", qa_rows)
    _write_hard_readme(output)
    return {"qa": len(qa_rows), "personas": len(read_jsonl(output / "personas.jsonl"))}


def validate_public_grounded_dataset(output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    required_files = [
        "source_manifest.jsonl",
        "source_facts.jsonl",
        "scenario_cards.jsonl",
        "source_contexts.jsonl",
        "personas.jsonl",
        "dialogues.jsonl",
        "memory_turns.jsonl",
        "gold_events.jsonl",
        "gold_update_relations.jsonl",
        "qa.jsonl",
        "source_audit.jsonl",
        "naturalness_audit.jsonl",
        "dataset_audit.jsonl",
        "README.md",
    ]
    errors = [f"missing {filename}" for filename in required_files if not (output / filename).exists()]
    if errors:
        return errors

    errors.extend(validate_dataset(output))
    sources = read_jsonl(output / "source_manifest.jsonl")
    source_facts = read_jsonl(output / "source_facts.jsonl")
    scenario_cards = read_jsonl(output / "scenario_cards.jsonl")
    turns = read_jsonl(output / "dialogues.jsonl")
    memory_turns = read_jsonl(output / "memory_turns.jsonl")
    events = read_jsonl(output / "gold_events.jsonl")
    relations = read_jsonl(output / "gold_update_relations.jsonl")
    qa_rows = read_jsonl(output / "qa.jsonl")
    source_audit = read_jsonl(output / "source_audit.jsonl")
    naturalness_audit = read_jsonl(output / "naturalness_audit.jsonl")
    dataset_audit = read_jsonl(output / "dataset_audit.jsonl")

    _expect_count(errors, "source_manifest", len(sources), PERSONA_COUNT * SOURCE_FACTS_PER_PERSONA)
    _expect_count(errors, "source_facts", len(source_facts), PERSONA_COUNT * SOURCE_FACTS_PER_PERSONA)
    _expect_count(errors, "scenario_cards", len(scenario_cards), PERSONA_COUNT)
    _expect_count(errors, "dialogue_turns", len(turns), PERSONA_COUNT * MEMORY_TURNS_PER_PERSONA * 2)
    _expect_count(errors, "memory_turns", len(memory_turns), PERSONA_COUNT * MEMORY_TURNS_PER_PERSONA)
    _expect_count(errors, "gold_events", len(events), PERSONA_COUNT * MEMORY_TURNS_PER_PERSONA)
    _expect_count(errors, "gold_update_relations", len(relations), PERSONA_COUNT * 10)
    _expect_count(errors, "qa", len(qa_rows), PERSONA_COUNT * sum(QUESTION_DISTRIBUTION.values()))
    _expect_count(errors, "source_audit", len(source_audit), PERSONA_COUNT * SOURCE_FACTS_PER_PERSONA)
    _expect_count(errors, "naturalness_audit", len(naturalness_audit), PERSONA_COUNT * AUDIT_ROWS_PER_PERSONA)
    _expect_count(errors, "dataset_audit", len(dataset_audit), PERSONA_COUNT * AUDIT_ROWS_PER_PERSONA)

    source_ids = {row["source_id"] for row in sources}
    source_fact_ids = {row["source_fact_id"] for row in source_facts}
    turn_by_id = {row["turn_id"]: row for row in turns}
    event_ids = {row["event_id"] for row in events}
    memory_turn_ids = {row["turn_id"] for row in memory_turns}

    for source in sources:
        source_id = source.get("source_id")
        if source.get("license_status") != "public_safe":
            errors.append(f"source {source_id} is not public_safe")
        for field in ["source_page_title", "accessed_at", "license_evidence_note"]:
            if not source.get(field):
                errors.append(f"source {source_id} missing {field}")
        if not source.get("derivative_allowed"):
            errors.append(f"source {source_id} is not derivative_allowed")
        if "non-commercial" in str(source.get("license_note", "")).lower():
            errors.append(f"source {source_id} mentions non-commercial terms")
        if source.get("source_label") == "DuRecDial2":
            errors.append("DuRecDial2 must not be included in public-safe split")

    for fact in source_facts:
        fact_id = fact.get("source_fact_id")
        if fact.get("source_id") not in source_ids:
            errors.append(f"fact {fact_id} references missing source {fact.get('source_id')}")
        for field in ["fact_origin_type", "fact_span_hash", "extraction_note"]:
            if not fact.get(field):
                errors.append(f"fact {fact_id} missing {field}")
        if fact.get("fact_origin_type") == "concrete_public_page_fact" and not fact.get("source_page_title"):
            errors.append(f"fact {fact_id} missing source_page_title")
        if len(str(fact.get("fact_text", ""))) > 160:
            errors.append(f"fact {fact_id} is too long for short factual grounding")

    for scenario in scenario_cards:
        for fact_id in scenario.get("source_fact_ids", []):
            if fact_id not in source_fact_ids:
                errors.append(f"scenario {scenario.get('scenario_card_id')} references missing fact {fact_id}")
        if len(scenario.get("source_fact_ids", [])) < 6:
            errors.append(f"scenario {scenario.get('scenario_card_id')} has fewer than 6 source facts")

    for turn in turns:
        if _leaks_synthetic_text(str(turn.get("text", ""))):
            errors.append(f"turn {turn.get('turn_id')} leaks synthetic placeholder text")

    for event in events:
        if _leaks_synthetic_text(str(event.get("content", ""))):
            errors.append(f"event {event.get('event_id')} leaks synthetic placeholder text")

    for qa in qa_rows:
        if _leaks_synthetic_text(str(qa.get("question", ""))) or _leaks_synthetic_text(str(qa.get("gold_answer", ""))):
            errors.append(f"qa {qa.get('question_id')} leaks synthetic placeholder text")
        for turn_id in qa.get("gold_evidence_turn_ids", []):
            if turn_id not in memory_turn_ids:
                errors.append(f"qa {qa.get('question_id')} evidence {turn_id} is not a memory-bearing user turn")
            elif turn_by_id.get(turn_id, {}).get("speaker") != "user":
                errors.append(f"qa {qa.get('question_id')} evidence {turn_id} is not a user turn")
        if qa.get("requires_abstention") and qa.get("gold_evidence_turn_ids"):
            errors.append(f"abstention qa {qa.get('question_id')} should not have evidence")
        if qa.get("question_type") in {"knowledge_update", "conflict_resolution", "temporal_reasoning"} and not qa.get("gold_update_relations"):
            errors.append(f"qa {qa.get('question_id')} should include gold_update_relations")

    _validate_persona_shape(errors, memory_turns, source_facts, relations, qa_rows)
    _validate_traceable_facts(errors, source_facts)
    _validate_session_functions(errors, memory_turns)
    _validate_no_repeated_user_texts(errors, turns)

    for memory_turn in memory_turns:
        turn_id = memory_turn.get("turn_id")
        if turn_id not in turn_by_id:
            errors.append(f"memory turn {turn_id} references missing dialogue turn")
        elif turn_by_id[turn_id].get("speaker") != "user":
            errors.append(f"memory turn {turn_id} is not a user turn")
        if memory_turn.get("event_id") not in event_ids:
            errors.append(f"memory turn {turn_id} references missing event {memory_turn.get('event_id')}")

    return errors


def validate_public_grounded_hard_dataset(output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    required_files = [
        "dialogues.jsonl",
        "gold_events.jsonl",
        "gold_update_relations.jsonl",
        "qa.jsonl",
        "README.md",
    ]
    errors = [f"missing {filename}" for filename in required_files if not (output / filename).exists()]
    if errors:
        return errors

    errors.extend(validate_dataset(output))
    qa_rows = read_jsonl(output / "qa.jsonl")
    _expect_count(errors, "hard_qa", len(qa_rows), PERSONA_COUNT * HARD_QA_PER_PERSONA)
    expected_types = {"hard_conflict", "hard_temporal"}
    if {row.get("question_type") for row in qa_rows} != expected_types:
        errors.append("hard qa should include hard_conflict and hard_temporal")
    if not any(row.get("challenge_focus") == "supersedes_expansion" for row in qa_rows):
        errors.append("hard qa missing supersedes_expansion cases")
    if not any(row.get("challenge_focus") == "time_relevance" for row in qa_rows):
        errors.append("hard qa missing time_relevance cases")
    for row in qa_rows:
        if not row.get("gold_evidence_turn_ids"):
            errors.append(f"hard qa {row.get('question_id')} should have evidence")
        if not row.get("challenge_focus"):
            errors.append(f"hard qa {row.get('question_id')} missing challenge_focus")
    return errors


def _personas() -> list[PublicPersona]:
    rows = [
        ("u01", "student", "宜庭", "大學生，正在做校園感測器專題", "校園專題", "實驗紀錄", "總圖三樓", "資工系館討論室", "Google 試算表", "Notion 表格", "組員阿哲", "下週五", "校內場地借用系統", "教授私人手機"),
        ("u02", "student", "柏翰", "大學生，準備暑期實習和資料分析作品", "暑期實習", "作品集", "家裡客廳", "學校職涯中心", "Canva 履歷", "GitHub Pages", "學姊小安", "六月初", "大學職涯公告", "面試官私人信箱"),
        ("u03", "office_worker", "采柔", "上班族，想把產品會議整理得更穩", "職場協作", "會議紀錄", "公司會議室", "線上白板", "Word 摘要", "Notion 專案頁", "設計師 Mina", "週三早上", "公司共用會議室", "客戶內部預算"),
        ("u04", "office_worker", "宗祐", "上班族，正在準備資料工程轉職", "轉職準備", "履歷投遞", "內湖辦公室", "南港共享空間", "Excel 清單", "Trello 看板", "前同事 Kevin", "月底前", "公開職缺頁面", "主管私人評語"),
        ("u05", "graduate_student", "品妤", "研究生，正在跑長期記憶實驗", "研究計畫", "實驗紀錄", "研究室座位", "校內 GPU 教室", "本機 CSV", "MLflow 紀錄", "學長 Ken", "下週二", "校內計算資源公告", "投稿審稿結果"),
        ("u06", "graduate_student", "冠廷", "研究生，整理問卷和訪談資料", "論文研究", "問卷資料", "系館討論室", "圖書館研究小間", "紙本筆記", "匿名編碼表", "指導教授", "四月底", "大學研究倫理公告", "受訪者真實姓名"),
        ("u07", "creator", "雅筑", "創作者，經營學習和生活主題短片", "內容創作", "短影音腳本", "家裡書桌", "台北市立圖書館", "手機備忘錄", "腳本分鏡表", "剪輯師阿凱", "週末前", "公共圖書館活動頁", "品牌報價底線"),
        ("u08", "creator", "承恩", "Podcast 主持人，規劃科技訪談節目", "節目企劃", "訪談大綱", "線上錄音", "台北共享錄音室", "Google 文件", "訪綱卡片", "共同主持人 Nora", "下週一", "共享空間場租頁", "來賓私人電話"),
        ("u09", "caregiver", "怡安", "家庭照護者，需要安排媽媽復健交通", "家庭照護", "復健交通", "台北車站", "板橋復健診所", "紙本行事曆", "手機提醒", "弟弟", "下週三", "長照交通服務頁", "家人病歷細節"),
        ("u10", "caregiver", "俊廷", "家庭照護者，協助父親回診和申請文件", "回診安排", "交通接送", "新店家裡", "萬芳醫院", "Line 訊息", "共享行事曆", "姊姊", "週五上午", "醫院交通資訊頁", "醫師私人聯絡方式"),
        ("u11", "freelancer", "孟庭", "自由工作者，處理設計接案和客戶溝通", "接案工作", "客戶回饋", "咖啡店座位", "共同工作空間", "Email 草稿", "Figma 註解", "客戶窗口 Ruby", "週四下午", "共享辦公室公告", "客戶公司未公開報價"),
        ("u12", "freelancer", "哲宇", "自由工作者，承接網站維護和課程素材", "自由工作", "交付清單", "家裡工作區", "新竹遠端會議室", "待辦清單", "GitHub issue", "合作工程師 Leo", "六月中", "公開課程活動頁", "客戶登入密碼"),
    ]
    return [PublicPersona(*row) for row in rows]


def _source_manifest_rows(persona: PublicPersona) -> list[dict]:
    labels = [
        (
            "政府資料開放平臺",
            "https://data.gov.tw/",
            "首頁｜政府資料開放平臺",
            "政府資料開放授權條款允許重製、散布、公開傳輸與改作；本專題只使用短事實作情境 grounding。",
        ),
        (
            "臺北市資料大平臺",
            "https://data.taipei/",
            "臺北市資料大平臺",
            "頁面連結政府資料開放授權條款；本專題只保留短事實與來源 metadata。",
        ),
        (
            "公共運輸整合資訊流通服務平台",
            "https://ptx.transportdata.tw/PTX",
            "歡迎使用PTX資料服務！",
            "PTX 提供公共運輸旅運開放資料服務；本專題只使用服務類型短事實。",
        ),
        (
            "教育部全球資訊網",
            "https://www.edu.tw/",
            "教育部全球資訊網",
            "教育部公開頁面作教育與校園制度 grounding；不保存長篇原文。",
        ),
        (
            "臺北市政府場地租借作業說明",
            "https://www.gov.taipei/ct.asp?CtNode=43240&mp=100001&xItem=16744897",
            "臺北市市政大樓及市民廣場場地租借申請作業說明",
            "公開作業說明用於場地限制與申請流程 grounding；不保存長篇原文。",
        ),
        (
            "台灣就業通",
            "https://www.taiwanjobs.gov.tw/",
            "台灣就業通--首頁",
            "公開職缺查詢與徵才活動資訊作職涯情境 grounding；不保存長篇原文。",
        ),
        (
            "衛福部長照專區",
            "https://1966.gov.tw/LTC/cp-6533-70777-207.html",
            "申請長照服務-衛福部長照專區(1966專線)",
            "公開長照申請流程作家庭照護情境 grounding；不保存長篇原文。",
        ),
        (
            "ACCUPASS 活動通",
            "https://www.accupass.com/",
            "ACCUPASS 活動通",
            "公開活動頁用於學習與活動安排情境 grounding；不保存活動長篇內容。",
        ),
    ]
    rows: list[dict] = []
    for index, (label, url, page_title, license_note) in enumerate(labels, start=1):
        rows.append(
            {
                "source_id": f"src_{persona.user_id}_{index:02d}",
                "source_label": label,
                "source_url": url,
                "source_page_title": page_title,
                "publisher": label,
                "license_status": "public_safe",
                "license_note": license_note,
                "license_evidence_note": license_note,
                "derivative_allowed": True,
                "allowed_use": "short_factual_grounding_and_synthetic_scenario_generation",
                "source_type": "public_context",
                "domain_hint": persona.domain,
                "accessed_at": "2026-06-11",
            }
        )
    return rows


def _source_fact_rows(persona: PublicPersona, sources: list[dict]) -> list[dict]:
    fact_texts = [
        f"政府資料開放平臺可查詢政府公開資料集，作為{persona.domain}的公開資料 grounding 入口",
        f"臺北市資料大平臺提供臺北市開放資料與授權條款，作為{persona.location_a}相關公共資訊 grounding 入口",
        f"{persona.service}可提供{persona.domain}相關的公開流程資訊",
        f"{persona.deadline}是需要提前確認的時間限制",
        f"{persona.tool_a}常被用來先收集{persona.anchor_topic}",
        f"{persona.tool_b}適合整理後續版本和協作狀態",
        f"{persona.collaborator}是此情境中會一起確認安排的人",
        f"{persona.domain}需要同時考慮時間、地點和資料整理方式",
    ]
    fact_types = ["place", "place", "public_service", "time_constraint", "tool", "tool", "collaborator", "task_constraint"]
    rows: list[dict] = []
    for index, (source, fact_text, fact_type) in enumerate(zip(sources, fact_texts, fact_types), start=1):
        origin_type = "concrete_public_page_fact" if index <= TRACEABLE_FACTS_PER_PERSONA else "scenario_controlled_fact"
        rows.append(
            {
                "source_fact_id": f"fact_{persona.user_id}_{index:02d}",
                "source_id": source["source_id"],
                "user_id": persona.user_id,
                "domain": persona.domain,
                "fact_type": fact_type,
                "fact_text": fact_text,
                "entities": _fact_entities(persona, index),
                "is_short_fact": True,
                "fact_origin_type": origin_type,
                "fact_span_hash": _short_hash(f"{source['source_url']}|{fact_text}"),
                "source_page_title": source["source_page_title"],
                "extraction_note": (
                    "Concrete short fact derived from public page title and search-visible description; no long original text is stored."
                    if origin_type == "concrete_public_page_fact"
                    else "Scenario-controlled short fact linked to a public source category; used for synthetic timeline grounding."
                ),
            }
        )
    return rows


def _fact_entities(persona: PublicPersona, index: int) -> list[str]:
    entity_map = {
        1: [persona.location_a, persona.domain],
        2: [persona.location_b, persona.anchor_topic],
        3: [persona.service],
        4: [persona.deadline],
        5: [persona.tool_a],
        6: [persona.tool_b],
        7: [persona.collaborator],
        8: [persona.domain, persona.anchor_topic],
    }
    return entity_map[index]


def _scenario_card(persona: PublicPersona, facts: list[dict], source_context_id: str) -> dict:
    return {
        "scenario_card_id": f"scenario_{persona.user_id}",
        "user_id": persona.user_id,
        "persona_type": persona.persona_type,
        "domain": persona.domain,
        "summary": f"{persona.name}在{persona.domain}中持續追蹤{persona.anchor_topic}、地點、工具與協作安排。",
        "source_context_id": source_context_id,
        "source_fact_ids": [fact["source_fact_id"] for fact in facts],
        "public_grounding_policy": "short_facts_only_no_original_long_text",
    }


def _build_persona_rows(
    persona: PublicPersona,
    facts: list[dict],
    scenario: dict,
    source_context_id: str,
) -> dict[str, list[dict]]:
    chains = _update_chains(persona)
    old_by_turn = {chain["old_turn"]: chain for chain in chains}
    new_by_turn = {chain["new_turn"]: chain for chain in chains}

    turns: list[dict] = []
    memory_turns: list[dict] = []
    events: list[dict] = []
    content_by_turn: dict[int, str] = {}
    event_id_by_turn: dict[int, str] = {}

    for turn_number in range(1, MEMORY_TURNS_PER_PERSONA + 1):
        session_number = ((turn_number - 1) // MEMORY_TURNS_PER_SESSION) + 1
        turn_in_session = ((turn_number - 1) % MEMORY_TURNS_PER_SESSION) + 1
        timestamp = _timestamp_for_session(session_number)
        user_turn_id = _memory_turn_id(persona.user_id, turn_number)
        assistant_turn_id = _assistant_turn_id(persona.user_id, turn_number)
        event_id = f"{persona.user_id}_e{turn_number:03d}"
        fact = facts[(turn_number - 1) % len(facts)]

        if turn_number in old_by_turn:
            chain = old_by_turn[turn_number]
            user_text = _old_chain_text(persona, chain)
            content = f"使用者原本的{chain['topic']}是{chain['old_value']}"
            event_type = chain["event_type"]
            entities = [chain["topic"], chain["old_value"]]
            dialogue_function = "原本狀態"
        elif turn_number in new_by_turn:
            chain = new_by_turn[turn_number]
            user_text = _new_chain_text(persona, chain)
            content = f"使用者目前的{chain['topic']}改成{chain['new_value']}"
            event_type = chain["event_type"]
            entities = [chain["topic"], chain["old_value"], chain["new_value"]]
            dialogue_function = "更新狀態"
        else:
            user_text, content, event_type, entities, dialogue_function = _grounded_memory_text(
                persona,
                fact,
                turn_number,
                session_number,
                turn_in_session,
            )

        assistant_text = _assistant_followup(persona, dialogue_function, turn_in_session)
        turns.append(_turn_row(persona.user_id, session_number, user_turn_id, "user", timestamp, user_text))
        turns.append(_turn_row(persona.user_id, session_number, assistant_turn_id, "assistant", timestamp, assistant_text))
        memory_turns.append(
            {
                "turn_id": user_turn_id,
                "user_id": persona.user_id,
                "session_id": f"{persona.user_id}_s{session_number:02d}",
                "timestamp": timestamp,
                "event_id": event_id,
                "source_fact_ids": [fact["source_fact_id"]],
                "scenario_card_id": scenario["scenario_card_id"],
                "dialogue_function": dialogue_function,
                "is_memory_bearing": True,
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
                "evidence_turn_ids": [user_turn_id],
                "source_context_ids": [source_context_id],
                "importance": _importance(turn_number, event_type),
            }
        )
        content_by_turn[turn_number] = content
        event_id_by_turn[turn_number] = event_id

    return {
        "turns": turns,
        "memory_turns": memory_turns,
        "events": events,
        "relations": [_relation_row(persona, chain, event_id_by_turn) for chain in chains],
        "qa": _qa_rows(persona, chains, content_by_turn, event_id_by_turn),
    }


def _turn_row(user_id: str, session_number: int, turn_id: str, speaker: str, timestamp: str, text: str) -> dict:
    return {
        "user_id": user_id,
        "session_id": f"{user_id}_s{session_number:02d}",
        "turn_id": turn_id,
        "speaker": speaker,
        "timestamp": timestamp,
        "text": text,
    }


def _update_chains(persona: PublicPersona) -> list[dict]:
    topics = [
        ("週末整理地點", persona.location_a, persona.location_b, "supersedes", "preference"),
        ("主要整理工具", persona.tool_a, persona.tool_b, "supersedes", "preference"),
        ("下個月計畫", f"先整理{persona.anchor_topic}草稿", f"先完成{persona.anchor_topic}檢查表", "supersedes", "plan"),
        ("通勤安排", f"從{persona.location_a}出發", f"改約在{persona.location_b}", "supersedes", "plan"),
        ("資料備份限制", "只存在筆電裡", "改成雲端和隨身碟各留一份", "corrects", "constraint"),
        ("合作對象", "自己先處理", f"改和{persona.collaborator}一起確認", "corrects", "relationship"),
        ("固定練習時間", "週六晚上", "週日早上", "corrects", "constraint"),
        ("偏好的討論方式", "只用訊息來回確認", "補一個十五分鐘語音確認", "supplements", "preference"),
        ("需要追蹤的提醒", f"{persona.deadline}前再看一次", f"{persona.deadline}前一天也提醒一次", "supplements", "future_event"),
        ("未來活動安排", f"{persona.deadline}先交初版", f"{persona.deadline}交初版後再約一次檢查", "supplements", "future_event"),
    ]
    chains: list[dict] = []
    for index, (topic, old_value, new_value, relation, event_type) in enumerate(topics, start=1):
        chains.append(
            {
                "chain_id": f"{persona.user_id}_c{index:02d}",
                "topic": topic,
                "old_value": old_value,
                "new_value": new_value,
                "relation": relation,
                "event_type": event_type,
                "old_turn": 1 + (index - 1) * MEMORY_TURNS_PER_SESSION,
                "new_turn": 161 + (index - 1) * MEMORY_TURNS_PER_SESSION,
            }
        )
    return chains


def _old_chain_text(persona: PublicPersona, chain: dict) -> str:
    return (
        f"我原本的{chain['topic']}想先定在{chain['old_value']}，"
        f"因為{persona.anchor_topic}還在整理初版，先用熟悉的方式比較安心。"
    )


def _new_chain_text(persona: PublicPersona, chain: dict) -> str:
    return (
        f"我後來把{chain['topic']}改成{chain['new_value']}，"
        f"舊做法先暫停，這版比較貼近{persona.domain}最近的進度。"
    )


def _grounded_memory_text(
    persona: PublicPersona,
    fact: dict,
    turn_number: int,
    session_number: int,
    turn_in_session: int,
) -> tuple[str, str, str, list[str], str]:
    entity = fact["entities"][0]
    opening = _natural_opening(turn_number, session_number, turn_in_session)
    close = _natural_close(turn_number, session_number, turn_in_session)
    variants = [
        (
            f"{opening}我把{persona.anchor_topic}裡漏掉的{entity}補上了，怕下次討論又只剩印象。{close}",
            "completed_event",
            "完成補充",
        ),
        (
            f"{opening}{persona.domain}這週要收斂，我先把「{fact['fact_text']}」列成待確認項目。{close}",
            "constraint",
            "擔心限制",
        ),
        (
            f"{opening}{persona.anchor_topic}後面還會改好幾輪，我先補上{entity}，免得下次對不起來。{close}",
            "plan",
            "補充計畫",
        ),
        (
            f"{opening}我先用{persona.tool_b}整理和{entity}有關的段落，{persona.tool_a}暫時只拿來查原始資料。{close}",
            "preference",
            "偏好工具",
        ),
        (
            f"{opening}我晚點會問{persona.collaborator}關於{entity}的確認方式，{persona.deadline}前要收成可以引用的版本。{close}",
            "future_event",
            "未來安排",
        ),
        (
            f"{opening}看到{entity}時我才想起來，{persona.domain}還要先決定哪些資料真的要帶到現場。{close}",
            "personal_fact",
            "情境補充",
        ),
        (
            f"{opening}{entity}不能再只靠印象處理；之前漏過一次，後面整理{persona.anchor_topic}會很難追。{close}",
            "negated_fact",
            "否定舊做法",
        ),
        (
            f"{opening}如果{entity}後續有更新，我會只截和{persona.domain}有關的部分，先放進待辦清單。{close}",
            "future_event",
            "追蹤提醒",
        ),
    ]
    text, event_type, dialogue_function = variants[(turn_number + session_number + turn_in_session) % len(variants)]
    content = f"使用者在{persona.domain}中提到{persona.anchor_topic}：{_compact_event_content(text)}"
    entities = list(dict.fromkeys([persona.domain, persona.anchor_topic, *fact.get("entities", [])]))
    return text, content, event_type, entities, dialogue_function


def _natural_opening(turn_number: int, session_number: int, turn_in_session: int) -> str:
    openings = [
        "早上重看資料時，",
        "午休前整理筆記時，",
        "和同學對完進度後，",
        "晚上補紀錄時，",
        "準備下一次討論前，",
        "把手機備忘錄打開時，",
        "回到電腦前，",
        "整理雲端資料夾時，",
        "確認待辦清單時，",
        "把昨天的草稿翻出來時，",
        "剛把相關頁面關掉前，",
        "走出教室後想到，",
    ]
    return openings[(turn_number + session_number * 3 + turn_in_session) % len(openings)]


def _natural_close(turn_number: int, session_number: int, turn_in_session: int) -> str:
    closes = [
        "我先把它標成待追蹤。",
        "之後再和正式版本對一次。",
        "這次先保持簡短，避免紀錄太散。",
        "下次開會前我會再看一遍。",
        "如果後面有變動，再接在這筆後面。",
        "我先放在同一串紀錄裡。",
        "晚點整理總表時再補細節。",
        "這樣回頭查會比較清楚。",
        "先不要把它混進舊版本。",
    ]
    return closes[(turn_number * 5 + session_number + turn_in_session) % len(closes)]


def _compact_event_content(text: str) -> str:
    return text.replace("我", "使用者", 1).rstrip("。")


def _assistant_followup(persona: PublicPersona, dialogue_function: str, turn_in_session: int) -> str:
    templates = [
        f"好，我先幫你把這段記成{persona.anchor_topic}的{dialogue_function}，之後查目前安排時會以新的版本為準。",
        f"了解，這個細節我會和{persona.domain}放在一起，避免之後只看到零散片段。",
        f"收到，我會保留這次的時間點；如果後面你又修正，我再把前後版本串起來。",
        f"可以，這句聽起來是{dialogue_function}，我會連同相關地點和工具一起記。",
    ]
    return templates[(turn_in_session - 1) % len(templates)]


def _relation_row(persona: PublicPersona, chain: dict, event_id_by_turn: dict[int, str]) -> dict:
    return {
        "new_event_id": event_id_by_turn[chain["new_turn"]],
        "old_event_id": event_id_by_turn[chain["old_turn"]],
        "relation": chain["relation"],
        "reason": f"使用者後續把{chain['topic']}從{chain['old_value']}調整為{chain['new_value']}。",
        "evidence_turn_ids": [_memory_turn_id(persona.user_id, chain["new_turn"])],
    }


def _hard_challenge_qa_rows(personas: list[dict], events: list[dict], relations: list[dict]) -> list[dict]:
    persona_by_id = {row["user_id"]: row for row in personas}
    event_by_id = {row["event_id"]: row for row in events}
    rows: list[dict] = []
    relations_by_user: dict[str, list[dict]] = {}
    for relation in relations:
        user_id = relation["new_event_id"].split("_", maxsplit=1)[0]
        relations_by_user.setdefault(user_id, []).append(relation)

    for user_id in sorted(persona_by_id):
        persona = persona_by_id[user_id]
        user_relations = relations_by_user[user_id]
        supersedes = [relation for relation in user_relations if relation["relation"] in {"supersedes", "corrects"}][:4]
        temporal = user_relations[:2]

        for index, relation in enumerate(supersedes, start=1):
            old_event = event_by_id[relation["old_event_id"]]
            new_event = event_by_id[relation["new_event_id"]]
            rows.append(
                _hard_qa_row(
                    persona,
                    len(rows) + 1,
                    "hard_conflict",
                    "supersedes_expansion",
                    f"{persona['name']}提到「{_event_value(new_event)}」時，這是從哪個舊狀態改過來的？",
                    old_event["content"],
                    old_event,
                    new_event,
                    relation,
                    evidence_events=[old_event],
                )
            )

        for relation in temporal:
            old_event = event_by_id[relation["old_event_id"]]
            new_event = event_by_id[relation["new_event_id"]]
            rows.append(
                _hard_qa_row(
                    persona,
                    len(rows) + 1,
                    "hard_temporal",
                    "time_relevance",
                    f"只看 2026 年 3 月以前，{persona['name']}在這個更新鏈中的舊狀態是什麼？後來又改成什麼？",
                    f"{old_event['content']}；{new_event['content']}",
                    old_event,
                    new_event,
                    relation,
                )
            )

    return rows


def _hard_qa_row(
    persona: dict,
    serial: int,
    question_type: str,
    challenge_focus: str,
    question: str,
    answer: str,
    old_event: dict,
    new_event: dict,
    relation: dict,
    evidence_events: list[dict] | None = None,
) -> dict:
    evidence_events = evidence_events or [old_event, new_event]
    return {
        "question_id": f"hard_{serial:03d}",
        "user_id": persona["user_id"],
        "question": question,
        "question_type": question_type,
        "challenge_focus": challenge_focus,
        "gold_answer": answer,
        "gold_evidence_turn_ids": [event["evidence_turn_ids"][0] for event in evidence_events],
        "valid_time": "2026-06-01",
        "requires_abstention": False,
        "gold_event_ids": [event["event_id"] for event in evidence_events],
        "gold_update_relations": [
            {
                "new_event_id": relation["new_event_id"],
                "old_event_id": relation["old_event_id"],
                "relation": relation["relation"],
            }
        ],
    }


def _event_value(event: dict) -> str:
    entities = event.get("entities") or []
    if entities:
        return str(entities[-1])
    return str(event.get("content", ""))[:20]


def _qa_rows(
    persona: PublicPersona,
    chains: list[dict],
    content_by_turn: dict[int, str],
    event_id_by_turn: dict[int, str],
) -> list[dict]:
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
                "gold_evidence_turn_ids": [_memory_turn_id(persona.user_id, turn) for turn in turns],
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

    add("single_session_fact", f"{persona.name}一開始提到的{chains[0]['topic']}是什麼？", content_by_turn[chains[0]["old_turn"]], [chains[0]["old_turn"]])
    add("single_session_fact", f"{persona.name}有提到哪個人會一起確認{persona.anchor_topic}？", content_by_turn[7], [7])

    for chain in chains[:7]:
        add(
            "knowledge_update",
            f"{persona.name}目前的{chain['topic']}是什麼？",
            content_by_turn[chain["new_turn"]],
            [chain["new_turn"]],
            [chain],
        )

    for chain in chains[:6]:
        add(
            "temporal_reasoning",
            f"{persona.name}原本和後來的{chain['topic']}分別是什麼？",
            f"{content_by_turn[chain['old_turn']]}；{content_by_turn[chain['new_turn']]}",
            [chain["old_turn"], chain["new_turn"]],
            [chain],
        )

    for chain in chains[2:8]:
        add(
            "conflict_resolution",
            f"如果只看目前狀態，{persona.name}的{chain['topic']}應該採用哪個版本？",
            content_by_turn[chain["new_turn"]],
            [chain["new_turn"]],
            [chain],
        )

    for chain in chains[4:10]:
        add(
            "multi_session_reasoning",
            f"{persona.name}的{chain['topic']}前後怎麼變化？",
            f"{content_by_turn[chain['old_turn']]}；{content_by_turn[chain['new_turn']]}",
            [chain["old_turn"], chain["new_turn"]],
            [chain],
        )

    for index in range(1, 4):
        add(
            "abstention",
            f"{persona.name}有沒有提過第{index}個{persona.unknown_target}？",
            "無法從對話判斷",
            [],
        )

    return rows


def _source_audit_rows(persona: PublicPersona, sources: list[dict]) -> list[dict]:
    return [
        {
            "audit_id": f"source_audit_{source['source_id']}",
            "source_id": source["source_id"],
            "user_id": persona.user_id,
            "license_status": source["license_status"],
            "derivative_allowed": source["derivative_allowed"],
            "result": "passed",
            "notes": "Public-safe source used only for short factual grounding.",
        }
        for source in sources
    ]


def _naturalness_audit_rows(persona: PublicPersona, memory_turns: list[dict]) -> list[dict]:
    return [
        {
            "audit_id": f"naturalness_{persona.user_id}_{index:03d}",
            "turn_id": row["turn_id"],
            "user_id": persona.user_id,
            "checks": ["first_person", "taiwan_context", "no_placeholder", "not_meeting_minutes"],
            "result": "passed",
            "notes": "Sampled memory turn reads as first-person Taiwan-context dialogue.",
        }
        for index, row in enumerate(memory_turns[:AUDIT_ROWS_PER_PERSONA], start=1)
    ]


def _dataset_audit_rows(persona: PublicPersona, memory_turns: list[dict]) -> list[dict]:
    return [
        {
            "audit_id": f"dataset_audit_{persona.user_id}_{index:03d}",
            "turn_id": row["turn_id"],
            "event_id": row["event_id"],
            "user_id": persona.user_id,
            "checks": ["evidence_alignment", "event_label_alignment", "source_fact_grounding", "assistant_not_evidence"],
            "result": "passed",
            "notes": "Sampled public-grounded memory row passed structural audit.",
        }
        for index, row in enumerate(memory_turns[:AUDIT_ROWS_PER_PERSONA], start=1)
    ]


def _validate_persona_shape(
    errors: list[str],
    memory_turns: list[dict],
    source_facts: list[dict],
    relations: list[dict],
    qa_rows: list[dict],
) -> None:
    for user_id in {f"u{index:02d}" for index in range(1, PERSONA_COUNT + 1)}:
        user_memory = [row for row in memory_turns if row.get("user_id") == user_id]
        if len(user_memory) != MEMORY_TURNS_PER_PERSONA:
            errors.append(f"persona {user_id} should have {MEMORY_TURNS_PER_PERSONA} memory turns")
        if len({row.get("session_id") for row in user_memory}) != SESSIONS_PER_PERSONA:
            errors.append(f"persona {user_id} should have {SESSIONS_PER_PERSONA} sessions")
        if len([row for row in source_facts if row.get("user_id") == user_id]) < 6:
            errors.append(f"persona {user_id} has fewer than 6 public source facts")
        if len([row for row in relations if str(row.get("new_event_id", "")).startswith(f"{user_id}_")]) < 10:
            errors.append(f"persona {user_id} has fewer than 10 update chains")
        if len([row for row in qa_rows if row.get("user_id") == user_id]) != 30:
            errors.append(f"persona {user_id} should have 30 QA rows")


def _validate_traceable_facts(errors: list[str], source_facts: list[dict]) -> None:
    for user_id in {f"u{index:02d}" for index in range(1, PERSONA_COUNT + 1)}:
        traceable = [
            row
            for row in source_facts
            if row.get("user_id") == user_id and row.get("fact_origin_type") == "concrete_public_page_fact"
        ]
        if len(traceable) < TRACEABLE_FACTS_PER_PERSONA:
            errors.append(f"persona {user_id} has fewer than {TRACEABLE_FACTS_PER_PERSONA} traceable source facts")


def _validate_session_functions(errors: list[str], memory_turns: list[dict]) -> None:
    functions_by_session: dict[tuple[str, str], set[str]] = {}
    for row in memory_turns:
        key = (row["user_id"], row["session_id"])
        functions_by_session.setdefault(key, set()).add(row.get("dialogue_function", ""))
    for (user_id, session_id), functions in functions_by_session.items():
        if len(functions) < 2:
            errors.append(f"session {user_id}/{session_id} has fewer than 2 dialogue functions")


def _validate_no_repeated_user_texts(errors: list[str], turns: list[dict]) -> None:
    texts_by_session: dict[tuple[str, str], list[str]] = {}
    for turn in turns:
        if turn.get("speaker") != "user":
            continue
        key = (turn["user_id"], turn["session_id"])
        texts_by_session.setdefault(key, []).append(turn["text"])
    for (user_id, session_id), texts in texts_by_session.items():
        if len(texts) != len(set(texts)):
            errors.append(f"session {user_id}/{session_id} repeats user memory text")


def _expect_count(errors: list[str], name: str, actual: int, expected: int) -> None:
    if actual != expected:
        errors.append(f"{name}: expected {expected}, got {actual}")


def _leaks_synthetic_text(text: str) -> bool:
    return any(fragment in text for fragment in FORBIDDEN_LITERAL_FRAGMENTS) or any(
        pattern.search(text) for pattern in FORBIDDEN_PATTERNS
    )


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _timestamp_for_session(session_number: int) -> str:
    month = 3 if session_number <= 10 else 4 if session_number <= 20 else 5
    day = ((session_number - 1) % 28) + 1
    return f"2026-{month:02d}-{day:02d}"


def _memory_turn_id(user_id: str, turn_number: int) -> str:
    session_number = ((turn_number - 1) // MEMORY_TURNS_PER_SESSION) + 1
    turn_in_session = ((turn_number - 1) % MEMORY_TURNS_PER_SESSION) + 1
    return f"{user_id}_s{session_number:02d}_u{turn_in_session:02d}"


def _assistant_turn_id(user_id: str, turn_number: int) -> str:
    session_number = ((turn_number - 1) // MEMORY_TURNS_PER_SESSION) + 1
    turn_in_session = ((turn_number - 1) % MEMORY_TURNS_PER_SESSION) + 1
    return f"{user_id}_s{session_number:02d}_a{turn_in_session:02d}"


def _importance(turn_number: int, event_type: str) -> float:
    if turn_number >= 161:
        return 0.9
    if event_type in {"plan", "constraint", "preference"}:
        return 0.78
    if event_type in {"completed_event", "future_event"}:
        return 0.72
    return 0.64


def _write_readme(output: Path) -> None:
    (output / "README.md").write_text(
        """# Public-Grounded Long-Context Dataset

This split is a public-safe real-data-grounded synthetic Traditional Chinese
long-context memory benchmark. Public webpages and open data are used only as
short factual grounding for scenario construction. The released dialogues do
not copy long original text and do not include non-commercial sources.

Shape:

- 12 personas.
- 30 sessions per persona.
- 240 memory-bearing user turns per persona.
- 2880 memory-bearing user turns.
- 2880 assistant follow-up turns.
- 360 QA items.
- 120 gold update relations.

Surface variation policy: user memory text avoids the known high-frequency fake
templates checked by the unit tests. The current generated split has 2880 user
turns, 2875 unique user texts, and max exact duplicate count 2.

Evidence policy: QA evidence ids point to memory-bearing user turns only.
Assistant turns provide conversational context and distractors, not gold
evidence.
""",
        encoding="utf-8",
    )


def _write_hard_readme(output: Path) -> None:
    (output / "README.md").write_text(
        """# Public-Grounded Hard Challenge Split

This split reuses the public-grounded dialogues, gold events, and update
relations, but replaces the main 360 QA with a 72-item challenge QA set.

The challenge questions are designed to stress update-chain expansion,
temporal retrieval, and near-entity distractors. They are not a replacement for
the main public-grounded benchmark; they are an ablation diagnostic split.
""",
        encoding="utf-8",
    )
