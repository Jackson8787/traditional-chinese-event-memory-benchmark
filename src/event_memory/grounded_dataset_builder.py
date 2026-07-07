from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset_builder import validate_dataset
from .io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class GroundedPersona:
    user_id: str
    persona_type: str
    name: str
    profile: str
    initial_plan: str
    updated_plan: str
    old_location: str
    new_location: str
    stable_preference: str
    old_constraint: str
    new_constraint: str
    relationship: str
    completed_event: str
    future_event: str
    old_tool: str
    new_tool: str
    unknown_target: str
    source_domain: str
    adapted_source: str


def build_grounded_dataset(output_dir: str | Path) -> dict[str, int]:
    output = Path(output_dir)
    personas = _grounded_personas()
    source_rows = _source_context_rows(personas)
    persona_rows: list[dict] = []
    turn_rows: list[dict] = []
    event_rows: list[dict] = []
    relation_rows: list[dict] = []
    qa_rows: list[dict] = []

    for index, persona in enumerate(personas, start=1):
        source_context_ids = _persona_source_context_ids(persona)
        persona_rows.append(
            {
                "user_id": persona.user_id,
                "persona_type": persona.persona_type,
                "name": persona.name,
                "profile": persona.profile,
                "source_context_ids": source_context_ids,
            }
        )
        rows = _build_grounded_user_rows(index, persona, source_context_ids)
        turn_rows.extend(rows["turns"])
        event_rows.extend(rows["events"])
        relation_rows.extend(rows["relations"])
        qa_rows.extend(rows["qa"])

    write_jsonl(output / "source_contexts.jsonl", source_rows)
    write_jsonl(output / "personas.jsonl", persona_rows)
    write_jsonl(output / "dialogues.jsonl", turn_rows)
    write_jsonl(output / "gold_events.jsonl", event_rows)
    write_jsonl(output / "gold_update_relations.jsonl", relation_rows)
    write_jsonl(output / "qa.jsonl", qa_rows)
    return {
        "source_contexts": len(source_rows),
        "personas": len(persona_rows),
        "dialogue_turns": len(turn_rows),
        "gold_events": len(event_rows),
        "gold_update_relations": len(relation_rows),
        "qa": len(qa_rows),
    }


def validate_grounded_dataset(output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    errors = validate_dataset(output)
    source_contexts = read_jsonl(output / "source_contexts.jsonl")
    personas = read_jsonl(output / "personas.jsonl")
    events = read_jsonl(output / "gold_events.jsonl")
    source_ids = {row["source_context_id"] for row in source_contexts}

    if not source_contexts:
        errors.append("source_contexts.jsonl is empty")

    for persona in personas:
        references = persona.get("source_context_ids", [])
        if not references:
            errors.append(f"persona {persona['user_id']} has no source_context_ids")
        for source_context_id in references:
            if source_context_id not in source_ids:
                errors.append(f"persona {persona['user_id']} references missing source_context {source_context_id}")

    for event in events:
        references = event.get("source_context_ids", [])
        if not references:
            errors.append(f"event {event['event_id']} has no source_context_ids")
        for source_context_id in references:
            if source_context_id not in source_ids:
                errors.append(f"event {event['event_id']} references missing source_context {source_context_id}")

    return errors


def _grounded_personas() -> list[GroundedPersona]:
    return [
        GroundedPersona("u01", "student", "林亦辰", "新竹資工大學生，正在用公開實習公告規劃研究所與暑期實習取捨", "準備研究所考試", "尋找暑期實習", "台北", "新竹或遠端", "偏好 NLP 題目", "每週三晚上要社團開會", "每週三晚上改成留給專題討論", "和室友阿哲一起租屋", "完成履歷初稿", "五月底前投三間實習", "Notion", "Google 試算表", "Google 實習錄取結果", "university_internship", "NaturalConv"),
        GroundedPersona("u02", "student", "陳品妤", "商管學生，參考校內問卷與公開課程資料做資料分析專題", "準備交換學生申請", "改做校內資料分析專題", "台中", "台北", "偏好視覺化報告", "週五下午要打工", "週五下午改成可排小組討論", "和學姊小安討論專題", "完成英文自傳", "六月前做完問卷", "Excel", "Power BI", "交換學校錄取名單", "university_project", "LCCC"),
        GroundedPersona("u03", "office_worker", "張承翰", "軟體工程師，根據公開職缺描述規劃轉資料工程", "準備後端工程師面試", "改準備資料工程職缺", "內湖", "南港或遠端", "偏好 Python 工具鏈", "週二晚上固定上英文課", "週二晚上改成保留給線上面試", "和主管 Kelly 每週一同步", "完成作品集整理", "月底前投五個職缺", "Java", "Python", "特定公司面試結果", "job_transition", "NaturalConv"),
        GroundedPersona("u04", "office_worker", "吳佳蓉", "產品經理，參考公開會議工具與企業流程文章改善紀錄流程", "規劃轉到資料產品團隊", "改留在原團隊推 AI 會議摘要", "松山", "信義或遠端", "偏好簡潔摘要", "週四早上有部門例會", "週四早上改成跨部門測試時段", "和設計師 Mina 合作", "完成需求訪談", "下週整理 MVP 規格", "Confluence", "Notion", "客戶正式預算", "work_meeting", "LCCC"),
        GroundedPersona("u05", "graduate_student", "黃柏宇", "資工碩一，研究長期記憶與 RAG，參考公開論文與實驗 repo", "準備投稿 workshop", "改先完成期末專題實驗", "台北", "新竹", "偏好可重現實驗", "週三下午要 lab meeting", "週三下午改成跑實驗檢查點", "和學長 Ken 共用 GPU", "完成 baseline 初版", "六月前跑完 ablation", "FAISS", "Chroma", "論文接受結果", "research_project", "KdConv"),
        GroundedPersona("u06", "graduate_student", "蔡宜庭", "教育所研究生，參考公開教育問卷與學習平台研究設計問卷", "準備質性訪談", "改做問卷分析", "台南", "高雄", "偏好匿名資料", "週一晚上要帶討論課", "週一晚上改成整理問卷回收", "和指導教授約每兩週 meeting", "完成訪談大綱", "月底前收 80 份問卷", "逐字稿", "匿名問卷", "受訪者真實姓名", "education_survey", "NaturalConv"),
        GroundedPersona("u07", "creator", "周子晴", "影音創作者，參考公開求職文章與校園活動素材規劃內容", "準備拍研究所備考系列", "改拍實習求職系列", "台北", "新竹或遠端", "偏好短影音腳本", "週末要剪片", "週末改成集中拍攝素材", "和剪輯師阿凱合作", "完成三集大綱", "下月發布第一支影片", "長影片", "短影音", "品牌業配價格", "creator_content", "LCCC"),
        GroundedPersona("u08", "creator", "郭宇恩", "Podcast 主持人，參考公開開源活動與訪談節目資料規劃來賓", "規劃訪談 AI 研究者", "改做台灣開源社群系列", "線上", "台北現場", "偏好深度訪談", "週三晚上錄音", "週三晚上改成剪輯與上架", "和共同主持人 Nora 分工", "完成來賓邀約信", "六月排定四位來賓", "Zoom", "現場錄音", "來賓私人電話", "open_source_podcast", "KdConv"),
        GroundedPersona("u09", "caregiver", "許雅雯", "家庭照護者，參考公開長照與交通服務資訊安排家人行程", "安排媽媽復健交通", "改申請長照接送服務", "台北車站", "板橋", "偏好晚間可處理的事項", "每週三晚上要照顧家人", "每週三晚上改由弟弟接手", "和弟弟輪流陪診", "完成長照諮詢", "下週確認接送時段", "計程車", "長照接送", "家人病歷細節", "care_transport", "NaturalConv"),
        GroundedPersona("u10", "caregiver", "羅明哲", "遠距工作者，參考公開醫院交通與復康巴士資訊協助父親回診", "自己開車載父親回診", "改搭復康巴士", "新店", "萬芳", "偏好手機提醒", "週五上午要陪診", "週五上午改請家人陪診", "和家人分攤行政文件", "完成復康巴士註冊", "月底前設定回診提醒", "紙本行事曆", "手機提醒", "醫師私人聯絡方式", "care_admin", "LCCC"),
    ]


def _source_context_rows(personas: list[GroundedPersona]) -> list[dict]:
    rows: list[dict] = []
    for persona in personas:
        public_id, adapted_id = _persona_source_context_ids(persona)
        rows.append(
            {
                "source_context_id": public_id,
                "source_type": "public_context",
                "domain": persona.source_domain,
                "source_label": f"Taiwan public context notes for {persona.source_domain}",
                "license_note": "Public webpages or open-data style context used for scenario grounding only; no long text copied.",
                "entities": [persona.old_location, persona.new_location, persona.initial_plan, persona.updated_plan],
                "notes": "Use as Taiwan-context vocabulary and scenario constraints for synthetic long-term memory events.",
            }
        )
        rows.append(
            {
                "source_context_id": adapted_id,
                "source_type": "simplified_chinese_adaptation",
                "domain": persona.source_domain,
                "source_label": f"{persona.adapted_source} adapted dialogue style",
                "license_note": "Simplified Chinese public dialogue source used only as adaptation reference after terms review.",
                "entities": [persona.old_tool, persona.new_tool, persona.stable_preference],
                "notes": "Convert to Traditional Chinese, localize vocabulary to Taiwan usage, and re-annotate all evidence labels.",
            }
        )
    return rows


def _persona_source_context_ids(persona: GroundedPersona) -> list[str]:
    return [f"ctx_{persona.user_id}_{persona.source_domain}", f"ctx_{persona.user_id}_{persona.adapted_source.lower()}"]


def _build_grounded_user_rows(index: int, persona: GroundedPersona, source_context_ids: list[str]) -> dict[str, list[dict]]:
    uid = persona.user_id
    event = lambda n: f"{uid}_e{n:02d}"
    turn = lambda n: f"{uid}_s{n:02d}_t01"
    session = lambda n: f"{uid}_s{n:02d}"
    updated_plan = _clean_update_phrase(persona.updated_plan)

    dates = [
        "2026-03-01",
        "2026-03-05",
        "2026-03-09",
        "2026-03-13",
        "2026-03-17",
        "2026-03-21",
        "2026-03-25",
        "2026-04-02",
        "2026-04-08",
        "2026-04-14",
        "2026-04-20",
        "2026-04-26",
        "2026-05-02",
        "2026-05-08",
    ]
    texts = [
        f"我三月初主要在{persona.initial_plan}，想先照原本節奏準備。",
        f"地點一開始比較想選{persona.old_location}，因為那邊資料比較好找。",
        f"我一直{persona.stable_preference}，之後如果換題目也想保留這個方向。",
        f"{persona.old_constraint}，那段時間先不要排太滿。",
        f"目前主要用{persona.old_tool}整理資料，先不要換工具。",
        f"{persona.relationship}，很多安排都會一起確認。",
        f"我已經{persona.completed_event}，可以進到下一步。",
        f"後來想清楚了，我不想繼續{persona.initial_plan}，改成{updated_plan}。",
        f"不是{persona.old_location}了，現在覺得{persona.new_location}比較合理。",
        f"{persona.new_constraint}，所以原本的固定限制要改一下。",
        f"工具也調整了，之後改用{persona.new_tool}，不用{persona.old_tool}當主要工具。",
        f"我還是{persona.stable_preference}，只是會用新的計畫去搭配。",
        f"接下來希望{persona.future_event}，這是下一個時間點。",
        f"如果之後要回顧，可以記得這些安排是參考台灣公開情境和轉繁後的對話語氣整理的。",
    ]
    turns = [
        {
            "user_id": uid,
            "session_id": session(i),
            "turn_id": turn(i),
            "speaker": "user",
            "timestamp": date,
            "text": text,
        }
        for i, (date, text) in enumerate(zip(dates, texts), start=1)
    ]

    events = [
        _event(event(1), uid, dates[0], "plan", f"使用者{persona.initial_plan}", [persona.initial_plan], turn(1), 0.82, source_context_ids),
        _event(event(2), uid, dates[1], "preference", f"使用者一開始偏好{persona.old_location}", [persona.old_location], turn(2), 0.68, source_context_ids),
        _event(event(3), uid, dates[2], "preference", f"使用者{persona.stable_preference}", [persona.stable_preference], turn(3), 0.74, source_context_ids),
        _event(event(4), uid, dates[3], "constraint", persona.old_constraint, [persona.old_constraint], turn(4), 0.72, source_context_ids),
        _event(event(5), uid, dates[4], "preference", f"使用者主要使用{persona.old_tool}", [persona.old_tool], turn(5), 0.66, source_context_ids),
        _event(event(6), uid, dates[5], "relationship", persona.relationship, [persona.relationship], turn(6), 0.64, source_context_ids),
        _event(event(7), uid, dates[6], "completed_event", f"使用者已經{persona.completed_event}", [persona.completed_event], turn(7), 0.7, source_context_ids),
        _event(event(8), uid, dates[7], "negated_fact", f"使用者不再{persona.initial_plan}", [persona.initial_plan], turn(8), 0.86, source_context_ids),
        _event(event(9), uid, dates[7], "plan", f"使用者改成{updated_plan}", [updated_plan], turn(8), 0.9, source_context_ids),
        _event(event(10), uid, dates[8], "preference", f"使用者改為偏好{persona.new_location}", [persona.old_location, persona.new_location], turn(9), 0.8, source_context_ids),
        _event(event(11), uid, dates[9], "constraint", persona.new_constraint, [persona.new_constraint], turn(10), 0.78, source_context_ids),
        _event(event(12), uid, dates[10], "preference", f"使用者改用{persona.new_tool}", [persona.old_tool, persona.new_tool], turn(11), 0.76, source_context_ids),
        _event(event(13), uid, dates[11], "preference", f"使用者仍然{persona.stable_preference}", [persona.stable_preference], turn(12), 0.7, source_context_ids),
        _event(event(14), uid, dates[12], "future_event", f"使用者希望{persona.future_event}", [persona.future_event], turn(13), 0.72, source_context_ids),
    ]
    relations = [
        _relation(event(8), event(1), "supersedes", "使用者明確取消原本計畫", turn(8)),
        _relation(event(9), event(1), "supersedes", "使用者以新計畫取代原本計畫", turn(8)),
        _relation(event(10), event(2), "corrects", "使用者修正原本地點偏好", turn(9)),
        _relation(event(11), event(4), "corrects", "使用者修正原本固定時間限制", turn(10)),
        _relation(event(12), event(5), "corrects", "使用者修正主要工具偏好", turn(11)),
        _relation(event(13), event(3), "supplements", "使用者再次確認穩定偏好並補充新計畫脈絡", turn(12)),
    ]
    qa = _qa_rows(index, persona, event, turn)
    return {"turns": turns, "events": events, "relations": relations, "qa": qa}


def _event(
    event_id: str,
    user_id: str,
    time: str,
    event_type: str,
    content: str,
    entities: list[str],
    evidence_turn_id: str,
    importance: float,
    source_context_ids: list[str],
) -> dict:
    return {
        "event_id": event_id,
        "user_id": user_id,
        "time": time,
        "speaker": "user",
        "subject": "使用者",
        "event_type": event_type,
        "content": content,
        "entities": entities,
        "evidence_turn_ids": [evidence_turn_id],
        "source_context_ids": source_context_ids,
        "importance": importance,
    }


def _relation(new_event_id: str, old_event_id: str, relation: str, reason: str, evidence_turn_id: str) -> dict:
    return {
        "new_event_id": new_event_id,
        "old_event_id": old_event_id,
        "relation": relation,
        "reason": reason,
        "evidence_turn_ids": [evidence_turn_id],
    }


def _qa_rows(index: int, persona: GroundedPersona, event, turn) -> list[dict]:
    uid = persona.user_id
    base = (index - 1) * 15
    updated_plan = _clean_update_phrase(persona.updated_plan)

    def q(n: int) -> str:
        return f"gq{base + n:03d}"

    return [
        _qa(q(1), uid, f"{persona.name} 和誰有固定合作或照護關係？", "single_session_fact", persona.relationship, [turn(6)], "current", False, [event(6)]),
        _qa(q(2), uid, f"{persona.name} 現在的主要計畫是什麼？", "knowledge_update", f"使用者改成{updated_plan}", [turn(8)], "current", False, [event(9)], _sup(event(9), event(1))),
        _qa(q(3), uid, f"{persona.name} 現在還在{persona.initial_plan}嗎？", "knowledge_update", f"不是，使用者改成{updated_plan}", [turn(8)], "current", False, [event(8), event(9)], _sup(event(8), event(1))),
        _qa(q(4), uid, f"{persona.name} 後來和 {persona.new_tool} 有關的工具安排是什麼？", "knowledge_update", f"使用者改用{persona.new_tool}", [turn(11)], "current", False, [event(12)], _correct(event(12), event(5))),
        _qa(q(5), uid, f"{persona.name} 三月初原本在做什麼？", "temporal_reasoning", f"使用者{persona.initial_plan}", [turn(1)], "2026-03", False, [event(1)]),
        _qa(q(6), uid, f"{persona.name} 三月原本偏好的地點是哪裡？", "temporal_reasoning", f"使用者一開始偏好{persona.old_location}", [turn(2)], "2026-03", False, [event(2)]),
        _qa(q(7), uid, f"{persona.name} 原本的固定限制是什麼？", "temporal_reasoning", persona.old_constraint, [turn(4)], "2026-03", False, [event(4)]),
        _qa(q(8), uid, f"{persona.name} 現在偏好的地點是哪裡？", "conflict_resolution", f"使用者改為偏好{persona.new_location}", [turn(9)], "current", False, [event(10)], _correct(event(10), event(2))),
        _qa(q(9), uid, f"{persona.name} 現在的固定時間限制是什麼？", "conflict_resolution", persona.new_constraint, [turn(10)], "current", False, [event(11)], _correct(event(11), event(4))),
        _qa(q(10), uid, f"{persona.name} 目前和 {persona.new_tool} 有關的主要工具安排是什麼？", "conflict_resolution", f"使用者改用{persona.new_tool}", [turn(11)], "current", False, [event(12)], _correct(event(12), event(5))),
        _qa(q(11), uid, f"{persona.name} 跨 session 仍維持的偏好是什麼？", "multi_session_reasoning", f"使用者{persona.stable_preference}", [turn(3), turn(12)], "current", False, [event(3), event(13)]),
        _qa(q(12), uid, f"{persona.name} 目前的計畫與地點安排是什麼？", "multi_session_reasoning", f"使用者改成{updated_plan}，地點偏好為{persona.new_location}", [turn(8), turn(9)], "current", False, [event(9), event(10)]),
        _qa(q(13), uid, f"{persona.name} 接下來希望做什麼，且要避開哪個限制？", "multi_session_reasoning", f"使用者希望{persona.future_event}，並且{persona.new_constraint}", [turn(10), turn(13)], "future", False, [event(11), event(14)]),
        _qa(q(14), uid, f"{persona.name} 有沒有提過 {persona.unknown_target}？", "abstention", "", [], "current", True, []),
        _qa(q(15), uid, f"{persona.name} 有沒有提供私人聯絡方式或機密資料？", "abstention", "", [], "current", True, []),
    ]


def _qa(
    question_id: str,
    user_id: str,
    question: str,
    question_type: str,
    gold_answer: str,
    evidence_turn_ids: list[str],
    valid_time: str,
    requires_abstention: bool,
    gold_event_ids: list[str],
    gold_update_relations: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "user_id": user_id,
        "question": question,
        "question_type": question_type,
        "gold_answer": gold_answer,
        "gold_evidence_turn_ids": evidence_turn_ids,
        "valid_time": valid_time,
        "requires_abstention": requires_abstention,
        "gold_event_ids": gold_event_ids,
        "gold_update_relations": gold_update_relations or [],
    }


def _sup(new_event_id: str, old_event_id: str) -> list[dict[str, str]]:
    return [{"new_event_id": new_event_id, "old_event_id": old_event_id, "relation": "supersedes"}]


def _correct(new_event_id: str, old_event_id: str) -> list[dict[str, str]]:
    return [{"new_event_id": new_event_id, "old_event_id": old_event_id, "relation": "corrects"}]


def _clean_update_phrase(text: str) -> str:
    for prefix in ("改成", "改為"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    if text.startswith("改"):
        return text[1:]
    return text
