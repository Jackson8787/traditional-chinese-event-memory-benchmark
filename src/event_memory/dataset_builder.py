from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .io import write_jsonl


PERSONA_TYPES = [
    ("student", "大學生"),
    ("office_worker", "上班族"),
    ("graduate_student", "研究生"),
    ("creator", "創作者"),
    ("caregiver", "家庭照護者"),
]


@dataclass(frozen=True)
class SeedPersona:
    user_id: str
    persona_type: str
    name: str
    profile: str
    initial_plan: str
    updated_plan: str
    old_location: str
    new_location: str
    stable_preference: str
    constraint: str
    relationship: str
    completed_event: str
    future_event: str
    unknown_target: str


def seed_personas() -> list[SeedPersona]:
    rows: list[SeedPersona] = []
    names = [
        ("u01", "大學生", "林亦辰", "交大資工大三，正在思考研究所與暑期實習的取捨", "準備研究所考試", "尋找暑期實習", "台北", "新竹或遠端", "偏好 NLP 題目", "每週三晚上要社團開會", "和室友阿哲一起租屋", "完成履歷初稿", "五月底前投三間實習", "Google 實習"),
        ("u02", "大學生", "陳品妤", "台北大學商管學生，想把資料分析放進畢業專題", "準備交換學生申請", "改做校內資料分析專題", "台中", "台北", "偏好視覺化報告", "週五下午要打工", "和學姊小安討論專題", "完成英文自傳", "六月前做完問卷", "日本交換學校"),
        ("u03", "上班族", "張承翰", "軟體工程師，正在規劃轉職與進修", "準備後端工程師面試", "改準備資料工程職缺", "內湖", "南港或遠端", "偏好 Python 工具鏈", "週二晚上固定上英文課", "和主管 Kelly 每週一同步", "完成作品集整理", "月底前投五個職缺", "特定公司名稱"),
        ("u04", "上班族", "吳佳蓉", "產品經理，想改善跨部門會議紀錄流程", "規劃轉到資料產品團隊", "改留在原團隊推 AI 會議摘要", "松山", "信義或遠端", "偏好簡潔摘要", "週四早上有部門例會", "和設計師 Mina 合作", "完成需求訪談", "下週整理 MVP 規格", "客戶正式預算"),
        ("u05", "研究生", "黃柏宇", "資工碩一，研究方向在長期記憶與 RAG", "準備投稿 workshop", "改先完成期末專題實驗", "台北", "新竹", "偏好可重現實驗", "週三下午要 lab meeting", "和學長 Ken 共用 GPU", "完成 baseline 初版", "六月前跑完 ablation", "論文接受結果"),
        ("u06", "研究生", "蔡宜庭", "教育所研究生，研究數位學習平台的回饋品質", "準備質性訪談", "改做問卷分析", "台南", "高雄", "偏好匿名資料", "週一晚上要帶討論課", "和指導教授約每兩週 meeting", "完成訪談大綱", "月底前收 80 份問卷", "受訪者真實姓名"),
        ("u07", "創作者", "周子晴", "YouTube 創作者，經營台灣生活與學習主題", "準備拍研究所備考系列", "改拍實習求職系列", "台北", "新竹或遠端", "偏好短影音腳本", "週末要剪片", "和剪輯師阿凱合作", "完成三集大綱", "下月發布第一支影片", "品牌業配價格"),
        ("u08", "創作者", "郭宇恩", "Podcast 主持人，節目主題是科技工作者訪談", "規劃訪談 AI 研究者", "改做台灣開源社群系列", "線上", "台北現場", "偏好深度訪談", "週三晚上錄音", "和共同主持人 Nora 分工", "完成來賓邀約信", "六月排定四位來賓", "來賓私人電話"),
        ("u09", "家庭照護者", "許雅雯", "白天工作，晚上照顧家人，常需要調整行程", "安排媽媽復健交通", "改申請長照接送服務", "台北車站", "板橋", "偏好晚間可處理的事項", "每週三晚上要照顧家人", "和弟弟輪流陪診", "完成長照諮詢", "下週確認接送時段", "家人病歷細節"),
        ("u10", "家庭照護者", "羅明哲", "遠距工作者，協助父親回診與家庭行政", "自己開車載父親回診", "改搭復康巴士", "新店", "萬芳", "偏好手機提醒", "週五上午要陪診", "和家人分攤行政文件", "完成復康巴士註冊", "月底前設定回診提醒", "醫師私人聯絡方式"),
    ]
    type_lookup = {label: key for key, label in PERSONA_TYPES}
    for row in names:
        rows.append(SeedPersona(row[0], type_lookup[row[1]], row[2], row[3], *row[4:]))
    return rows


def build_dataset(output_dir: str | Path) -> dict[str, int]:
    output = Path(output_dir)
    personas = seed_personas()
    persona_rows: list[dict] = []
    turn_rows: list[dict] = []
    event_rows: list[dict] = []
    relation_rows: list[dict] = []
    qa_rows: list[dict] = []

    for index, persona in enumerate(personas, start=1):
        persona_rows.append(
            {
                "user_id": persona.user_id,
                "persona_type": persona.persona_type,
                "name": persona.name,
                "profile": persona.profile,
            }
        )
        rows = _build_user_rows(index, persona)
        turn_rows.extend(rows["turns"])
        event_rows.extend(rows["events"])
        relation_rows.extend(rows["relations"])
        qa_rows.extend(rows["qa"])

    write_jsonl(output / "personas.jsonl", persona_rows)
    write_jsonl(output / "dialogues.jsonl", turn_rows)
    write_jsonl(output / "gold_events.jsonl", event_rows)
    write_jsonl(output / "gold_update_relations.jsonl", relation_rows)
    write_jsonl(output / "qa.jsonl", qa_rows)
    return {
        "personas": len(persona_rows),
        "dialogue_turns": len(turn_rows),
        "gold_events": len(event_rows),
        "gold_update_relations": len(relation_rows),
        "qa": len(qa_rows),
    }


def validate_dataset(output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    import json

    def load(name: str) -> list[dict]:
        with (output / name).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    turns = load("dialogues.jsonl")
    events = load("gold_events.jsonl")
    relations = load("gold_update_relations.jsonl")
    qa_rows = load("qa.jsonl")

    turn_ids = {turn["turn_id"] for turn in turns}
    event_ids = {event["event_id"] for event in events}
    errors: list[str] = []

    for event in events:
        for turn_id in event["evidence_turn_ids"]:
            if turn_id not in turn_ids:
                errors.append(f"event {event['event_id']} references missing turn {turn_id}")

    for relation in relations:
        if relation["new_event_id"] not in event_ids:
            errors.append(f"relation references missing new event {relation['new_event_id']}")
        if relation["old_event_id"] not in event_ids:
            errors.append(f"relation references missing old event {relation['old_event_id']}")

    for qa in qa_rows:
        for turn_id in qa["gold_evidence_turn_ids"]:
            if turn_id not in turn_ids:
                errors.append(f"qa {qa['question_id']} references missing turn {turn_id}")
        for event_id in qa["gold_event_ids"]:
            if event_id not in event_ids:
                errors.append(f"qa {qa['question_id']} references missing event {event_id}")
        if qa["requires_abstention"] and qa["gold_evidence_turn_ids"]:
            errors.append(f"abstention qa {qa['question_id']} should not have evidence")

    return errors


def _build_user_rows(index: int, persona: SeedPersona) -> dict[str, list[dict]]:
    uid = persona.user_id
    event = lambda n: f"{uid}_e{n:02d}"
    turn = lambda n: f"{uid}_s{n:02d}_t01"
    session = lambda n: f"{uid}_s{n:02d}"
    updated_plan = _clean_update_phrase(persona.updated_plan)

    dates = [
        "2026-03-01",
        "2026-03-08",
        "2026-03-15",
        "2026-03-22",
        "2026-03-29",
        "2026-04-05",
        "2026-04-12",
        "2026-04-19",
        "2026-04-26",
        "2026-05-03",
    ]
    texts = [
        f"我最近在{persona.initial_plan}，先把時間排起來。",
        f"如果要安排地點，我一開始比較想選{persona.old_location}。",
        f"我其實一直{persona.stable_preference}，這點應該不會變。",
        f"{persona.constraint}，那段時間不要排重要事情。",
        f"{persona.relationship}，很多安排都會一起確認。",
        f"我已經{persona.completed_event}，接下來可以進下一步。",
        f"我後來不想繼續{persona.initial_plan}，改成{updated_plan}。",
        f"不是{persona.old_location}，後來覺得{persona.new_location}比較適合。",
        f"我還是{persona.stable_preference}，只是想把方向收斂一點。",
        f"我希望{persona.future_event}，這是下一個時間點。",
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
        _event(event(1), uid, dates[0], "plan", f"使用者{persona.initial_plan}", [persona.initial_plan], turn(1), 0.82),
        _event(event(2), uid, dates[1], "preference", f"使用者一開始偏好{persona.old_location}", [persona.old_location], turn(2), 0.66),
        _event(event(3), uid, dates[2], "preference", f"使用者{persona.stable_preference}", [persona.stable_preference], turn(3), 0.72),
        _event(event(4), uid, dates[3], "constraint", persona.constraint, [persona.constraint], turn(4), 0.74),
        _event(event(5), uid, dates[4], "relationship", persona.relationship, [persona.relationship], turn(5), 0.62),
        _event(event(6), uid, dates[5], "completed_event", f"使用者已經{persona.completed_event}", [persona.completed_event], turn(6), 0.7),
        _event(event(7), uid, dates[6], "negated_fact", f"使用者不再{persona.initial_plan}", [persona.initial_plan], turn(7), 0.86),
        _event(event(8), uid, dates[6], "plan", f"使用者改成{updated_plan}", [updated_plan], turn(7), 0.9),
        _event(event(9), uid, dates[7], "preference", f"使用者改為偏好{persona.new_location}", [persona.old_location, persona.new_location], turn(8), 0.78),
        _event(event(10), uid, dates[9], "future_event", f"使用者希望{persona.future_event}", [persona.future_event], turn(10), 0.72),
    ]

    relations = [
        {
            "new_event_id": event(7),
            "old_event_id": event(1),
            "relation": "supersedes",
            "reason": "使用者明確表示不再維持原計畫",
            "evidence_turn_ids": [turn(7)],
        },
        {
            "new_event_id": event(8),
            "old_event_id": event(1),
            "relation": "supersedes",
            "reason": "使用者以新計畫取代原計畫",
            "evidence_turn_ids": [turn(7)],
        },
        {
            "new_event_id": event(9),
            "old_event_id": event(2),
            "relation": "corrects",
            "reason": "使用者明確修正原先地點偏好",
            "evidence_turn_ids": [turn(8)],
        },
        {
            "new_event_id": event(3),
            "old_event_id": event(1),
            "relation": "supplements",
            "reason": "偏好補充原計畫方向，但不取代原計畫",
            "evidence_turn_ids": [turn(3)],
        },
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
        "importance": importance,
    }


def _qa_rows(index: int, persona: SeedPersona, event, turn) -> list[dict]:
    uid = persona.user_id
    base = (index - 1) * 10
    updated_plan = _clean_update_phrase(persona.updated_plan)

    def q(n: int) -> str:
        return f"q{base + n:03d}"

    return [
        _qa(q(1), uid, f"{persona.name} 一開始在準備什麼？", "single_session_fact", f"使用者{persona.initial_plan}", [turn(1)], "2026-03", False, [event(1)]),
        _qa(q(2), uid, f"{persona.name} 現在的主要計畫是什麼？", "knowledge_update", f"使用者改成{updated_plan}", [turn(7)], "current", False, [event(8)], _sup(event(8), event(1))),
        _qa(q(3), uid, f"{persona.name} 三月時原本偏好的地點是哪裡？", "temporal_reasoning", f"使用者一開始偏好{persona.old_location}", [turn(2)], "2026-03", False, [event(2)]),
        _qa(q(4), uid, f"{persona.name} 現在偏好的地點是哪裡？", "conflict_resolution", f"使用者改為偏好{persona.new_location}", [turn(8)], "current", False, [event(9)], _correct(event(9), event(2))),
        _qa(q(5), uid, f"{persona.name} 有什麼固定限制？", "single_session_fact", persona.constraint, [turn(4)], "current", False, [event(4)]),
        _qa(q(6), uid, f"{persona.name} 和誰有固定合作或照護關係？", "single_session_fact", persona.relationship, [turn(5)], "current", False, [event(5)]),
        _qa(q(7), uid, f"{persona.name} 已經完成了什麼？", "single_session_fact", f"使用者已經{persona.completed_event}", [turn(6)], "current", False, [event(6)]),
        _qa(q(8), uid, f"{persona.name} 接下來希望完成什麼？", "single_session_fact", f"使用者希望{persona.future_event}", [turn(10)], "future", False, [event(10)]),
        _qa(q(9), uid, f"{persona.name} 的穩定偏好是什麼？", "multi_session_reasoning", f"使用者{persona.stable_preference}", [turn(3), turn(9)], "current", False, [event(3)]),
        _qa(q(10), uid, f"{persona.name} 有沒有提過 {persona.unknown_target}？", "abstention", "", [], "current", True, []),
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
