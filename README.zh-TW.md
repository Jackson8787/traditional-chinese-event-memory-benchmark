# 繁體中文事件中心長期記憶 Benchmark

[English](README.md)

這個 repository 是一個繁體中文長期對話記憶的研究原型與 benchmark。專題重點放在事件中心檢索、衝突感知的記憶更新，以及跨多個 session 的 evidence-grounded 問答。

資料集是公開安全的合成資料，情境會參考公開來源做 factual grounding，但不包含私人聊天紀錄，也不包含 API key。

## 目前狀態

這個專題目前可以作為研究原型與 benchmark package 上傳。離線測試、public-grounded dataset、hard challenge split、cached realistic sample，以及 manual audit summary 都已放在資料夾內，因此主要實驗主張可以在不呼叫外部 API 的情況下檢查。

`event_memory_gold_v2` 應該被解讀為 oracle upper-bound，不是可部署模型。cached `event_memory_llm` run 是 user-scope realistic sample 與 diagnostic sample，使用 `--llm-turn-scope evidence` 產生評估；它不是完整 360 題主結果，也不是完整主結果。這個區分對 baseline fairness 很重要。

## 內含資料與結果

| 項目 | 路徑 | 用途 |
| --- | --- | --- |
| Main benchmark | `data/public_grounded_benchmark/` | 12 個 persona、每個 30 個 session、360 題 QA、120 條 update relation |
| Hard challenge | `data/public_grounded_hard_challenge/` | 72 題較困難 QA，用來測試 update-chain、temporal 與 distractor 壓力 |
| Main results | `outputs/public_grounded_benchmark/` | Oracle 與 baseline metrics |
| Hard challenge results | `outputs/public_grounded_hard_challenge/` | Hard split oracle metrics |
| Realistic LLM sample | `outputs/realistic_llm_memory_sample/` | Cached 60 題 LLM-memory diagnostic sample |
| Manual audit | `outputs/manual_audit/realistic_llm_memory_sample/` | Relation 與 answer quality 的 auto-labeled audit package |

## 關鍵指標

| Evaluation | Answer | Evidence@3 | Update | Temporal | 說明 |
| --- | ---: | ---: | ---: | ---: | --- |
| Main oracle upper-bound | 0.964 | 0.957 | 0.997 | 1.000 | 完整 public-grounded benchmark |
| Hard challenge oracle | 0.653 | 0.653 | 0.653 | 0.000 | Challenge split |
| Realistic LLM memory sample | 0.817 | 0.759 | 0.800 | 1.000 | 60 題 diagnostic sample |

Realistic sample 另外需要注意 `faithful = 0.433` 與 `strict_state = 0.360`。這兩個指標是目前最主要的弱點，代表回答可能看似可用，但內部狀態仍不夠乾淨或不夠忠實。

## 重要失敗模式

這些數字不應被解讀成成熟 benchmark 或 production memory system。

- Hard challenge 的表現很不平均：`hard_conflict` 是 47/48 正確，但 `hard_temporal` 是 0/24。整體 hard-challenge 分數會掩蓋 time-relevance 子集的完全失敗。
- Cached LLM result 只涵蓋 60 題，不是完整 360 題 main benchmark。它適合當 diagnostic sample，不適合當成統計上充分的最終模型結果。
- Realistic sample 的 state quality 很弱：`faithful = 0.433`、`strict_state = 0.360`。只看 answer accuracy 會高估記憶狀態的品質。
- Manual audit package 是 auto-labeled。它可以幫助定位錯誤，但不是獨立真人驗證。
- 資料是 synthetic 且 public-grounded，不應被視為真實私人長期對話記憶分佈的直接代表。

目前錯誤分析與下一步請看 `reports/failure_analysis_zh.md`。

## 快速開始

```bash
python3 -m venv .venv
source .venv/bin/activate
PYTHONPATH=src python3 -m unittest discover tests
```

跑測試或閱讀 cached results 不需要 API key。如果要重新跑 live LLM extraction，請先把 `api_config.example.env` 複製成 `api_config.env`，再填入本機 credentials。不要把 `api_config.env` commit 到 GitHub。

## 主要離線檢查

```bash
PYTHONPATH=src python3 -m event_memory.cli audit-public-grounded-dataset --dataset-dir data/public_grounded_benchmark
PYTHONPATH=src python3 -m event_memory.cli audit-public-grounded-hard-dataset --dataset-dir data/public_grounded_hard_challenge
PYTHONPATH=src python3 -m event_memory.cli run-experiment --method event_memory_gold_v2 --dataset-dir data/public_grounded_benchmark --output-dir outputs/public_grounded_benchmark
```

更完整的重現方式請看 `REPRODUCIBILITY.md`。
