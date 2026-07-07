# 錯誤分析

這份報告整理目前 artifacts 已經暴露出的主要弱點。結論刻意保守：這個專案目前應被解讀為研究原型與 diagnostic benchmark package，不是成熟 benchmark，也不是 production-ready memory system。

## Hard Challenge

Hard challenge 共有 72 題：

| 子集 | Total | Answer | Evidence@3 | Update | Temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard_conflict | 48 | 0.979 | 0.979 | 0.979 | 0.000 |
| hard_temporal | 24 | 0.000 | 0.000 | 0.000 | 0.000 |
| overall | 72 | 0.653 | 0.653 | 0.653 | 0.000 |

整體分數會掩蓋一個重要不對稱：oracle upper-bound 幾乎能處理明確的 supersedes-expansion 問題，但在 hard split 的 time-relevance 問題上全部失敗，`hard_temporal` 是 0/24。

Error records 顯示固定模式：當問題問「只看 2026 年 3 月以前，舊狀態是什麼、後來改成什麼」時，系統常抓到另一條 update chain，例如 future activity 或 reminder updates，而不是題目要求的 location/tool update chain。Hard split 的 25 筆 error row 都標成 `update_error`；其中 24 筆是 `hard_temporal`，對應到 0/24 的子集結果。

解讀：目前 event/update representation 可以處理很多直接 supersedes case，但還不能穩定地用 time scope 和目標 state dimension 共同限制 update-chain retrieval。

## Realistic LLM Sample

Realistic LLM memory result 是 60 題 diagnostic sample，不是完整 360 題 benchmark：

| Metric | Value |
| --- | ---: |
| Answer | 0.817 |
| Evidence@3 | 0.759 |
| All Evidence@3 | 0.481 |
| Update | 0.800 |
| Faithful | 0.433 |
| Strict state | 0.360 |

因此不能只用 answer accuracy 宣稱 memory system 已經穩定。低 faithful 與 strict-state 代表系統可能生成看似合理的答案，但內部記憶狀態仍不完整或不乾淨。

## Audit 限制

目前附上的 manual-audit package 是 auto-labeled。它適合用來快速找 failure modes，但不是獨立真人驗證。

目前 audit summary：

- Relation rows labeled: 100。
- `supersedes` precision: 0.333。
- `supplements` precision: 0.541。
- Answer rows labeled: 50。
- Evidence complete rate: 0.440。
- Evidence complete-or-partial rate: 0.780。

這些數字應被視為 triage evidence。若要做強主張，需要真人審核，最好至少兩位 annotator 並處理 disagreement。

## Dataset Scope

資料集是 synthetic 且 public-grounded。這讓資料比較安全、可重現，但不能證明真實私人長期對話中的錯誤分佈會相同。

目前規模適合作為專題 benchmark：

- 12 personas。
- 每個 persona 30 sessions。
- 360 main QA items。
- 72 hard-challenge QA items。
- 120 gold update relations。

但若要宣稱 general-purpose benchmark，仍需要更多資料、外部 replication 與真人標註。

## 建議下一步

- 將 hard temporal failure 依 state dimension、retrieved chain、missed evidence ids 分解。
- 增加同時使用 update-chain relation、valid time 與 target state dimension 的 retrieval constraint。
- API budget 足夠時，把 realistic LLM run 從 60 題擴到完整 360 題。
- 真人審核 audit rows，尤其是低信心 `supersedes`、temporal reasoning、evidence completeness 案例。
- 在把 benchmark 說成廣泛驗證前，加入外部 baseline 或第三方 rerun。
