# 實驗總結

這個專題目前定位為繁體中文長期對話記憶的研究原型與 benchmark。核心問題是：系統能不能在多 session 的長上下文中追蹤事件、辨識更新關係，並在回答時給出正確且可追溯的 evidence。

## 主要結果

| 評估 | Answer | Evidence@3 | Update | Temporal | 解讀 |
| --- | ---: | ---: | ---: | ---: | --- |
| Main oracle upper-bound | 0.964 | 0.957 | 0.997 | 1.000 | 測試資料與事件標註本身大致可用 |
| Hard challenge oracle | 0.653 | 0.653 | 0.653 | 0.000 | 困難題能有效拉開更新鏈與干擾實體能力 |
| Realistic LLM memory sample | 0.817 | 0.759 | 0.800 | 1.000 | LLM 建記憶可回答多數問題，但狀態忠實度仍不足 |

`event_memory_gold_v2` 是 oracle upper-bound，不是可部署模型。`event_memory_llm` 的 cached result 是 user-scope realistic sample 與 diagnostic sample，使用 `--llm-turn-scope evidence` 評估，不是完整 360 題主結果，也不是完整主結果。這個標示是 baseline fairness 的必要條件。

## 目前最弱處

- Hard challenge 的錯誤不是平均分布：`hard_conflict` 47/48 正確，但 `hard_temporal` 0/24，代表 time-relevance update-chain reasoning 目前是明確失敗模式。
- Realistic sample 的 faithful = 0.433，strict_state = 0.360，表示回答看似正確時，內部狀態仍可能沒有完整、乾淨地保留。
- Supersedes relation 的 precision 偏低，系統還不穩定地判斷何時新事件真的取代舊事件。
- Manual audit 目前是 auto-labeled 初稿，適合用來快速定位問題，但不是獨立真人驗證；正式報告前仍應人工抽查。

## 結論

這個專題已經足夠作為研究原型與 GitHub benchmark 上傳。若要進一步做成論文或展示，需要補強人工審核、擴大 realistic sample，並改善 relation update 的判斷品質。
