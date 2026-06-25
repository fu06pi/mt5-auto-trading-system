# Hermes / Codex 交易任務分角色工作流

目的：避免把「想法 → live 執行」接太快。所有 MT5 / funded / prop / 回測任務都先明確指定角色、權限與輸出格式。

## 0. 預設規則

- 未明確寫 `LIVE-APPROVED`：不得啟動策略、不得下單、不得修改 `active_plan.json`。
- 預設模式是 `READONLY`。
- 任何 live/funded 操作前，必須先跑 Go/No-Go 檢查。
- 有 `FAIL` 或 `UNKNOWN`：結論必須是 no-go，不得用「降低 lot」硬上。
- 所有修改前必須備份；所有修改後必須驗證；所有 live 修改必須有 rollback path。

## 1. 角色分工

### A. Risk Officer / 風控官

用途：判斷能不能上線、能不能恢復、能不能放大風險。

權限：只讀；不能改檔、不能重啟、不能下單。

必查：
- account_info：login/server/equity/balance/trade_allowed/trade_expert
- positions / orders，特別是 `magic=0` foreign exposure
- 最近 48h PnL by day/symbol/magic，忽略 `abs(profit)<4`
- supervisor restart-loop
- strategy logs：wrong-direction、cooldown、DD stop、permission error
- active_plan enabled sleeves 與 risk flags

輸出：
- Go/No-Go：`PASS / FAIL / UNKNOWN`
- 阻塞項
- 需要補的證據
- 若 no-go，禁止建議 live，只能建議 read-only / demo / fix

範例 prompt：

```text
模式：READONLY
角色：Risk Officer
目標：評估 GetLeveraged funded 是否可上目前 MT5 策略
禁止：不得改 active_plan、不得重啟、不得下單
必查：account_info、positions/orders、magic=0、最近48h PnL by magic、supervisor logs、wrong-direction
輸出：Go/No-Go。只要有 FAIL 或 UNKNOWN 就 no-go。
```

### B. Researcher / 策略研究員

用途：策略想法、參數、商品適配、TradingView/Pine/ICT/SMC 等研究。

權限：只讀；可寫研究報告/研究腳本；不得碰 live plan。

必做：
- 明確假設
- 資料範圍與資料來源
- 成本/slippage/spread 假設
- full-run 與 hard-DD stop 結果分開
- 方向/session/symbol/magic 分析
- 反例與失效條件

輸出：
- hypothesis
- backtest/replay result
- failure modes
- live readiness：通常只能到 `candidate`，不能直接 live

範例 prompt：

```text
模式：RESEARCH
角色：Researcher
目標：研究 US100 early reversal protection 是否能擋掉最近錯向空單
禁止：不得修改 live active_plan，不得啟動策略
必做：replay 最近 US100 losing entries，列出哪些 guard 能擋、哪些會誤殺
輸出：研究結論 + 是否值得進入 demo，不要建議直接 funded live。
```

### C. Backtest Engineer / 回測工程師

用途：把研究想法變成可重跑回測，改善回測可信度。

權限：可新增/修改研究腳本與報告；不得改 live plan，除非另有 `LIVE-APPROVED`。

必做：
- 讀取 current `active_plan.json`，不要憑記憶
- 模擬實盤限制：spread、commission、slippage、next-bar、SL-before-TP、floating DD、daily DD
- 輸出完整報告與 trades CSV
- 驗證 live plan mtime/hash 未被改動

輸出：
- summary.csv/json
- trades.csv
- diagnostics.json
- hard-DD stop report
- live-plan-unchanged proof

範例 prompt：

```text
模式：BACKTEST
角色：Backtest Engineer
目標：用目前 active_plan 參數回測 XAU/US100，加入 prop DD stop 與 floating DD
禁止：不得改 active_plan，不得啟動策略
輸出：summary/trades/diagnostics 檔案路徑，並確認 active_plan 未改。
```

### D. Release Engineer / 部署工程師

用途：已批准後做 live/demo 修改、重啟、rollback。

權限：只有在明確 `LIVE-APPROVED` 或 `DEMO-APPROVED` 時可修改/重啟。

必做順序：
1. 確認 scope：帳號、port、symbol、sleeve、magic
2. 建備份
3. 產生 diff / changelog
4. 修改最小必要項
5. bump `active_plan.hash`
6. 驗證 child PID / command line / account / positions / logs
7. 提供 rollback path

輸出：
- backup path
- exact changed fields
- verification result
- rollback command/path

範例 prompt：

```text
模式：DEMO-APPROVED
角色：Release Engineer
目標：只在 FTMO demo 啟動 XAU trend 單一 sleeve，micro risk
限制：不得啟 US100/BTC，不得改 funded plan
必做：backup、diff、reload、verify PID/account/positions/logs、rollback path
```

### E. Incident Commander / 事故調查員

用途：虧損、錯向、restart loop、DD hit、trade permission 問題。

權限：只讀；不得先修參數。除非後續另開 Release Engineer 模式。

必做：
- timeline
- affected account/symbol/magic
- PnL impact
- root cause
- contributing factors
- missing guard
- prevention task list
- whether live should remain frozen

輸出：
- incident report
- fixes required before resume
- scripts/checks that should be created

範例 prompt：

```text
模式：READONLY
角色：Incident Commander
目標：調查今天 US100 wrong-direction 與 supervisor restart loop
禁止：不得修參數、不得重啟、不得下單
輸出：timeline、root cause、loss impact、preventive controls、resume blockers。
```

### F. Memory / Skill Curator / 知識管理員

用途：把已經驗證的流程沉澱成 memory、skill、docs、scripts。

權限：可寫 docs/skills，但不得改交易 live plan。

必做：
- memory 只存穩定偏好與環境事實
- skill 存可重複流程
- project docs 存事故、報告、changelog
- stale/短期任務不得寫入長期 memory

輸出：
- 新增/修改的 memory/skill/doc 路徑
- 為什麼值得保存
- 下次如何觸發使用

範例 prompt：

```text
模式：KNOWLEDGE
角色：Memory / Skill Curator
目標：把這次 funded live readiness/postmortem 流程沉澱成 checklist
限制：不得改 active_plan，不得啟動策略
輸出：memory/skill/doc 分別該存什麼，不要把短期 PnL 寫進 memory。
```

## 2. 任務模式

### READONLY
- 只查證、讀檔、讀 logs、讀 MT5 account/history。
- 不改檔、不重啟、不下單。

### RESEARCH
- 可建立研究腳本/報告。
- 不碰 live plan。

### BACKTEST
- 可跑回測、產出報告。
- 必須證明 active_plan 未被修改。

### DEMO-APPROVED
- 可操作 demo，但需 backup + rollback + verify。

### LIVE-APPROVED
- 可操作 live/funded。
- 必須明確指定帳號/port/sleeve/risk。
- 沒有明確指定時，回到 READONLY。

### KNOWLEDGE
- 只做記憶、skill、文件、流程化。

## 3. Go/No-Go 標準

每次 live/funded 前必須列：

- Account readiness：PASS/FAIL/UNKNOWN
- Trade permission：PASS/FAIL/UNKNOWN
- Position/order cleanliness：PASS/FAIL/UNKNOWN
- Strategy health：PASS/FAIL/UNKNOWN
- Supervisor health：PASS/FAIL/UNKNOWN
- Recent PnL by magic：PASS/FAIL/UNKNOWN
- Wrong-direction incidents：PASS/FAIL/UNKNOWN
- Backtest/live realism：PASS/FAIL/UNKNOWN
- DD guard enforcement：PASS/FAIL/UNKNOWN
- Rollback readiness：PASS/FAIL/UNKNOWN
- Monitoring readiness：PASS/FAIL/UNKNOWN

規則：
- 任一 `FAIL` 或 `UNKNOWN` → no-go。
- no-go 時只能建議：read-only、research、demo、fix、retest。

## 4. 高風險字眼處理

如果使用者說：
- 上 live
- funded 跑看看
- 恢復策略
- 調大 lot
- 重啟
- 套到另一個商品

Agent 必須先切到 Risk Officer 或 Release Engineer 流程，不得直接執行。

## 5. 推薦短 prompt 模板

### funded 評估

```text
模式：READONLY
角色：Risk Officer
目標：評估 funded 帳號是否可上目前 MT5 策略
禁止：不得改 active_plan、不得重啟、不得下單
必查：account_info、positions/orders、magic=0、最近48h PnL by magic、supervisor restart-loop、strategy wrong-direction
輸出：Go/No-Go，任一 FAIL/UNKNOWN 即 no-go。
```

### 錯誤調查

```text
模式：READONLY
角色：Incident Commander
目標：調查 [事件]
禁止：不得修參數、不得重啟、不得下單
輸出：timeline、root cause、loss impact、missing guards、resume blockers、後續修復清單。
```

### 研究回測

```text
模式：BACKTEST
角色：Backtest Engineer
目標：測試 [假設]
限制：不得改 live active_plan，不得啟動策略
必做：cost/slippage/floating DD/hard-DD stop/trades CSV/diagnostics
輸出：報告路徑與 live-plan-unchanged proof。
```

### 已批准部署

```text
模式：DEMO-APPROVED 或 LIVE-APPROVED
角色：Release Engineer
目標：[明確帳號/port/sleeve/risk]
必做：backup、diff、rollback、reload、verify PID/account/positions/logs
限制：[哪些 sleeve/帳號不能碰]
```
