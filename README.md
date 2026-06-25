# MT5 Auto Trading System

XAUUSD Python 自動交易系統：在 Linux/Wine 上透過 pymt5linux bridge 驅動 MT5，由 fleet supervisor 監管多策略 child process，具備硬性風控、state 隔離與研究層分離。

> Status: research + demo/live operations。未通過 backtest、paper、operational review 三道門檻的策略一律視為實驗性。

## 核心架構

```text
MT5 terminal (Wine) ──IPC── pymt5linux bridge 127.0.0.1:18812
                                      │
                  supervisors/mt5_strategy_fleet_supervisor.py
                                      │
                  auto_quant/active_plan.json  (hash 驅動，git ignored)
                                      │
        ┌─────────────────────┬───────────────────┬────────────────────┐
        ▼                     ▼                   ▼                    ▼
  trend_strategy.py   momentum_surfer.py   asian_reversal.py   (future sleeves)
        │                     │                   │
        └─────────────────────┴───────────────────┘
                                      ▼
            auto_quant/state/*.json + auto_quant/logs/*.log
            shared/account_metrics.py (單一權益基準)
```

分層職責：
1. Terminal / bridge — MT5 GUI login、AutoTrading、IPC bridge 就緒。
2. Fleet supervisor — 監聽 `active_plan.json` 的 `hash` 欄位，hash 變動才重啟 strategy child。
3. Strategy child — 訊號、風控、下單、trailing、state save。
4. State / backups — runtime state 與 pre-edit 備份本地保存，不進 git。
5. Research layer — 只讀腳本產出 CSV/Markdown，不碰 active_plan。

## 策略 Sleeve

| Sleeve | 檔案 | 邏輯 | 狀態 |
|--------|------|------|------|
| Trend | `strategies/mt5_xauusd_trend_strategy.py` | SMA + HTF trend filter + ATR stop/reward + false-breakout complement | 主力 |
| Momentum Surfer | `strategies/mt5_xauusd_momentum_surfer_strategy.py` | M1 動量加速度，極緊 trailing，低勝率高盈虧比 | 活躍 |
| Asian Reversal | `strategies/mt5_xauusd_asian_reversal_strategy.py` | 亞洲盤 pivot stall 反轉 fade | 活躍 |
| False-breakout | 內嵌於 trend strategy | SELL-only upthrust reversal，獨立 magic | complement |
| OCC / Doomsday / Tide Wave Grid | `docs/retired_strategies/` | 歷史研究歸檔 | retired |

組合方式：避免簡單 overlay；若要併用多 sleeve，採獨立 process + magic number + 共享 exposure/DD cap。

## 硬性風控（所有 sleeve 共通）

- 起始資金參考：10000
- 日內最大回撤：3%
- 總回撤上限：10%
- 獲利目標：5%（達標暫停）
- 單倉、單 symbol、單帳戶
- warmup、cooldown、spread filter、max hold、max trades/day
- Best Day concentration guard（防單日暴利主導累積正 PnL）
- 週五 23:55 強制平倉
- `shared/account_metrics.py` 統一權益基準，避免各 sleeve 自行 seed

## IPC 穩定性（關鍵）

Wine + pymt5linux 長連線會退化（`stream has been closed`、`No IPC connection`、`history_deals_get` 回 None）。防禦模式：

- **Fresh-client-per-call**：每個重 MT5 操作（account/rates/positions/order/deals/symbol）都用短生命周期 client：`initialize() → 單一操作 → shutdown()` in `finally`。
- Reconnect-on-failure 作為 fallback：transient IPC 錯誤時 shutdown→重建 client→initialize→retry，並設 60s stabilization window 擋入場。
- `bridge_healthy()` 健檢**不可**帶 `path=` 參數，否則每次健檢都開新的 terminal64.exe GUI 視窗。
- 重新連線後若立即產生訊號，需等過 stabilization window 才可進場。

## 操作流程

### 啟停

```bash
./run_mt5_system.sh status     # 查 supervisor + bridge + child process + 最近 log
./run_mt5_system.sh start      # 啟動 supervisor（拒絕重複啟動）
./run_mt5_system.sh stop       # 透過 clean_mt5_restart.py 乾淨停止
./run_mt5_system.sh restart    # 殺 bridge + supervisor + child，依序重啟
```

### 修改 live 參數

1. 備份 `auto_quant/active_plan.json` 與 `auto_quant/state/`。
2. 編輯 JSON（cmd array 用 `set_flag` pattern 改參數）。
3. 更新 `hash` 欄位為新唯一字串。
4. 5-10 秒內 supervisor 偵測到變化，自動 terminate + respawn。

未更新 hash，supervisor 不會偵測到變化。

### 暫停 / 解除暫停

策略因 profit target 或 DD 暫停後，`_risk_guard()` 只會 set `paused=True`，不會自動清除。解除步驟：
1. 殺 supervisor PID → 殺 strategy PID。
2. 寫 `paused: false, paused_reason: ""` 到 state JSON。
3. 再重啟 supervisor。

順序錯了會被舊 process 的 `_save_state()` 覆蓋回去。

## 研究腳本

| 腳本 | 用途 |
|------|------|
| `backtest_compare_strategies.py` | 多策略回測比較，輸出至 `backtest_reports/` |
| `research_xauusd_weekly_complement_backtest.py` | 週級 complement 比較 |
| `research_xauusd_trend_plus_complement_backtest.py` | trend + false-breakout 模擬 |
| `research_xauusd_complement_stability.py` | 參數/區段穩定性診斷 |
| `research_xauusd_london_ny_session_backtest.py` | session window 實驗（只讀） |
| `forward_atm_asia_sweep_signal_logger.py` | 亞洲盤 sweep 訊號記錄 |
| `scripts/readonly_mt5_status.py` | 唯讀帳戶/持倉健康檢查 |
| `scripts/export_live_trade_audit.py` | 匯出 live trade audit |
| `scripts/market_regime_trade_report.py` | regime 分析 |
| `scripts/build_trade_journal_db.py` | 交易日誌資料庫 |

注意：`backtest_compare_strategies.py` 的策略參數是 class attribute hardcoded，不會自動同步 `active_plan.json`。回測前先比對 live params 與 class params 是否一致。

回測限制：不模擬 trailing stop、loss pause、daily cap、consecutive loss cooldown、spread filter。live 績效會偏離回測（通常 trade 數更少、PnL 更緊）。回測用於相對比較，不預測絕對報酬。

## 測試

```bash
python3 -m py_compile \
  strategies/mt5_xauusd_trend_strategy.py \
  supervisors/mt5_strategy_fleet_supervisor.py

python3 -m pytest tests/ -q
# 關鍵安全測試
python3 -m pytest tests/test_xauusd_trend_strategy_risk_controls.py -q
python3 -m pytest tests/test_momentum_surfer_safety_features.py -q
```

## 不進 git 的檔案

`.gitignore` 排除：`auto_quant/*.json`、`auto_quant/state/`、`auto_quant/backups/`、`auto_quant/logs/`、`backtest_reports*/`、`*.log`、`*.out`、`*.csv`、`research/`、`research_*.py`、`backtest_*.py`、`tmp_*.py`、credentials。

避免意外發布帳戶狀態、log、報告、secret。

## 安全規則

- 不 commit secret、MT5 帳戶檔、log、state snapshot、報告。
- 修改 live 參數前必備份 `active_plan.json` 與 state。
- 改 `active_plan.json` + bump hash 讓 supervisor reload，不要手動開第二個 MT5 terminal 或第二個 strategy process。
- 重啟後驗證：MT5 login + AutoTrading、bridge 18812、active_plan enabled、child process PID、equity/positions/orders/magic、log 有新 bar/risk loop。
- 研究腳本不得修改 `active_plan.json`。
- 正常 shutdown/restart 不自動 `close_all_positions()`，除非用戶明確要求 flatten。

## 開發時程

- 2026-04 — MQL5 EA 盤點 + Pine Script porting
- 2026-05 early — Python live strategy + bridge + supervisor 成形
- 2026-05 mid — Prop/FTMO 風控門檻（daily DD、total DD、profit target、warmup、cooldown）
- 2026-05 late — live 參數治理：備份、half-close、trailing stability、per-order lot cap、concentration guard、account-switch recovery
- 2026-06 — complement sleeve 研究、false-breakout reversal、fresh-client-per-call IPC fix、HTF price-position override、unified chart-time、momentum surfer safety、Friday force-close、trade journal DB

## Repository

GitHub: https://github.com/fu06pi/mt5-auto-trading-system
Branch: `feature/xauusd-inverse-tight-main`
