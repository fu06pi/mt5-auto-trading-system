# doomsday 績效分析

## 報告範圍
本報告整合兩份資料：
1. MT5 history_deals_get 實成交紀錄（完整績效）
2. doomsday_trade_records.csv 進場/執行 proxy（輔助觀察）

## A. 實成交績效
- 來源：MT5 history_deals_get 的已成交紀錄
- 匯出 CSV：/home/chain4655/Documents/Projects/MT5/doomsday_full_history.csv
- 交易檔案數：15 deals / 7 closed trades / 1 open trades
- 時間範圍：2026-05-02 02:15:02 → 2026-05-04 20:20:08

### 總績效
- 起始資金：10000.00
- 累積已實現淨損益：77.94
- 勝率：71.43%
- Profit Factor：3.27
- Expectancy / trade：11.13
- 估算最大回撤：-0.29%
- 最長連勝：3
- 最長連敗：1

### 日別表現
| 日期 | 交易筆數 | 贏 | 輸 | 淨損益 |
|---|---:|---:|---:|---:|
| 2026-05-02 | 2 | 1 | 1 | 0.30 |
| 2026-05-04 | 5 | 4 | 1 | 77.64 |

## B. 4/15-4/16 進場 / 執行 proxy
- 來源：/home/chain4655/Documents/Sample/Python/doomsday_trade_records.csv
- 資料型態：ENTRY / BAR / ORDER_SEND / CLOSE_RESULT 的執行紀錄，不是完整 realized history
- 時間範圍：2026-04-15 13:59:48 → 2026-04-16 11:00:01
- 進場筆數：32
- Bars：285
- Order_send：214
- Close_result：42

### Proxy equity 觀察
- 起始 equity：10000.00
- 結束 equity：9779.79
- 變化：-220.21
- 最低 equity：9726.03
- 最高 equity：10000.00
- 兩日分布：2026-04-15 16 筆 / 2026-04-16 16 筆

### Proxy 解讀
- 這份資料只能代表「進場時的 equity 漂移」與「執行過程」
- 不能直接當成已實現損益
- 但它顯示 4/16 這段 doomsday 確實有在跑，且進場節奏密集

## 綜合判讀
- 實成交績效：截至 5/4 早上，doomsday 是獲利狀態，PF > 3，勝率約 71%
- 早期 proxy：4/15-4/16 的進場/執行紀錄呈現 equity 下滑，代表那一段偏弱
- 兩者合起來看，策略不是一路都強；中間應該有版本/參數/市場狀態差異
- 執行層仍有 No IPC connection / symbol_info unavailable 風險，需要另行處理

## 備註
- 這份報告已把兩份資料一起納入。
- 若你要 thesis 用圖表，我可以下一步直接幫你做：
  - 累積 equity curve
  - 日別損益圖
  - 交易分布圖
  - 長文版論文敘述段落
