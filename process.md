# process.md — 两位 AI 的协作进展总结

> 项目：`trade-agent-benchmark`（量化交易 Agent 基准：AlphaCrafter + FactorMiner 日频前向走步，9 条世界线为未来行情，统一 3bps 摩擦）。
> 本文件汇总 **AI-1（数据/管线派）** 与 **AI-2（架构/计价派）** 两人各自完成的工作，并确认当前数据状态。
> 更新：2026-07-23

---

## 一、AI-1（数据抓取 + 管线打通）

**主线：把"抓不到的数据"变成"全量落盘 + 管线跑通"。**

### 1. warmup 真实日频数据（2020-01-02 ~ 2026-07-16）
- 写 `data-prepare/fetch_daily_data.py` + `asset_spec.py` + `make_panel.py`：多源优先级抓取，单资产失败不中断、可 `--only` 断点续抓。
- **攻克本机网络难题**（Clash TUN 下 Yahoo/eastmoney 限流）：
  - 实测结论：**国内源才稳**（sina/163 指数、akshare 中美债/SOX、BOC 央行中间价外汇、Binance 加密）；Yahoo/eastmoney 限流。
  - Yahoo 两坑已内置解决：① 默认 TW 节点被限 → `--rotate-clash` 逐资产换 KR/TW 节点；② **长 Chrome UA 被限流、短 UA `Mozilla/5.0` 通过**。
  - DXY 用 BOC 6 成分货币篮子公式合成（国内免 Yahoo）；铜 sina 报 cents/lb 自动 ÷100。
- 校验：19 资产 0 NaN / 0 重复 / 0 负价；vs 2026-07-16 基线 SPX 0%、DXY +0.2%、USDJPY +0.2%、BTC -1.5%、N225 -1.7%（SOX/NDX/SX5E/CN10Y 基线估计显著偏离真实，已记录）。
- 产物：`data-prepare/asset-daily-data/`（每资产 CSV + `all_close_wide.csv` + `COPPER_USD_PER_TON.csv` + `panel.parquet/csv` + `COVERAGE.md`）。

### 2. 合成在线世界线数据（2026-07-17 ~ 末阶段）
- 写 `data-prepare/gen_worldline_online.py`：解析 `wordline1..9.md` 每阶段资产终点 → 插值日频 close。
- **锚点策略（关键）**：warmup 真实价 vs 世界线"估计基线"差异大 → 默认 **re-anchor**：按世界线相对路径锚到真实 2026-07-16 价（价格 log-linear、收益率/VIX 线性），消除边界断层。
- 校验 WL1（台海闪电战）：SPX 7534→5204（stage2 -33%）、VIX 16.7→84.9、黄金 3980→5896、沪深300 4699→3166（-36%），与世界线叙事一致。

### 3. 管线打通 + 文档
- `adapters/build_inputs.py`（panel→AC session + FM panel）+ `scheduler/walk_forward.py`（日频游标/防穿越/每月新闻）dryrun 全链路验证通过。
- 写 `RUN.md`（操作手册）、更新 `plan.md` 进度。
- **提交**：`b086837`(warmup数据) → `f04f474`(合成数据) → `73774bc` → `d30cd01`(文档)。

---

## 二、AI-2（架构修订 + 计价统一）

**主线：把"数据口径"升级为"可配置的 Agent 宇宙 + USD 单一计价"。**（改动尚未提交，全部在工作区）

### 1. 宇宙拆分：15 可交易 + 5 信号 = 20（plan.md §7.1）
- **汇率（DXY/USDCNY/USDJPY/EURUSD）与 VIX 不进持仓权重 w_t**，降级为宏观/状态信号特征（VIX 不可现货持有；汇率非独立配置资产）。
- 可交易 15 = 8 权益 + 3 商品 + 2 加密 + 2 债券，参与 ∑wᵢ=1 与 MPT 组合优化。

### 2. USD 单一计价（plan.md §7.2）
- 所有持仓收敛到美元计价，避免 NAV 与协方差 Σ 的货币量纲错配：
  - CNY 资产（沪深300/科创50/中债10Y）`P_usd = P_cny / USDCNY`
  - JPY 资产（日经）`P_usd = P_jpy / USDJPY`
  - EUR 资产（斯托克50）`P_usd = P_eur × EURUSD`
  - HKD 资产（恒生）`P_usd = P_hkd / 7.80`（联系汇率常数，无需抓汇率）
- **新增 EURUSD**（BOC 欧元/美元交叉，国内源），用于斯托克50 折算。
- `ASSETS.yaml` 加 `ccy` 字段 + `to_usd` 折算表；`build_inputs.py`/`asset_universe.py`/`prepare_data.py` 同步。

### 3. 前向终点 = 各 WL 末阶段真实日（plan.md §7.3）
- 不硬编码 2030-12-31；逐 WL 读 `max(阶段 end_date)`。9 条 WL 末阶段均结束于 **2035-12-31**（原 2030 会截断 5 年）。
- `gen_worldline_online.py` 改 `online_end` 逐 WL 动态；已重生成 online（20 资产，至 2035）。

### 4. 同步的数据层改动（工作区，未提交）
- `asset_spec.py`/`fetch_daily_data.py`：补 EURUSD（`boc_eur` 源）；宇宙=20。
- 重抓 EURUSD（1582 行，2020-01-02~2026-07-22）；重生成 9 条 `WL*_online.csv`/`WL*_full.*`（20 资产，至 2035）。

---

## 三、当前数据状态确认（2026-07-23）

### ✅ warmup（2020-01-02 ~ 2026-07-16）：**全部汇率/信号齐全且为真实值**
20 资产 CSV 全部落盘：8 权益 + 3 商品 + 2 加密 + 2 债券 + 5 信号(DXY/USDCNY/USDJPY/EURUSD/VIX)，外加 WL 特有 KOSPI/USDKRW/JP_SEMI_EQUIP。0 NaN。

### ⚠️ 在线合成（2026-07-17 ~ 2035-12-31）：**汇率/信号"在场但部分是平的"**
合成数据由世界线 `.md` 的阶段终点表插值而来。**只有世界线表格里写明轨迹的资产才会动；没写的被 flat-hold（常数）**。逐资产世界线覆盖：

| 信号资产 | 世界线表格里有其轨迹的 WL | 在线阶段表现 |
|---|---|---|
| DXY（美元指数） | 全部 9 条 | ✅ 全动 |
| VIX | 全部 9 条 | ✅ 全动 |
| USDCNY | WL1,4,7,9（4 条） | ⚠️ 仅 4 条动，其余 5 条 flat |
| USDJPY | WL5（1 条） | ⚠️ 仅 WL5 动，其余 8 条 flat |
| EURUSD | **无任何一条** | ❌ 9 条全 flat（世界线成文时未含 EURUSD，§7 后补仅为折算） |

**所以你的判断基本正确**：2020-2026 的货币汇率是真实齐全的；2026-2035 的在线阶段里，**DXY/VIX 有轨迹，但 USDJPY（仅 WL5）、EURUSD（全无）、USDCNY（仅 4 条）是平的/缺失的**。

### 影响与建议
- EURUSD/USDJPY 在线为常数 → ① Screener 的汇率信号无信息量；② §7.2 的 USD 折算在在线阶段退化为"只换本币收益、不含汇率变动"。
- **建议补法**（待你拍板）：对世界线未给轨迹的信号汇率，用 warmup 实测的"该汇率 vs DXY"弹性，由该 WL 的 DXY 路径派生（DXY 9 条全有）。EUR 是 DXY 主成分(57.6%)，派生最稳；JPY(13.6%)/CNY(管理汇率)次之。或由你为各 WL 补汇率叙事。

---

## 四、尚未做 / 阻塞项
- ⛔ **live LLM 跑批**：AC（gpt-5.3-codex，需 OPENAI_API_KEY）+ FM（LLM 因子矿工）。本机仅有 Claude Code 会话的 Anthropic 凭据，无 OpenAI 兼容 key。dryrun 已通；live 待 key。
- ⚠️ AI-2 的全部改动**尚未 git 提交**（在工作区），process.md 之后应一并提交保存。
- FM 依赖（torch CPU + sklearn/xgboost 等）大部分已装；FactorMiner 在真实 panel 上的因子评估（MockProvider，无需 key）可作为下一步产出。
