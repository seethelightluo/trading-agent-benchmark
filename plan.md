# plan.md — 世界线基准数据宇宙与 Agent 执行计划

> 创建：2026-07-20　|　目标来源：`data-prepare/wordline-simple/wordline1-9.md` + 统一基线价格表

---

## 一、总目标

为 9 条世界线准备基准数据宇宙：**15 项可交易资产 + 5 项只读指数/宏观参考**。真实历史严格截至
**2026-07-15**；从 **2026-07-16** 起的数据均为虚构世界线数据。该数据用于因子挖掘、回测与 Agent
决策，资产定义以 `data-prepare/asset_spec.py` 和 `agent-framework/ASSETS.yaml` 为单一事实源。

## 二、基准数据宇宙（15 可交易 + 5 只读参考）

| 角色 | 类别 | 数量 | 资产 ID |
|---|---|---:|---|
| 可交易 | 权益 | 8 | `000300.SH`, `SPX`, `HSI`, `N225`, `SX5E`, `000688.SH`, `SOX`, `NDX` |
| 可交易 | 商品 | 3 | `XAU`, `COPPER`, `WTI` |
| 可交易 | 加密 | 2 | `BTC`, `ETH` |
| 可交易 | 债券 | 2 | `US10Y`, `CN10Y` |
| 只读参考 | 指数/宏观/汇率 | 5 | `DXY`, `USDCNY`, `USDJPY`, `EURUSD`, `VIX` |

只读参考仅供 Agent 识别宏观状态或完成 USD 折算，不进入持仓权重向量。仓库仍保留旧世界线抓取产生的
`KOSPI`、`USDKRW`、`JP_SEMI_EQUIP` 文件用于历史兼容和审计，但它们**不属于基准 Agent 输入宇宙**。

## 三、数据源与口径

- 每个资产的当前来源优先级以 `data-prepare/asset_spec.py` 为准，抓取器按候选顺序先成先用，避免文档与代码映射漂移。
- **新浪 / 东方财富 / akshare / 中国银行**：覆盖权益、商品、债券、DXY 与汇率，是当前国内可达环境下的主要来源。
- **Binance**：提供 BTC/ETH 日线 O/H/L/C/Volume。
- **Yahoo Finance / CBOE**：作为国际源或长尾兜底；请求失败时由抓取器执行退避和来源切换。

## 四、输出结构

```
data-prepare/asset-daily-data/
├── <ASSET>.csv          # 每个资产一文件：date,open,high,low,close,volume(,adjclose)
├── all_close_wide.csv   # 日期(并集) × 资产收盘 宽表（非交易日为空）
├── COPPER_USD_PER_TON.csv  # 铜 USD/吨（HG=F × 2204.62262）
└── COVERAGE.md          # 每资产：起止日、行数、缺失天数、单位、来源
```

- 日期范围：`2020-01-01 ~ 2026-07-15`；各资产按自身交易日历（加密 365 天，股市工作日，各国节假日不同 → 宽表必有空值，正常）。
- 每资产 CSV 含各自交易日；宽表取所有日期并集。

## 五、执行步骤与进度

| # | 步骤 | 状态 |
|---|------|------|
| 0 | 安装 git / uv（uv 用官方安装器 → `~/.local/bin`） | ✅ 完成（git 2.53.0 / uv 0.11.29） |
| 1 | `git init` + `.gitignore` + user 配置 + 初始提交 | ✅ 完成（commit ea78c80，314 文件） |
| 2 | `uv venv .venv` + 装 pandas/requests/akshare/pyarrow/pyyaml | ✅ 完成 |
| 3 | 写 `data-prepare/fetch_daily_data.py`（多源 + 重试 + 宽表 + COVERAGE） | ✅ 完成（含 asset_spec/make_panel） |
| 4 | 运行抓取，产出 `asset-daily-data/`（20 基准 + 3 历史兼容产物） | ✅ 完成（Agent 真实历史截断至 2026-07-15） |
| 5 | 校验：每资产日期范围/行数/缺失，与基线 2026-07 价比对 | ✅ 完成（见 `COVERAGE.md`） |
| 6 | 提交并推送到 GitHub（SSH） | ✅ 完成（commit b086837+） |
| 7 | 合成在线世界线日频（2026-07-16 ~ 2035-12-31，逐 WL 末阶段） | ✅ 完成（`gen_worldline_online.py` + 9 条 WL，re-anchor） |
| 8 | build_inputs + walk_forward dryrun 全链路 | ✅ 完成（详见 [`RUN.md`](RUN.md)） |
| 9 | live LLM 前向跑批（ac/fm/both） | 🟡 AC 2-cycle 真烟测完成；全量待受控启动 |

## 六、注意事项

- 本机走 Clash TUN 代理：**国际站（Binance/Yahoo/FRED/astral.sh）正常**；**国内 apt 源 InRelease 过代理 SSL 会断流**（aliyun/tuna 的 .deb 仍可下，故 git 能装）。数据抓取走国际站，不受影响。
- akshare 依赖较多，安装偏慢；若失败则中债10Y 改直连 chinabond/eastmoney API。
- 宽表非交易日空值是预期行为，非缺失。

---

## 七、Agent 配置宇宙与计价基准（关键架构修订 · 2026-07-22）

> 本节细化第二节的同一套 **15 可交易 + 5 只读参考**口径，适用于 2026-07-16 之后的虚构前向持仓阶段。
> 已同步：`agent-framework/ASSETS.yaml`、`integration/asset_universe.py`、`gen_worldline_online.py`。

### 7.1 可交易持仓宇宙 = 15（汇率 / 波动率剔除为信号）

**汇率（DXY / USDCNY / USDJPY）与 VIX 不进入持仓权重向量 w_t**，降级为"宏观 / 状态信号特征"：
- VIX 是期权隐含波动率指数，**不可现货持有**（仅衍生品），塞进 MPT 协方差矩阵会量纲错配。
- 汇率是宏观传导媒介，非独立配置资产（本测试不做杠杆 FX）。

**可交易持仓（15 项，参与 ∑wᵢ = 1 与组合优化）：**

| 类别 | 数量 | 资产 |
|---|---:|---|
| 权益 | 8 | 沪深300、标普500、恒生、日经225、斯托克50、科创50、费城半导体、纳斯达克100 |
| 商品 | 3 | 黄金、铜、原油 |
| 加密 | 2 | BTC、ETH |
| 债券 | 2 | 美债10Y、中债10Y |

**宏观 / 状态信号（5 项，仅作 Agent 输入特征，不持仓）：** DXY、USDCNY、USDJPY、EURUSD、VIX
→ 供 AlphaCrafter Screener 判 Risk-On/Off；供 FactorMiner 生成跨资产因子（如 `Ts_Rank(VIX, 20)`）。其中 USDCNY/USDJPY/EURUSD 同时承担对应外币资产的 USD 折算（见 §7.2）。

> 合计 **20 = 15 可交易持仓 + 5 信号指标**。

### 7.2 统一以 USD 计价与结算（单一计价货币）

所有持仓资产统一收敛到**美元计价**，避免组合 NAV 与协方差矩阵 Σ 的货币量纲错配；日收益 R_t 已含"价格变动 + 汇率变动"的综合回报。

- **原生 USD（不变）**：SPX、NDX、SOX、XAU、Copper、WTI、BTC、ETH、US10Y
- **CNY 资产**：`P_USD(t) = P_CNY(t) / USDCNY(t)` —— 沪深300、科创50、中债10Y
- **JPY 资产**：`P_USD(t) = P_JPY(t) / USDJPY(t)` —— 日经225
- **EUR 资产**：`P_USD(t) = P_EUR(t) × EURUSD(t)` —— 斯托克50（注意 EURUSD 报价为「USD per EUR」，故相乘，与 CNY/JPY 相除相反）
- **HKD 资产**：恒生—— HKD 与美元**联系汇率（peg ≈ 7.80，区间 7.75–7.85）**，用常数折算 `P_USD = P_HKD / 7.80`，**无需抓取汇率**（误差 <1.3%，可忽略）。

### 7.3 前向结束时间 = 世界线末阶段真实结束日（不硬编码 2030）

融合 / 前向终止日 **不固定 2030-12-31**，而取**当前世界线最后一个阶段的真实结束日期**。
- 例：WL1 末阶段「新均衡形成（2030–2031）」→ 终点落在 2031，硬编码 2030-12-31 会**截断**该 WL。
- 不同世界线终点不同 → **逐 WL 读取 `wordlineN.md` 末阶段日期**，作为该 WL 前向窗口终点。
- 现有 `ASSETS.yaml`(`online_end`)、`integration/asset_universe.py`(`FORWARD_END`)、`gen_worldline_online.py` 中的 `2030-12-31` 需改为按 WL 动态读取。

### 7.4 待同步项（2026-07-22 已完成）

- [x] **补抓 EURUSD**（斯托克50 的 EUR→USD 折算，BOC 欧元/美元中间价比，国内源免 Yahoo）；恒生用 HKD-USD 联系汇率常数 ≈7.80，无需抓取。
- [x] `ASSETS.yaml` / `asset_universe.py` / `asset_spec.py` / `build_inputs.py`：宇宙拆为 **15 可交易 + 5 信号 = 20**，信号项（DXY/USDCNY/USDJPY/EURUSD/VIX）不进权重向量（AC watch_list 仅 15，信号入 index_data）。
- [x] 前向终点改为**逐 WL 动态**读取 `max(阶段 end_date)`（9 条 WL 均至 2035-12-31，原 2030-12-31 会截断 5 年）；`gen_worldline_online.py` 已重生成。

---

## 八、Agent 执行架构与实现（2026-07-25）

> 数据宇宙与计价口径见 §二/§七；本节描述两框架如何**消费**该数据走步前向。
> 设计目标：**公平**（两框架见同一研究历史）· **防穿越**（只用 ≤t 数据）· **抗断**（长任务可续跑）· **可审计**（指纹 + 状态持久化）。
> 代码主入口：`agent-framework/scheduler/run_pipeline.py`（断电可恢复运行器）。

### 8.1 两框架角色（核心对照 = news 是否有用）

| 框架 | 机制 | 信息优势 | 决策节奏 |
|---|---|---|---|
| **AlphaCrafter (AC)** | 3×Miner → Screener → Trader 多智能体轮转 | 吃 macro+event **news**（含内幕抢跑对齐） | 每 cycle 推进 10 交易日 |
| **FactorMiner (FM)** | RalphLoop 公式化因子自进化 + IC 加权组合 | 纯价量，**不吃 news** | 每 10 交易日重算 + 本地每日盯盘 |

### 8.2 共享 warm-up + 逐 WL 播种（核心架构）

9 条世界线的真实研究历史在 ≤2026-07-15 段**字节一致**，故每框架只做**一次冻结资金的研究 warm-up**，再把不可变成果播种到 9 条独立世界线账户：

- **AC warm-up**（session `ws1`，`AC_WARMUP_ONLY=1`，`max_cycles=1`）：跑 1 个完整 Miner×3 + Screener + Trader cycle；资金冻结（100M 全现金、0 持仓）、step 工具禁用、不前进日期；产出 `workspace/strategy.py`（带 `@register_hook`）+ `workspace/factors/*.json`。
- **FM warm-up**（`results/fm/shared_warmup/`）：Ralph loop 在 ≤2026-07-15 切片上挖因子库 + checkpoint + 经验记忆，再 `combine` 出 IC 加权组合。
- **指纹闸（防漂移）**：`history_digest + 研究代码 sha + 配置 sha + assets sha` → 任一改动 → 归档旧 session、强制重跑，保证 warm-up 永远匹配当前代码。
- **逐 WL 播种**：`seed_worldline_workspace`（AC）把 ws1 的 workspace 拷进 `sandbox/wlN/`，`execute_seeded_first_block` 用共享 strategy 本地跑前 10 天，再 `--resume` 进入 WL 专属 cycle；FM 则 `seed_fm_online_state` 克隆 library/checkpoint/memory 到每条 WL，每 10 天做 1 次 Ralph 小更新 + `run_forward` 确定性撮合。
- **收益**：省 9× 重复 warm-up 的 LLM 成本；保证两框架研究输入完全一致（公平性基准）。

### 8.3 节奏、摩擦、资金

- **决策节奏 = 10 交易日**（`ASSETS.yaml: decision_cadence_trading_days`）。AC：`agent/toolkit/step.py` 读 `AC_CADENCE_DAYS` 强制每 cycle 推进 N 天。FM：`scheduler/fm_walk_forward.py: run_forward` 在 `idx % cadence == 0` 重算因子权重；**日间盯盘本地确定性、不调 LLM**。
- **统一摩擦 = 单边 3bps**（1bp 佣金 + 2bp 滑点）：AC 的 `sim/exchange_a.py`/`sim/exchange_us.py` 在 `executed_price` 上叠加 `slippage_rate=0.0002`；FM 的 `run_forward` 按 `turnover × cost_bps` 扣成本，并有 `min_round_trip_edge_bps=6` 门槛（预期收益不覆盖双边成本则不交易）。
- **初始资金 100M USD**，warm-up 期间冻结；`baseline_date=2026-07-16` 为首笔前向执行日。

### 8.4 沙箱（防作弊 / 防穿越）

- **两框架均移除 web-search**；AC `agent/toolkit/shell.py` 给子进程注入黑洞代理（`HTTP_PROXY=http://127.0.0.1:1`、`NO_PROXY=""`），禁止联网偷看未来。
- **防穿越**：walk-forward 游标——每个决策日 t 只喂 ≤t 数据；FM `_slice_fm_panel`/`slice_panel` 切片后断言 `datetime.max() <= cutoff`；AC news 经 `build_inputs --stage-news` 对齐（news_date 滞后 leak 时点，价格先于 news）。

### 8.5 韧性（长任务抗断电 / 抗崩溃）

- **递增退避** `EscalatingBackoff [0, 60, 600, 3600]s`，**任一成功立即重置**（匹配配额刷新模型）；全 pipeline 共享一条退避链。
- **逐 WL 原子状态** `results/run_state.json`（`.tmp` + `replace` 原子写）→ 断电重启跳过已完成 WL；AC `--resume` cycle 级续跑。
- **卡死检测**：`run_ac_wl` 若 `rc==0` 但日期游标未推进（且非 warmup_only）→ 判失败（rc=2）→ 退避重试（catches「agent loop ran but never called step」）。
- **完成判定**：`ac_session_complete` 只认 `date.json.simulation_complete=true`（走完最后在线日），**max_cycles 到了不算 WL 完成**——避免烟测被误标为全量完成。
- **三层守护**：`setsid nohup run_all.sh`（脱离终端）+ 内层 while（崩溃 10s 重拉）+ systemd `run_pipeline.service`（断电重启）。

### 8.6 模型与端点（已锁定，勿擅改）

- **统一 `glm-5.2`**（智谱 BigModel OpenAI 兼容端点 `https://open.bigmodel.cn/api/coding/paas/v4`）：AC `config.yaml` miner/screener/trader 三处 `model.code` + FM `fm_live.yaml` `model` 必须三处一致；密钥在 gitignored `AlphaCrafter/.env`（`run_all.sh` 用 `set -a` 导出）。**用户已锁定，改前须征得同意。**
- FM `OpenAIProvider` 读 `OPENAI_API_URL`/`OPENAI_BASE_URL`；`models.json` 已加 `glm-5.2` 条目（AC `_load_model_config` 找不到会 ValueError）；`.env` 被 root `.gitignore`（第 30-31 行）覆盖、未追踪。
- `glm-5.2` 为**推理模型**（返回带 `reasoning_content`），单调用 token 含推理、偏慢——成本估算见 §8.8。

### 8.7 评估口径

- **真值** = 9 条世界线 2026-07-16→2035-12-31 的虚构未来行情（GBB 噪声 σ=warmup 实现波动率、端点归零命中阶段终点；price-leads-news 内幕抢跑；DXY-β 派生缺失汇率）。生成方法见 `data-prepare/process.md`。
- **核心问题**：扣 3bps 摩擦后，LLM Agent 能否盈利？两框架 NAV 曲线 + 风险调整收益对照；AC vs FM = news 价值的最直接 A/B。

### 8.8 成本估算与可行域（2026-07-25，基于 ws1 实测外推）

> 口径：决策点 = 前向 2468 交易日 ÷ 10 ≈ **246/WL × 9 WL = 2214 个**。
> AC 每 cycle 5 相位（3 Miner 并发 + Screener + Trader，各最多 25 迭代）**全调 LLM**；FM 唯一花 API 的是 `mine`（`combine`/`run_forward` 确定性、免费）。
> 基线取自 ws1 12:45 真实 warm-up cycle 实测：**单 cycle 27 次调用、~181k token**（input 主导 170k / output 11k）。

**单决策点开销**

| 框架 | 每点 LLM 调用 | token | 说明 |
|---|---|---|---|
| AC | ~27（warm-up 实测）~ 45（前向 screener/trader 多跑选/回测/注册） | ~180k–300k | 5 相位全 LLM；glm-5.2 推理模型，output 含 reasoning |
| FM（实时挖，默认） | 1 Ralph 迭代/窗口 | ~5k | `mine --resume-checkpoint` 追加 1 次；combine/forward 免费 |

**全量估算（9 WL × 246 决策点）**

| 框架 | warm-up（共享 1×） | 前向（9 WL） | **总量级** |
|---|---|---|---|
| **AC** | 1 cycle ≈ 27 调用 / 181k tok | 2214 cycle × 27–45 | **~60k–100k 调用 / ~4.5亿–7亿 token** |
| **FM（实时挖）** | mine ≈ 10–100 调用（target=110, batch=40, 封顶 200 迭代）/ ~150k tok | 2214 窗 × 1 iter ≈ 2.2k 调用 / ~11M tok | **~2.3k 调用 / ~11M token** |

**结论**

1. **warm-up（2020-2026）是小头**：AC 占 1/2214 ≈ 0.045%；FM 占 ~1.3%。两框架 warm-up 各只跑 1 次共享，前向是 9 WL × 246 点。
2. **FM ≪ AC**：FM 实时挖的调用量约为 AC 的 **1/30–1/60**、token 约 **1/40–1/64**。根因：AC 每 cycle 5 个 LLM 相位；FM 只 `mine` 调 LLM。

**FM 实时挖矿机制（已确认 = 所需）**：从第 2 个决策点起，每 10 个交易日用**截至当天的世界线数据**（expanding window、防穿越）追加 1 次 Ralph 迭代进化因子库 → `combine` 选 top-10 + IC 加权 → 确定性盯盘 10 天。即"实时根据虚拟世界线数据更新因子"。旋钮 `--fm-online-iterations`（默认 1）。

**⚠️ 可行性提示**：AC 的 ~60k–100k 调用 / ~5亿 token 对单一智谱 key 是很大体量；退避按"5h 配额刷新"设计，全量 AC 光配额等待可能拖成数天–数周挂机，且 glm-5.2 推理模型单调用慢。**建议先 `--only 1 --max-cycles 2` 小范围验通再定全量参数。**

**压 AC 开销旋钮**（任选组合）：
- `--only 1,3,5,7,9` 跑 5 条 WL → 砍 ~45%。
- `ASSETS.yaml: decision_cadence_trading_days` 10→20 → cycle 数减半。
- `online_end` 提前（如 2032）→ 按比例砍。
- `config.yaml` 三处 `max_iterations: 25→12` → 每相位迭代上限减半。
