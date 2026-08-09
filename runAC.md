# runAC.md：AlphaCrafter 运行手册（恢复版）

> 本文件只恢复 Codex 记忆中已经确认的 AC 参数、运行过程和断点约定；本次另加对 `WL-data-final` 的当前数据复现审计。没有把历史短 smoke 或 API 协议探测写成完整 AC 成功证据。

## 1. 数据输入与复现审计

AC 的权威输入是最终快照，不再使用旧的 `data-prepare/online-worldline/`：

```text
WL-data-final/
├── panels/WL1_full.parquet ... WL9_full.parquet
├── news/WL1_stage_news.json ... WL9_stage_news.json
├── warmup/panel.csv
├── wordline-md/wordline1.md ... wordline9.md
└── manifest.json
```

当前审计结果：

- 9 个面板均为 `83,347` 行、20 个资产、无 `(asset_id, date)` 重复键，日期范围为 `2020-01-01` 至 `2035-12-31`。
- 使用 `warmup/panel.csv` 与 `wordline-md/wordline1-8.md` 临时重跑最终包脚本，WL1–8 的面板与现有快照一致到浮点舍入误差；新闻 JSON 完全一致。算法本身是确定性的。
- 直接执行 `WL-data-final/scripts/gen_worldline_online.py` 当前不能复现，因为生成脚本仍默认查找 `scripts/asset-daily-data/`、`scripts/wordline-simple/` 和 `scripts/online-worldline/`；AC 调度器已单独固定读取最终快照的 `panels/` 与 `news/`，不会调用这条旧生成路径。
- WL9 是最终扩展版：现有 `wordline9.md` 的阶段三标题没有可被当前解析器识别的区间终点，快照额外保留了 `2030-12-31` 锚点。因此 WL9 必须直接使用 `panels/WL9_full.parquet` 与 `news/WL9_stage_news.json`，不能宣称由当前 wordline 源文件完全重建。
- 生成包没有保存 `generation_meta.json`，也没有为每个最终面板保存完整 SHA256 manifest；`manifest.json` 中的 `8410ae8b...` 是共享 warmup 指纹，不是 9 个完整面板的逐文件校验值。

已发现的输入质量边界：

- `CN10Y`、`DXY`、`EURUSD`、`SOX`、`US10Y`、`USDCNY`、`USDJPY` 的在线合成 high/low 为空。原因是 warmup 末 60 日 high=low 时，`warmup_stats()` 对零振幅替换为 NaN 后仍得到 NaN 的 `rng`，没有回落到默认范围。AC 的 open/close 路径仍在，但依赖 high/low 的撮合或回测不能把这些字段当作完整 OHLC。
- 部分阶段终点落在周末，例如 WL7 的 `2029-06-30`、WL8/WL9 的 `2033-12-31`；新闻 JSON 保留这些 stage_end，但工作日面板没有对应交易 bar。使用时应把它视为阶段锚点日期，而不是可成交日期。

因此，`WL-data-final` 当前适合作为 AC 的权威实验快照；它不是一个“默认命令即可从零重建、且所有 OHLC 字段完整”的自包含生成包。正式 AC 前应修正生成脚本路径、WL9 扩展锚点记录、`rng` fallback，并补齐逐文件 manifest。

## 2. 固定实验契约

| 项目 | 当前口径 |
|---|---|
| 初始资本 | `1_000_000` USD-equivalent |
| 可交易资产 | 15 个；5 个 signal 资产仅供观察，不进入持仓 |
| 研究截止 | `2026-07-15`；`2026-07-16` 是首个前向执行日 |
| 在线终点 | `2035-12-31` |
| 决策频次 | 每 10 个交易日一个 AC cycle |
| 共享 warmup | `40` 个 AC cycle；每个 cycle 由 3 个并行 Miner + 1 个 Screener + 1 个 Trader 组成，共 5 个 agent 步骤 |
| 因子准入 | `abs(IC) >= 0.007`、`abs(ICIR) >= 0.084`、库内最大 `abs(Spearman rho) < 0.5` |
| 研究/滚动因子库 | 持久化上限 30；每轮新因子准入或已有因子重评估后，按 `abs(IC)*abs(ICIR)` 从末尾淘汰到 best30 |
| 活跃因子 | 每次组合决策只从当前因子库选最多 10 个；可以少于 10 个。这个上限不是 warmup 的研究因子数上限 |
| 因子 tilt | 对选中的活跃因子按质量/IC tilt 归一化配权；允许只选部分因子，不足 10 个不补无效因子 |
| 投资 | long-only；15 个资产的目标权重非负且和为 1；允许小数份额；online cash=0 |
| 首次建仓 | 2026-07-16 首个 forward block 必须把 1M 全部配置到 15 个资产权重向量 |
| 调仓成本 | 首次建仓免费；后续仅对资产迁移总额一次收 `3 bps` |

`0.007` 与 `0.084` 是按 15 个资产相对论文 500 资产横截面规模缩放后的门槛：

```text
0.04 × sqrt((15-1)/(500-1)) ≈ 0.00669 → 0.007
0.50 × sqrt((15-1)/(500-1)) ≈ 0.08375 → 0.084
```

这里必须区分两个集合：warmup 和 online mining 可以持续研究、准入和重评估因子，但持久化库滚动裁剪为 30；每个组合时点再从这 30 个以内的当前库中选择不超过 10 个活跃因子。被末尾淘汰的 factor ID 必须同时从顶层库、checkpoint/library 和 signals 中移除，resume 不得把它们复活。

唯一合同源是 `agent-framework/ASSETS.yaml`；旧的 100M、6bps、整手交易和 online 持有现金口径不适用。

## 3. AC 运行结构

这里的 shared warmup 按实验分开，不是 Terra 与 DeepSeek 共用：Terra 的 `ws1` 只供 Terra
实验内世界线共享，DeepSeek 的 `ws1` 只供 DeepSeek 实验内世界线共享。两者不共享因子、
checkpoint、signals、ensemble 或 workflow。两份 warmup 都必须保留；在线合同变化不得修改
或重跑其中任何一份。

```text
Terra shared warmup（Terra 实验内部 9 条 WL 复用）
  2020-01-01 ~ 2026-07-15 可见历史
  40 个 AC cycle；每 cycle = Miner1/2/3 + Screener + Trader（5 步）
  Miner 研究、准入并持久化滚动因子库（上限 30）
  库超限后按质量从末尾淘汰，保留 best30
  Screener/combiner 从当前库选 <=10 个活跃因子
  对活跃因子做质量/IC tilt；可以只选部分因子
  Trader 注册策略，但不能调用 Step 推进市场
  账户保持 1M 全现金、无持仓、无订单
        |
        v
Terra 逐 WL 前向（wl1 ... wl9 独立持久化）
  从 Terra shared warmup workspace 播种
```

DeepSeek 实验使用同样的生命周期，但从 DeepSeek 自己的 shared warmup workspace 播种，
绝不读取 Terra 的 `ws1`。

```text
DeepSeek shared warmup（DeepSeek 实验内部 9 条 WL 复用）
  2020-01-01 ~ 2026-07-15 可见历史
  独立的 40-cycle 研究进度、因子库、ensemble、workflow 和 checkpoint
        |
        v
DeepSeek 逐 WL 前向（独立持久化）
  从 DeepSeek shared warmup workspace 播种
  2026-07-16 执行首个 10 日 block，首次满仓
  每 10 个交易日运行一次 Miner/Screener/Trader
  block 内逐日只做本地行情推进和 mark-to-market
  每次目标组合仅允许 15 资产、cash=0、迁移额 3bps
```

Terra 和 DeepSeek 各自使用自己的 AC session `ws1`；每个实验内部的世界线使用 `wl1` 至
`wl9`，两个实验之间绝不复用 `ws1`。warmup 指纹命中且 session、workspace、日期边界和
冻结账户都有效时，只在对应实验内复用一次，不重复消耗该实验的 LLM。每条 WL 的 workspace、
workflow log、account 和 date cursor 独立保存。

### 3.1 与 FM 的同步口径

AC 的因子生命周期按 FM 的三层结构解释：

```text
候选因子
  -> IC/ICIR/相关性准入
  -> 当前持久化因子库（最多 30，滚动淘汰质量末尾）
  -> 当前决策选择最多 10 个活跃因子
  -> 质量/IC tilt 的因子合成
  -> 15 资产目标权重与调仓
```

这与 `runFM.md` 中的 `mine -> trim -> combine --method ic-weighted --top-k 10 -> run_forward` 对齐。FM 的 warmup `target pool=110` 是挖矿阶段的候选生成目标，不是最终持久化库容量；最终库仍须在每次 trim 后不超过 30。FM 当前 forward 实现按 `abs(IC)` 排序取 top-10，保留 `sign(IC)` 方向，并按 IC 权重合成信号贡献；AC 的共同约束是相同的 top-10 上限、方向处理和归一化 tilt，同时用本 benchmark 的 `q_i` 做库淘汰/质量排序，不应把“最多 10 个活跃因子”误写成“warmup 最多研究 10 个因子”。

质量字段沿用本实验准入合同的 `q_i = abs(IC_i) * abs(ICIR_i)`：库淘汰和活跃因子排序使用 `q_i`，合成时保留 `sign(IC_i)` 的方向，并在被选中的因子之间归一化 tilt 权重。这里的因子 tilt 是资产打分的上游权重，不是把 30 个因子直接当成 30 个可交易资产；最终仍必须输出 15 个资产、cash=0 的目标向量。

AC 原生 `template_a` 中的 `FW=(.17,.15,...,.05)` 是固定的 10 项策略模板，可作为资产打分/排名的结构参考，但它没有实现动态因子库、滚动末尾淘汰或质量 tilt。正式 AC 若继续使用该模板，必须由 Screener/Trader 产物把当前活跃因子及其方向、质量权重显式注入，不能把固定 `FW` 当作 FM 同步合同的替代品。

### 3.2.1 原版 AC 全景与本世界线适配

原版 AlphaCrafter 的 `factor_screening` skill 只提供方法建议：对选中因子计算 pairwise
correlation，用 `correlation > 0.7` 识别 cluster，并可选择保留最高 ICIR、正交化或限制相关
组权重。初始 AC 代码没有一个把共同样本 signal、因子质量和滚动 library 一起强制执行的
`factor_contract`；也没有 benchmark 的 library=30、active<=10 硬容量。

原版 strategy baseline 是通用股票示例：`TOP_N=50`、目标 gross exposure=0.6、整数整手
计算，并由 Agent 直接调用 `add_order`。原版 Exchange 的通用费率是 1bp commission + 2bp
slippage（买高卖低，单边约 3bp），订单需落在当日 low/high 才成交，支持 pending、T+0、做空
和保证金。这些规则保留为原生模拟器参考，但不作为本 15 资产世界线的 online 合同；否则会
出现部分成交、现金残留、做空和直接下单绕过预测成本决策。

为了贴合当前资产世界线，两个 AC 版本统一采用以下适配计划（代码已落地的部分由同一份
实现覆盖）：

- 研究保留 Miner → Screener → Trader 流程，但在 post-Miner 处执行 `.007/.084` 准入、共同
  signal 的 `abs(Spearman rho)` 冲突筛选和 `quality=abs(IC)*abs(ICIR)`；`rho < .5` 双保留，
  `rho >= .5` 淘汰质量较低者，再滚动保留 best30。
- active ensemble 最多 10 个、可少于 10 个，按质量/IC tilt 配权并保留方向；10 是活跃上限，
  不是 warmup 最大研究数。缺少可重算 signal/provenance 的旧因子进入 quarantine。
- online 仅使用 15 个 tradable asset，5 个 signal 不持仓；long-only、fractional quantity、
  cash=0、权重和为 1。首次 `2026-07-16` 全额 `1M` 建仓；无 ensemble 时等权 `1/15`。
- Trader 只提交完整 target + forecast proposal。后续执行门槛是
  `gross_edge_bps > one_way_turnover * 3`，其中 `one_way_turnover=0.5*sum(abs(delta))`；
  `edge <= migration*3bp` 跳过。实际成本是 `NAV*migration*3/10000`，不是总资产固定 3bp，
  也不是双边固定 6bp。no-trade 研究与 proposed target 持久化，executed target/真实持仓不变。
- 15 资产 benchmark 的 `add_order` 入口硬失败，`ensure_fully_invested` 只处理明确账户损坏；
  它不能把合法 no-trade 强行变成交易。Terra 和 DeepSeek 的 factor gate、portfolio contract、
  StepTool 和 audit 字段保持一致，provider 只影响 LLM 路由和重试。

本次修正还把 AC Miner 自报的 `max_abs_library_correlation` 降级为 audit/provenance 字段，
不再在候选阶段直接拒绝高自报 rho；是否冲突必须由两份真实 signal 在共同样本上重算，并按
双方质量决定保留者。冲突记录写入 library audit 和 `evicted/*.reason.json`。

### 3.2 online 小数持仓的实现边界

在线组合不是整数股/整手交易。两个 AC 版本都已把小数数量贯通到：

- `sim/schemas/account.py` 的 `PositionData.quantity`、`available_quantity`；
- `sim/schemas/order.py` 的订单 `quantity` 和 `OrderResultSchema.executed_quantity`；
- `sim/utils/add_order.py`、`agent/toolkit/add_order.py` 的参数与 OpenAI tool schema；
- online 策略的数量计算，不再用 `int(...)` 截断；
- `persistent/account.json` 的读写链路，保留 JSON 浮点值。

online 调仓现在必须走 `sim/utils/rebalance_to_weights.py` 的原子路径：目标字典必须精确覆盖
`watch_list` 的 15 个资产，权重和为 1；它直接写入 15 个 fractional long positions、将
`available_cash` 置零，并对后续资产迁移收取 3 bps。两个 AC 版本的 `StepTool` 还会在策略
hook 后调用 `ensure_fully_invested()` 兜底，清理旧式 pending order 或现金残留，避免上一收盘价
订单因下一交易日价格区间不匹配而只成交部分资产。

因此 2026-07-16 的首次建仓可以把 1M 按 15 资产目标权重精确分配，产生小数单位；后续调仓同样按迁移名义金额计算一次 `3 bps`，不因整数舍入制造现金残留。小数只放宽 online 持仓/订单数量，不改变 15 个资产、long-only、cash=0、权重和为 1 的约束。旧的整手检查和“必须是 100 的倍数”已从两个版本的在线下单工具删除。

### 3.3 DeepSeek 原生 Responses 路由与已有因子恢复

DeepSeek AC 不再启用旧的 `AC_DEEPSEEK_CHAT_COMPAT` 本地 Python 请求/响应转换。AC 仍调用与 Luna 相同的 `client.responses.create(...)`、相同的 prompt、tool schema 和 Responses 历史；sub2api 负责协议适配：

```text
AC -> http://127.0.0.1:8080/v1/responses
   -> sub2api group ac-deepseek-paid
   -> account extra: openai_responses_mode=force_chat_completions
   -> https://opencode.ai/zen/go/v1/chat/completions
   -> sub2api Responses response/SSE bridge
```

两个 DeepSeek APIKey 账号、独立 group 和 AC 专用 client key 已在本机 sub2api 数据库配置；凭证不写入仓库，保存在 `/home/lxx/.config/alphacrafter/deepseek-sub2api.env`。`AC-deepseek/run_deepseek_ac9.sh` 默认读取该文件、使用 `deepseek-v4-flash`，不再读取 `opencode-api/keys.txt` 作为客户端入口。sub2api 的 `GET /v1/models` 已验证返回 200；加入直连规则后，Responses 请求已正确进入新 group 并由两个账号返回 200，不是本地格式转换路径。

为修复该 egress，Clash Verge 的规则 profile 和当前生效配置都加入了最高优先级 `DOMAIN-SUFFIX,opencode.ai,DIRECT`；重新加载后核心 `/rules` 已确认命中 `DIRECT`。主机模型请求返回 200，sub2api Responses 返回 `completed`，带函数工具的 Responses 请求也成功返回 `function_call`。规则备份保存在 Clash Verge 数据目录的 `backups/` 下。

DeepSeek 之前 40 cycle 的研究并非质量失败：日志中已经出现通过 `abs(IC)>=0.007`、`abs(ICIR)>=0.084` 的候选，但 Miner 没把 JSON 写入 `factors/`，导致审计记录误显示为 0 因子。由于 DeepSeek 因子不能继承 Luna，本次把 Luna 的完整 warmup 先保存到根目录 `Luna-warmup-archive-20260809/`，其中包括 6 个因子 JSON、公式/参数、全部研究脚本、ensemble、策略、审计与日志；它只作为 Luna 的可恢复证据，不作为 DeepSeek 的活动因子库。

DeepSeek 的旧 `ws1` 已被强制失效并由调度器归档，新的 warmup workspace 的 `factors/` 初始为空；调度器会使用 `deepseek-v4-flash`，经 sub2api `/v1/responses` 和两个付费账号重新完成 40 cycle warmup。只有 DeepSeek 自己写入并读回、通过 IC/ICIR/相关性门槛的 JSON 才能进入其因子库，不再把 Luna 因子复制进去。

后续 Miner 的保存补丁要求：通过门槛的候选必须立即写入并读回 `factors/<factor_id>.json`，包含 `validation.status=EFFECTIVE`、同周期 IC/ICIR 和 `max_abs_library_correlation`；只在文本中报告通过而没有可加载文件，视为保存失败，不得让 cycle 静默推进。

## 4. Terra API 与 AC 入口

API 凭证只从 `agent-framework/AlphaCrafter/.env` 或环境变量读取，不能写入日志或提交。此前已确认的协议约定是：

- OpenAI-compatible URL 必须包含 `/v1`；
- AC 使用模型名 `gpt-5.6-terra`；
- 本机 relay 负责把 Terra 请求转给上游 sub2api-codex；
- `/v1/models` 和 function-call 协议探测只能证明接口兼容，不能证明完整 AC 已完成。

AC 原生 CLI 的 session 参数是位置参数，不使用 `--session_id`：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework/AlphaCrafter/alphacrafter
PYTHONPATH=/home/lxx/trade-agent-benchmark/agent-framework/AlphaCrafter \
/home/lxx/trade-agent-benchmark/.venv/bin/python \
  main.py wl1 --config run_config.yaml --resume
```

正式运行应由 scheduler 统一设置 `AC_CADENCE_DAYS=10`、因子门槛环境变量和 `PYTHONPATH`，不要手工绕过 scheduler 改 cadence 或切换旧数据目录。

## 5. 全量 AC 持久化启动方式

以下命令启动共享 AC warmup（40 个 cycle、每 cycle 5 个 agent 步骤）；它不会进入任何 WL 前向交易：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework
mkdir -p results
setsid --wait nohup env AC_DATA_ROOT=/home/lxx/trade-agent-benchmark/WL-data-final \
  /home/lxx/trade-agent-benchmark/.venv/bin/python \
  -m scheduler.run_pipeline \
  --mode ac \
  --warmup-only \
  --cadence 10 \
  --max-cycles 40 \
  --max-attempts 0 \
  --state results/ac_shared_warmup_40_state.json \
  </dev/null > results/ac_shared_warmup_40.log 2>&1 &
echo $! > results/ac_shared_warmup_40.pid
```

约定：

- `--max-attempts 0` 是正式任务的无限重试；退避为 `0s -> 60s -> 600s -> 3600s`，成功后复位。
- `--max-cycles 40` 是共享 warmup 的 40 个研究 cycle；不是 40 个交易日，也不是 40 个活跃因子。
- 共享 warmup 只物化一次 `ws1` 数据/工作区；后续 WL 只复制该研究工作区并各自物化自己的最终面板与 stage news，不重复运行共享研究。
- 运行器只接受 `WL-data-final/panels/WL<n>_full.parquet` 和 `WL-data-final/news/WL<n>_stage_news.json`，缺失即失败，不得静默回退到旧 `online-worldline`。
- 不要使用旧的 `results/run_state.json` 或历史并发 supervisor 状态作为跳过依据；只有当前 session 的持久化 cursor、manifest 和最终交易日完成标记可信。

实时监控共享 warmup：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework
tail -F results/ac_shared_warmup_40.log
```

另一个终端查看当前 cycle、5 步完成数、进程和持久化状态：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework
watch -n 30 '
  echo "=== process ===";
  ps -o pid,etime,stat,cmd -p "$(cat results/ac_shared_warmup_40.pid 2>/dev/null)";
  echo "=== workflow ===";
  jq -r "if length == 0 then \"no workflow entries\" else \"last_cycle=\(.|map(.cycle)|max) success_entries=\([.[]|select(.success==true)]|length)\" end" AlphaCrafter/alphacrafter/sandbox/ws1/logs/workflow.json 2>/dev/null || true;
  echo "=== state ===";
  jq ".shared_warmup" results/ac_shared_warmup_40_state.json 2>/dev/null || true
'
```

## 6. 断点、重复运行和成功判定

断电或 API 临时失败后，重新执行同一启动命令。AC 的 `--resume` 从 workflow log 的最近完整 cycle 或未完成 cycle 恢复；共享 warmup 由 fingerprint/manifest 保护，WL session 由 `date.json` 和 workspace seed marker 保护。

启动前检查：

```bash
ps -ef | rg 'scheduler.run_pipeline|main.py wl|ac_shared_warmup' | rg -v 'rg ' || true
tail -n 80 results/ac_wl_data_final.log
cat results/ac_wl_data_final_state.json
```

共享 warmup 必须同时满足：

1. `ws1` workflow 包含完整 Miner/Screener/Trader cycle；
2. `ws1` 的 `run_config.yaml` 明确为 `max_cycles: 40`；
3. workspace 存在可加载的 `strategy.py`、当前滚动因子库和 active ensemble 产物；库不超过 30，active ensemble 不超过 10，且有质量/IC tilt 权重；
4. `date.json` 为 `current_date=2026-07-16`、`visible_through=2026-07-15`，未完成；
5. account 仍是初始 1M 全现金、无持仓、无订单；
6. warmup fingerprint 与当前 `ASSETS.yaml`、AC 代码和最终 WL1 历史一致。

每条 WL 必须同时满足：

1. 使用对应 `WL<n>` 最终面板和 stage news；
2. 首个 forward block 从 `2026-07-16` 开始，完成 10 个交易日边界；
3. 每次 online update 后，因子库完成 trim 且不超过 30；active ensemble 从当时的库中选择不超过 10 个，并持久化其方向、质量/IC tilt 权重；
4. online account 只有 15 个 tradable 资产，cash=0，目标权重和为 1；
5. 后续 rebalance history 记录一次资产迁移额和 3bps 成本；
6. `persistent/date.json.simulation_complete=true`，且 `visible_through` 达到 `2035-12-31` 前最后一个有效交易日。

## 7. 当前实现审计边界

当前代码已经具备：冻结 warmup、共享 workspace 播种、位置式 AC CLI、10 日 cadence、session resume、状态文件和重试退避；两个 AC 版本的 online 数量链路也已改成允许小数。`rebalance_to_weights()` 本身实现了 15 资产权重校验、首次免费、后续迁移额 3bps 和 cash=0。

### 7.1 统一 proposal / gate 合同

Trader 只能产生 proposal，统一执行层才允许改账户。proposal 至少保存
`current_weights`、`proposed_target_weights`、`executed_target_weights`、
`forecast_returns`、`factor_ids`、`horizon_days=10`、
`one_way_turnover`、`gross_edge_bps`、`decision_edge_threshold_bps=one_way_turnover*3`、
`actual_cost`、`executed` 和 `skip_reason`。计算规则与 FM 完全相同：

```text
one_way_turnover = 0.5 * sum(abs(target-current))
gross_edge_bps = 10000 * sum((target-current) * forecast_returns)
execute = initial_allocation or gross_edge_bps > one_way_turnover * 3
actual_cost = NAV * one_way_turnover * 3 / 10000
```

首次 `2026-07-16` 有 ensemble 用其质量 tilt target，无 ensemble 等权 15 资产；后续 `edge <= migration*3bp` 只持久化研究和 proposed target，不改变真实持仓。`add_order` 在 15 资产 benchmark 账户中被拒绝，`ensure_fully_invested()` 只允许明确账户损坏修复，不得覆盖合法 no-trade。

仍需持续验证：

- `factor_contract.py` 现在在每个 Miner 阶段后由 AC 主循环单点接入，负责 IC/ICIR、真实 signal 上的 pairwise 相关性、库容量 30 和滚动末尾淘汰；缺少 signal artifact 的旧因子进入 quarantine，不再静默视为 rho=0；Screener 只负责 active ensemble <=10 和质量 tilt，不重复做库排序；
- AC 原生固定 `FW` 模板不是动态质量 tilt；正式策略仍需消费当前 `factor_ensemble.json`，不能把固定 `FW` 当作 FM 同步合同的替代品；
- 两个 AC 的 StepTool 已在非 warmup 的 15-资产 session 中执行 `ensure_fully_invested()`；旧式 Agent strategy 的订单会被清理并按最近有效目标权重修复，仍需在每个 online block 审计 15 个 positions、cash=0 和 `rebalance_history`；
- 对 warmup 中仍调用原生 `add_order` 的历史 strategy，播种器只在新的 online workspace 安装 proposal/gate adapter，不修改 warmup 原件；两份 AC 的 A/US Exchange 保存器会保留 `portfolio_contract_version`、proposal、executed target、成本和 no-trade audit 等扩展字段，避免 post-tick 被原生 AccountSchema 抹掉；
- Trader 指令和执行 helper 现在要求 proposal/gate/no-trade 持久化；旧 workspace 中固定 FW 策略仍是历史 artifact，不能作为新 fingerprint 的正式执行策略；
- scheduler 的数据入口已切换到 `WL-data-final`，通过 `AC_DATA_ROOT`/`panels`/`news` 单一解析点读取；最终数据包自身的重建/OHLC 缺口仍按第 1 节记录，不能在运行时静默修补。

在这些项目完成前，不能把“AC warmup 通过”或“API function-call 通过”当成 9 条 WL 全量 AC 完成。

## 8. 相关入口

- `agent-framework/scheduler/run_pipeline.py`
- `agent-framework/scheduler/ac_shared_warmup.py`
- `agent-framework/AlphaCrafter/alphacrafter/main.py`
- `agent-framework/AlphaCrafter/alphacrafter/agent/toolkit/step.py`
- `agent-framework/AlphaCrafter/alphacrafter/sim/utils/rebalance_to_weights.py`
- `agent-framework/AlphaCrafter/alphacrafter/factor_contract.py`
- `agent-framework/ASSETS.yaml`
- `WL-data-final/docs/GENERATION_PROVENANCE.md`
- `runFM.md`

## 9. 已放弃的旧 online 阶段

此前的 Luna/Terra 三 WL online 和 DeepSeek online 使用了旧调仓语义。由于现在
proposal、migration gate 和 fractional full-investment 合同已经改变，旧 online 只保留
为历史结果，不再使用 runner 恢复或续跑：

```text
旧 runner：scheduler.run_ac_luna_3
旧状态：agent-framework/results/ac_luna_3wl/
处理：保留，不 resume，不删除
```

旧状态中的 initialized account 若缺少
`portfolio_contract_version=ac-worldline-v2-migration-gate`，新的 AC 执行层会 fail-closed，
不会清理订单、重写账户或静默迁移持仓。

新 online 只能在对应实验自己的 shared warmup 完成或确认可恢复后，复制出全新的 workspace、
account、workflow 和 code/contract fingerprint，再做 1 block smoke；不得从旧 online account
继续运行。

2026-08-09 Terra/Luna 已按此流程启动新的 `ac_luna_3wl_v4`：使用 `terra_v4_wl1`、
`terra_v4_wl2`、`terra_v4_wl3` 三个独立 session；旧 `ac_luna_3wl` 与失败的 v2/v3
只保留为历史证据，不 resume。三条 WL 的首个 10 日 seed block 已完成，账户均为
15 个 fractional positions、cash=0、`portfolio_contract_version=ac-worldline-v2-migration-gate`。
