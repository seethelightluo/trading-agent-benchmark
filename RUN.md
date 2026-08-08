# RUN — 跑通 trade-agent-benchmark 的操作手册

> 本机环境：Ubuntu 26.04 + Clash TUN 代理；Python 3.14 via uv venv (`.venv/`)。
> 最后更新：2026-07-25

---

## 0. 一句话现状

- ✅ **数据链路全通**：20 项（15 可交易 + 5 只读信号）。原始抓取数据含 2026-07-16，但 benchmark 共享研究期严格截止 **2026-07-15**；**2026-07-16 是 1M 全现金账户的首个前向执行日**。7 月 16 日为九线共同锚定 bar，7 月 17 日起情景路径分化，一直模拟到未来的 2035-12-31。
- ✅ **统一机制**：AC 与 FM 都只做一次九线共享历史热身；每条 WL 独立持久化账户、因子库和记忆。Agent 每 10 个交易日研究/决策一次，区间内逐日本地撮合和估值。
- ✅ **真实工具协议已验证**：OpenAI 兼容端点的 `gpt-5.6-terra` 能驱动 AC 原生 function calling。API URL 必须带 `/v1`。
- ⚠️ **live 凭证不入库**：临时导出环境变量或填写被 gitignore 的 `.env`；调度器发现空凭证会在请求前退出。

---

## 1. 环境

```bash
cd /home/lxx/trade-agent-benchmark
uv venv .venv                                  # 已建
uv pip install pandas requests akshare pyarrow pyyaml numpy \
    --python .venv/bin/python                  # 已装；AC/FM 各自依赖见其 setup.py/pyproject.toml
```

AC 还需 `openai python-dotenv cvxpy`；FM 需 `click xgboost gplearn`（live 前再装）。

---

## 2. 抓 warmup 真实数据（data-prepare/）

```bash
.venv/bin/python data-prepare/fetch_daily_data.py            # 复用已落盘 CSV（幂等）
.venv/bin/python data-prepare/fetch_daily_data.py --force    # 全量重抓
.venv/bin/python data-prepare/make_panel.py                  # 仅重建 panel.parquet/csv
```

**数据源可靠性（本机 Clash TUN 实测，关键经验）**：

| 源 | 可用性 | 覆盖 | 备注 |
|---|---|---|---|
| sina/163 (`stock_zh_index_daily` / `index_us_stock_sina` / `stock_hk_index_daily_sina` / `futures_foreign_hist`) | ✅ 稳定 | A股/港股/美股指数、外盘期货(GC/HG/CL) | 国内直连 |
| akshare `bond_zh_us_rate` / `macro_global_sox_index` | ✅ 稳定 | 中美10Y国债收益率、SOX | 国内 |
| BOC `currency_boc_sina` | ✅ 稳定 | USDCNY/USDJPY/USDKRW 交叉 + DXY 6成分篮子公式 | 国内 |
| Binance klines | ✅ 稳定 | BTC/ETH | 国际站 Clash 下正常 |
| Yahoo chart | ⚠️ 需技巧 | VIX/N225/SX5E/KOSPI/JP_SEMI、DXY/forex 兜底 | 见下 |
| eastmoney `_em` | ⚠️ 间歇 RemoteDisconnected | 仅兜底 | 重试 |

**Yahoo 抓取的两个坑（已内置解决）**：
1. 默认 TW 出口节点被 Yahoo rate-limit → `--rotate-clash` 逐资产换 KR/TW 节点（每 IP 仅 1 次请求）。
2. **长 Chrome UA 被 Yahoo 限流，短 UA `Mozilla/5.0` 通过**（`f_yahoo` 已用短 UA）。

```bash
# Yahoo 资产补抓（自动轮换节点并在结束后恢复原节点）：
.venv/bin/python data-prepare/fetch_daily_data.py --only VIX,N225,SX5E,KOSPI,JP_SEMI_EQUIP --rotate-clash --force
```

产物 `data-prepare/asset-daily-data/`：每资产 CSV + `all_close_wide.csv` + `COPPER_USD_PER_TON.csv` + `panel.csv`(parquet) + `COVERAGE.md`（含 vs 2026-07-16 基线比对）。

---

## 3. 合成在线世界线数据（data-prepare/online-worldline/）

```bash
.venv/bin/python data-prepare/gen_worldline_online.py          # 9 条世界线，默认 re-anchor
.venv/bin/python data-prepare/gen_worldline_online.py --only 1,3,5
```

把 `wordline-simple/wordline1..9.md` 每阶段资产终点 → 插值日频 close → 拼到历史段形成完整 2020-2035 面板。

**锚点策略（重要）**：warmup 真实价 (2026-07-16) 与世界线「估计基线」差异大（SOX 真~11700 vs 估 5800；NDX 真~28000 vs 估 20500；CN10Y 真~1.74% vs 估 2.20%）。默认 **re-anchor**：按世界线**相对路径**锚到真实价（价格 log-linear 缩放、收益率/VIX 线性平移），消除边界断层；`--no-reanchor` 用绝对价（有断层，仅对照）。详见 `WAYPOINTS.md`。

产物：`WL1..9_online.csv`（合成未来）+ `WL1..9_full.parquet/csv`（warmup+online，可再生，已 gitignore）。

---

## 4. 跑前向走步（agent-framework/）

```bash
cd agent-framework
VENV=/home/lxx/trade-agent-benchmark/.venv/bin/python

# 4.1 生成某世界线的 AC session + FM 面板（warmup-only 用 panel.parquet；含在线用 WLx_full.parquet）
$VENV -m adapters.build_inputs --panel ../data-prepare/online-worldline/WL1_full.parquet \
    --assets ASSETS.yaml --ac-session wl1 --fm-dir FactorMiner/data

# 4.2 两框架共享历史热身（九条 WL 只跑各一次；不进入前向撮合）
$VENV -m scheduler.run_pipeline --only 1 --mode both --warmup-only \
  --max-attempts 1

# 4.3 AC：共享热身 + WL1 首块 + 2 个专属 cycle 真冒烟
$VENV -m scheduler.run_pipeline --only 1 --mode ac --max-cycles 2 \
  --max-attempts 1 --state results/ac_smoke_state.json

# 4.4 FM：共享热身 + 2 个十日块；第二块追加可见数据 Ralph 迭代
$VENV -m scheduler.run_pipeline --only 1 --mode fm --fm-max-windows 2 \
  --fm-online-iterations 1 --max-attempts 1 \
  --state results/fm_smoke_state.json

# 4.5 全量（必须先通过上述真冒烟）
setsid nohup bash scheduler/run_all.sh --mode both --cadence 10 \
  > results/run_all.log 2>&1 &
tail -f results/run_pipeline.log
```

### 4a. LLM API Key（live 运行的唯一前置）

AlphaCrafter：`AlphaCrafter/.env`（由 `.env.example`）：
```
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
```
模型在 `AlphaCrafter/alphacrafter/config.yaml` 与 session 的 `config/models.json` 中配置，当前为 `gpt-5.6-terra`。
FactorMiner：LLM 接口见 `FactorMiner/` 配置（`factorminer/agent/llm_interface.py`）。

> 不要把 key 写进命令历史、文档、日志或 git。调度器只检查非空，不打印密钥。

### 4b. 资金、因子、调仓与成本口径

- 本 benchmark 的目标是比较 Agent 在九条不同世界线中的自主表现，不是利用世界线数据训练/调参以提高分数。九条 WL 全部是并列前向评估环境。
- 2026-07-16 前冻结人工配置。之后世界线数据只能随时间游标逐步揭示；Agent 可以在已揭示历史内自主回测和更新策略，但禁止人工查看世界线结果后改参重跑、挑 WL 调参或跨 WL 共享在线经验。
- 2020-01-01～2026-07-15 只做研究：允许积累因子、memory、组合和策略，但账户资本冻结，不建立历史持仓。
- 2026-07-16 以 **1,000,000 USD-equivalent 全现金、空持仓、空订单**进入前向模拟；15 项只是候选交易宇宙，不会默认平均买满。策略可以只持有少数资产或继续全现金。
- AC 共享策略先本地执行首个 10 日块；之后每 10 个交易日运行 Miner/Screener/Trader。FM 从共享 checkpoint 为每条 WL 克隆独立 library/memory，随后每 10 日只用当时可见数据追加 Ralph 迭代并生成组合。
- 研究库可超过 10 个因子，但两个框架进入活跃组合的因子最多 10 个。
- 单边摩擦为 3 bps。最低增量边际门槛为往返 6 bps；预测收益不足时允许不交易或保持原持仓。
- 每日数据推进不等于每日 LLM 调用：区间内撮合/估值完全本地，token 主要消耗在每 10 日的 Agent 决策，显著低于逐日运行完整 Agent cycle。

### 4b. 摩擦（已内置，无需再动）

- AC：`sim/exchange_a.py` / `exchange_us.py` 一次资产转移 **1bp 佣金 + 2bp 滑点 = 3bps**（总共）。
- FM：`configs/default.yaml` `execution.cost_bps: 3.0`，`portfolio.py` 按 `cost_bps/10000×换手` 扣因子净收益；`admission.turnover_penalty: 0.05`。

---

## 5. 关键设计决策（偏离原 plan，已记录于 agent-framework/plan.md §11.2）

1. warmup 用**真实抓取价**；在线用世界线**合成价**（re-anchor 到真实价）。两者在 2026-07-16 平滑衔接。
2. 数据源以**国内 sina/BOC/akshare 为主**（本机出口 IP 被 Yahoo/eastmoney 限流），Yahoo 仅兜底且需节点轮换 + 短 UA。
3. 单边摩擦统一 3bps，已在交易所/适应度层落地（非 LLM prompt 层）。
