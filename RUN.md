# RUN — 跑通 trade-agent-benchmark 的操作手册

> 本机环境：Ubuntu 26.04 + Clash TUN 代理；Python 3.14 via uv venv (`.venv/`)。
> 最后更新：2026-07-21

---

## 0. 一句话现状

- ✅ **数据链路全通**：19 基准资产 warmup 真实日频 (2020-01-02 ~ 2026-07-16) + 9 条世界线在线合成日频 (2026-07-17 ~ 2030-12-31)。
- ✅ **dryrun 全链路验证**：`build_inputs` → `walk_forward --mode dryrun` 游标/防穿越/每月新闻逻辑通过。
- ⛔ **live LLM 运行需 API Key**：AlphaCrafter (Trader/Miner/Screener) 与 FactorMiner (LLM 因子矿工) 都要 OpenAI 兼容 LLM；本机未配。提供 key 后即可 `--mode ac/fm/both` 实跑（见 §4）。

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

把 `wordline-simple/wordline1..9.md` 每阶段资产终点 → 插值日频 close → 拼到 warmup 形成完整 2020-2030 面板。

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

# 4.2 dryrun（无 LLM，验证游标/防穿越/每月新闻）
$VENV -m scheduler.walk_forward --session wl1 --mode dryrun

# 4.3 live（需 LLM key）
$VENV -m scheduler.walk_forward --session wl1 --mode both --fm-freq monthly
```

### 4a. LLM API Key（live 运行的唯一前置）

AlphaCrafter：`AlphaCrafter/.env`（由 `.env.example`）：
```
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
```
模型在 `alphacrafter/config.yaml`（默认 `gpt-5.3-codex`）+ `sandbox/<session>/config/models.json`。
FactorMiner：LLM 接口见 `FactorMiner/` 配置（`factorminer/agent/llm_interface.py`）。

> 本机当前只有 Claude Code 会话的 Anthropic 凭据（`ANTHROPIC_*`），**无 OpenAI 兼容 key**。提供任一 OpenAI 兼容端点（含可指向 Anthropic 的网关）即可实跑。日频×4.5年×19资产×9世界线×两框架的 LLM 调用量巨大，建议先用 `--limit` 小样本试跑。

### 4b. 摩擦（已内置，无需再动）

- AC：`sim/exchange_a.py` / `exchange_us.py` 单边 **1bp 佣金 + 2bp 滑点 = 3bps**（买卖对称，覆盖开/平/做空/部分成交）。
- FM：`configs/default.yaml` `execution.cost_bps: 3.0`，`portfolio.py` 按 `cost_bps/10000×换手` 扣因子净收益；`admission.turnover_penalty: 0.05`。

---

## 5. 关键设计决策（偏离原 plan，已记录于 agent-framework/plan.md §11.2）

1. warmup 用**真实抓取价**；在线用世界线**合成价**（re-anchor 到真实价）。两者在 2026-07-16 平滑衔接。
2. 数据源以**国内 sina/BOC/akshare 为主**（本机出口 IP 被 Yahoo/eastmoney 限流），Yahoo 仅兜底且需节点轮换 + 短 UA。
3. 单边摩擦统一 3bps，已在交易所/适应度层落地（非 LLM prompt 层）。
