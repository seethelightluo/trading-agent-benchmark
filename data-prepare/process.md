# process.md — 在线合成数据生成方法（严谨可复现）

> 子模块：`data-prepare`　|　创建：2026-07-24　|　生成脚本：`gen_worldline_online.py`
> 适用范围：2026-07-17 ~ 各世界线末阶段（9 条 WL 均至 **2035-12-31**）的 20 资产日频合成行情。
> 本文件是生成方法的**单一事实源**，确保任何人、任何时刻用同一脚本+同一输入可逐 K 线复现。

---

## 0. 一句话总览

每条世界线、每个资产：以**世界线阶段终点为不可改锚点** → 对数线性插值命中锚点（含 price-leads-news 的 leak 航点）→ 叠加**几何布朗桥（GBB）日频噪声**（σ 取自 warmup 2020–2026 实测已实现波动率，**端点处噪声归零 → 严格命中世界线每阶段数值**）→ 全程确定性、定种子、无 AI/LLM。

---

## 1. 不可改节点数据（权威锚点）

**来源**：`data-prepare/wordline-simple/wordline1..9.md` 每个阶段的资产终点表（人工设定的世界线情景值）。
**解析**：`parse_worldline()` 抽取每个阶段 `## 阶段N：标题（起 - 止）` 的：
- `end_date`：阶段结束日（由 `parse_stage_end` 从 `(起-止)` 取"止"，`tok_to_date` 转月末/半年末；如 `2028.5→2028-05-31`）。
- `rows[asset_id]`：该阶段末各资产的目标**水平值**（价格类=点位/USD，收益率类=百分数，VIX=点数）。
- 阶段一第 2 列 = 世界线估计基线 `baseline[aid]`。

**这些锚点在生成中不可改动**：
- 凡世界线**指定了某资产在某阶段**的终点 → 该 (资产, end_date) 是硬锚，生成路径**必须精确命中**（GBB 噪声在该日归零，实测偏差 0.000%）。
- 凡世界线**未指定某资产在某阶段**（资产缺席该阶段表）→ 该日非锚点，资产跨该阶段做更长 bridge 的**内部点**（有噪声，属正常内插）。
- 锚点清单机器解析后落盘到 `online-worldline/WL<n>_stage_news.json`（含 `stage_end` / `asset_endpoints`），供人工核对与下游使用。

> 注：warmup 段（≤ 2026-07-16）是**真实抓取价**（见 `asset-daily-data/`），不在本生成范围；在线段从 2026-07-16 真实收盘出发。

---

## 2. 波动率 σ 的取法（来自 2020–2026 真实数据）

`compute_realized_vols(panel)`：对 warmup 面板（`asset-daily-data/panel.parquet`，2020-01-02 ~ 2026-07-16）每资产：
- **价格类**（权益/商品/加密/汇率）：`σ = std(日 log-return)` = `np.log(close).diff().std()`。
- **线性类**（`US10Y/CN10Y/VIX`，水平值）：`σ = std(日一阶差)` = `close.diff().std()`。

**实测 σ（2026-07-16 截止的 warmup）**：

| 资产 | σ | 资产 | σ | 资产 | σ |
|---|---:|---|---:|---|---:|
| 000300.SH | 0.0119 | BTC | 0.0324 | US10Y | 0.0596 |
| 000688.SH | 0.0209 | ETH | 0.0433 | CN10Y | 0.0206 |
| SPX | 0.0129 | XAU | 0.0116 | DXY | 0.0041 |
| NDX | 0.0160 | COPPER | 0.0167 | USDCNY | 0.0022 |
| SOX | 0.0247 | WTI | 0.0332 | USDJPY | 0.0058 |
| HSI | 0.0155 | SX5E | 0.0128 | EURUSD | 0.0045 |
| N225 | 0.0145 | | | VIX | 2.1912 |

σ 仅取自**历史真实数据**，**不读 news、不用 AI/LLM**（世界线终点已编码 news 的影响，见根 `process.md` 决策）。warmup 数据更新后 σ 需重算。

---

## 3. 噪声采样：几何布朗桥（GBB）

对每段相邻锚点 `[t0,v0]→[t1,v1]`（锚点含：baseline、各 leak 航点、各阶段终点），在两者之间的交易日上叠加**离散布朗桥噪声**：

1. 取该段落在 `days`（在线交易日列表）中的索引集 `idx`，长度 `m`。
2. 抽 `m-1` 个独立增量 `eps ~ N(0, σ²)`（用下节种子）。
3. 累计 `B[0]=0, B[i]=Σ_{j<i} eps[j]`（长 `m`）。
4. 布朗桥 `bb[i] = B[i] − B[m-1]·(i/(m-1))`，满足 `bb[0]=bb[m-1]=0`（**段两端归零**）。
5. 叠加：
   - 价格类（对数空间）：`close = smooth × exp(bb)`；
   - 线性类（加性）：`close = smooth + bb`，再按下节地板防负。

**为何是"桥"而非随机游走**：桥在两端强制归零，保证**锚点（世界线终点 + leak 航点）逐个精确命中**；噪声只出现在段内部 → 既给日频波动（摩擦测试有意义），又不偏移任何世界线设定的阶段终点。

**段边界共享**：相邻段共享锚点日，该日两段噪声都为 0 → 总噪声 0 → 命中。资产缺某阶段终点时，该阶段无锚点，两边的锚点直接连成更长的一段，该阶段日期成内部点（有噪声）。

---

## 4. 种子（完全确定性、可复现）

- 每 (世界线, 资产) 一个主种子：`seed = zlib.crc32(f"{wl_num}|{aid}".encode())`。
- 每段（段号 `k`）独立子种子：`rng = np.random.default_rng(zlib.crc32(f"{seed}|{k}".encode()))`。
- `zlib.crc32` 跨平台、跨进程确定（不依赖 Python hash 随机化）→ **相同输入重生成逐 K 线一致**。
- 随机源仅 `numpy.random.default_rng`（不使用 `Math.random`/全局状态）。

---

## 5. price-leads-news（内幕抢跑，强制）

每段 `[t0,v0]→[t1,v1]` 内插入一个 **leak 航点**（`_insert_leak_waypoints`）：
- news 破裂日 `t_news = t0 + LEAD_TIME_FRAC·(t1−t0)`（默认 `0.35`）。
- leak 值 `v_leak = v0·(v1/v0)^LEAD_MOVE_FRAC`（价格类，对数空间；线性类 `v0 + LEAD_MOVE_FRAC·(v1−v0)`；默认 `0.25`）。
- 即到 news 破裂时价格已走 25%（慢 leak），剩余 75% 在 news 后加速反应 → **价格先于 news，news 绝不先于价格**。
- leak 航点同样是 GBB 的锚点 → 噪声在其上归零 → leak 比例不被噪声破坏。
- news 文本（标题=世界线阶段叙事）以 `publish_date = t_news`（滞后 leak）写入 `WL<n>_stage_news.json`，供 `build_inputs --stage-news` 注入 AC Screener。
- CLI：`--no-lead` 关；`--lead-time-frac` / `--lead-move-frac` 可调。

---

## 6. 信号汇率派生（世界线无轨迹时）

世界线表格只给部分汇率轨迹（DXY/VIX 9 条全有；USDCNY 仅 WL1/4/7/9；USDJPY 仅 WL5；EURUSD 全无）。**缺失者由 DXY 按 warmup β 派生**（`compute_fx_betas`）：

- 回归 `r_fx` 日 log-return 对 `r_DXY` 日 log-return：`β = cov(r_fx, r_DXY)/var(r_DXY)`。
- 实测 β：**EURUSD −1.063**（DXY 主成分，强负相关）、**USDJPY +0.889**、**USDCNY +0.303**（管理汇率弱传导）。
- 派生路径 `close = real_0716 × exp(β · logret_DXY_online)`，连续锚定，**继承 DXY 的 GBB 噪声**（不重复加噪）。
- 有世界线轨迹的（USDJPY@WL5、USDCNY@WL1/4/7/9）保留原轨迹+自身噪声。
- CLI：`--no-derive-fx` 关（保持 flat）。

---

## 7. re-anchor（消除 warmup↔世界线边界断层）

warmup 真实价与世界线"估计基线"差异大（SOX 真~11700/估5800 等）。默认 re-anchor（`--no-reanchor` 关）：
- 价格类：`close = real_0716 × wl_path / wl_baseline`（乘性，保留世界线相对涨跌）。
- 线性类：`close = real_0716 + (wl_path − wl_baseline)`（加性，保留 bp/点数变动）。
- GBB 噪声在 re-anchor 之后叠加（`exp(bb)`/`+bb`），不影响锚点命中。

---

## 8. 线性资产防负

线性类加性噪声可能把水平值推向负（VIX 尤甚，σ=2.19）。叠加后按下限钳位（`LINEAR_FLOOR`）：
- `VIX ≥ 9.0`（历史地板）、`US10Y ≥ 0.05`、`CN10Y ≥ 0.05`。
- 钳位只影响段内异常下冲日；锚点日噪声=0、值远高于地板，不受影响。
- 价格类 `exp(bb)` 天然为正，无需钳位。

---

## 9. 可复现命令

```bash
cd /home/lxx/trade-agent-benchmark
# 前置：warmup 真实数据已抓全（asset-daily-data/panel.parquet，20 资产，≤2026-07-16）
.venv/bin/python data-prepare/gen_worldline_online.py           # 默认：re-anchor + FX派生 + leak + GBB噪声
.venv/bin/python data-prepare/gen_worldline_online.py --only 1  # 单条 WL
# 关闭某项：--no-reanchor / --no-derive-fx / --no-lead / --no-noise
```

**产物**（`data-prepare/online-worldline/`，可再生、已 git 跟踪）：
- `WL<n>_online.csv`：在线阶段长表（date,asset_id,open,high,low,close,volume,amount）。
- `WL<n>_full.parquet/csv`：warmup（真实）+ 在线（合成）完整面板（gitignore，可再生）。
- `WL<n>_stage_news.json`：阶段终点 + news_date + 叙事（锚点清单 + 注入用 news）。
- `WAYPOINTS.md`：各 WL 解析的阶段终点 + 抽样首末（人工核对）。

**校验**（已通过，见 commit dc86dba）：
- 日 close-to-close 波动率 ≈ warmup σ（SPX 1.32% vs σ 1.29%；BTC 3.27% vs 3.24%）。
- 世界线指定的阶段终点 noisy==smooth（0.000%）；资产缺终点的阶段为内插内部点。
- 价格全正；VIX≥9；leak≈25%（news 前）；EURUSD/USDJPY/USDCNY 不再 flat。

---

## 10. 变更须知（保严谨）

- 改 `σ 取法` / `噪声公式` / `种子方案` / `leak 参数` / `FX β` / `re-anchor` 任一项 → 生成结果变化 → **必须更新本文件**并重生成 9 条 WL。
- warmup 数据补抓/修订 → σ 与 β 重算 → 重生成。
- 世界线 `.md` 阶段终点表是**人工权威输入**，脚本只解析不修改；要改情景只能改 `.md`。
