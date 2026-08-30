# Retrospective Replay (RR) — 复用已有因子库对历史做回顾性回测

创建：2026-08-28。本目录实现 FORESIGHT-9 论文的对照实验 **Retrospective Replay**：
把四个实验（AC-terra / AC-DS / FM-terra / FM-DS）在 2026-07-15 边界已产出的
warmup qualified factor library 原封不动地拿来，在**它们被挖掘/筛选出来的同一段
真实历史**（2020-01-02 → 2026-07-15）上回放，得到 RR leaderboard；然后与九条
世界线上的前瞻结果做对照，回答：

> **Retrospective evaluation 到底给了我们多少关于 future robustness 的信息？**

## 1. 定位与措辞边界（必须遵守）

- **RR is intentionally retrospective and in-sample.** 这不是缺陷，是实验变量：
  它复现的正是当前金融 agent 工作最常见的评价方式——agent 在已实现历史上发现
  策略、又在同一段历史上给策略打分。
- **禁止的 claim**：不得声称"证明了 LLM 因训练数据污染导致历史回测虚高"。
  RR 中混杂 historical contamination 与 in-sample selection 两个因素，本实验
  无法拆开它们。
- **允许的最强 claim**：
  *High historical score does not imply transferable future competence；retrospective
  evaluation cannot distinguish historical familiarity and in-sample selection from
  transferable decision ability.* FORESIGHT-9 的九条世界线为右边提供测试。
- 论文里建议的定性句（备用）：
  "Retrospective Replay is deliberately not an out-of-sample test. It reproduces the
  evaluation regime whose limitations motivate FORESIGHT-9. We do not interpret its
  return as an estimate of future performance; we ask whether the ranking it induces
  agrees with rankings under unseen worldlines."

## 2. 复用的资产（不重新调模型，全程确定性）

| 资产 | 位置 | 说明 |
|---|---|---|
| 历史面板 | `ACFM_WL_paperwriting/3 benchmark时间线设计/worldlines_and_raw_panels/repro_wldatafinal/asset-daily-data/panel.csv` | 20 资产 × 2020-01-01–2026-07-16，OHLC+volume+amount；回放只用 ≤2026-07-15 |
| FM-terra warmup 库 | `ACFM_WL_paperwriting/2data/FM_factor_data_complete/luna/warmup_library_luna.csv` | 21 因子：name/formula/ic_mean/icir |
| FM-DS warmup 库 | 同目录 `ds/warmup_library_ds.csv` | 8 因子 |
| AC-terra warmup 库 | `agent-framework/AlphaCrafter/alphacrafter/sandbox/ws1/workspace/factors/*.json` | 6 因子 JSON（expression + ic） |
| AC-DS warmup 库 | `AC-deepseek/AlphaCrafter/alphacrafter/sandbox/ws1/workspace/factors/*.json` | 4 因子 JSON |
| AC ensemble 权重 | 同目录 `factor_ensemble.json`（quality_ic_tilt） | 4 因子带权重与方向 |
| FM DSL 求值器 | `report-and-output/FM-live/FM acceleration/bundle/agent-framework/FactorMiner/factorminer/core/{parser,expression_tree}.py` | `parse(str)` → ExpressionTree；`evaluate({feature: (M,T) ndarray})`；算子只向后看 |
| FORESIGHT-9 执行合同 | 论文 §5：10 交易日决策栅格、long-only Σw=1、e>3τ 迁移门、3 bps 单边、首仓免费 | 与前向完全一致 |
| EW-15 基线 | 同窗口同栅格的等权再平衡（带 3 bps），复刻 `picturegenerate1/src/make_foresight9_nav_figures.py::equal_weight_nav` | |
| 前瞻对照数据 | `ACFM_WL_paperwriting/5论文框架/iaeval_neurips2026/data/nav_metrics.csv` | 九 WL 终值，用于 rank/alpha 对照 |

## 3. 流水线（`rr_replay.py`，单文件、固定种子、无 LLM 调用）

1. **面板** → 15 tradable 的 (M=15, T) 特征矩阵（$open/$high/$low/$close/$volume/
   $amt/$vwap=amt/volume/$returns），另有 20 资产宽表供 AC 因子使用（VIX 等
   observation-only 序列只在因子里出现，不进组合）。
2. **因子值**：FM 公式经官方 parser+evaluator 全矩阵求值；AC 十个因子为 pandas
   直译（表达式见附录 A.1，全部为 lag/rolling/横截面中位数类，无歧义）。
   算子沿时间轴只用 ≤t 数据，无前视。
3. **组合**：因子值横截面 z 分数 → S = Σ_f w_f·dir_f·z_f。
   FM：warmup 库按 |ic_mean| 取 top-10（ds 全部 8 个），w_f = ic_mean（带符号，
   即论文 §4.1 的 IC-weighted sign-preserved 组合）；AC：ws1 factor_ensemble.json
   的权重与方向原样使用。
4. **映射**：long-only、Σw=1，w_i ∝ max(S_i, 0)；Σ≤0 或非有限 → 合同内置 1/15
   等权回退（与前向一致）。**说明**：AC 原生 trader 是 LLM 环节，回放以基准的
   确定性映射替代——四套配置共用同一映射器，配置间唯一差异是因子库与组合权重，
   这正是"library transfer"想要的受控变量。
5. **执行**：决策栅格 = 自 t_start（所有因子 burn-in 完成后的首个 10 的倍数日，
   主要是 125 日 lag 的 burn-in）起每 10 个交易日；迁移门 e>3τ，e 用 S 作 10 日
   预测收益（与合同同式），首仓免费，其余 3 bps×τ×V。
6. **EW-15**：同 t_start、同栅格等权再平衡（3 bps），同窗口终值。
7. **产出**（`outputs/`）：逐日 NAV csv、`rr_leaderboard.csv`（四配置 RR 终值 +
   EW-15 + RR alpha）、`rr_vs_worldlines.csv`（前瞻逐 WL 终值/alpha、
   Spearman ρ(RR 排名, WL_j 排名) j=1..9、alpha 保持率）、`summary.md`。

## 4. 验收标准

- **确定性**：固定随机种子（无随机源也应如此）；重跑逐字节一致。
- **无前视审计**：t 日决策只读 ≤t 行；t_start ≥ max lag+burn-in。
- **映射器校验**（工程自检，非论文内容）：用同一引擎在**世界线面板**上复算
  fm-terra-wl1 首个决策（2026-07-15）的目标权重，与
  `processed-core-data/fm-terra/fm-terra-wl1/holdings/holdings_timeline.csv` 首行
  比对（秩相关 + L1）；不匹配则如实记录差异并在 README 标注"统一代理映射器"。
- **口径**：终值 / 初始 1,000,000，与前向 nav_metrics.csv 同口径。
- **结论表述**：rank 转移（ρ_j 分布）、alpha 转移（多少 WL 保持正 alpha），
  只描述"不稳定/是否转移"，不归因污染。

## 5. 论文接入计划（下一步，另行一轮）

- Results 升级为 RQ1：Does retrospective performance predict prospective robustness?
  先 RR leaderboard，再 rank correlation 表/热图，再 alpha transfer 矩阵。
- 正文只放 RR 排名 vs 九 WL 排名的稳定性结论；窗口级 bootstrap 维持附录 C。
- `1theory/paper-improvement-plan-20260827.md` 记第十五轮。

/home/lxx/trade-agent-benchmark/backtest在这里写readme记录指导并复用已有产出因子完成对历史的回测。，**这里 A 反而更符合你这篇论文的核心论证**。我会收回前面“B 更学术干净”的倾向：对于一般因子论文 B 更干净，但对于 **FORESIGHT-9 这篇论文，B 会把你真正想研究的问题洗掉一部分**。

你的中心问题本来就不是：

> 一个严格 train/test split 的传统量化模型有没有 OOS 泛化能力？

而是：

> **Foundation-model trading agent 在已经发生、极可能进入预训练语料和公开金融知识体系的历史上表现很好，这种 retrospective evaluation 到底能不能说明它面对真正未知未来的能力？**

论文 Introduction 现在其实已经这样定位了：历史回测无法区分 transferable decision-making 与 historical memorization / overfitting。

因此 A 是非常自然的对照。

---

## A 最好不要被描述成普通“历史回测”

我会给它一个专门名字，例如：

**Retrospective Replay (RR)**

然后明确：

$$
\boxed{
\text{RR is intentionally retrospective and in-sample}
}
$$

这不是 weakness，而是**实验设计的一部分**。

你是在复现当前很多金融 agent 工作最容易得到的那种评价：

$$
\text{agent discovers strategy on realized history}
\rightarrow
\text{strategy scores well on realized history}
$$

然后问：

$$
\text{Does that apparent skill survive unseen futures?}
$$

所以实验逻辑可以非常漂亮：

$$
\underbrace{
D_{\text{2020--2026}}
\rightarrow
\text{LLM factor mining}
\rightarrow
\text{selected factor library}
\rightarrow
\text{replay on }D_{\text{2020--2026}}
}_{\text{Retrospective Replay}}
$$

vs.

$$
\underbrace{
\text{same boundary state}
\rightarrow
WL_1,\ldots,WL_9
}_{\text{Prospective evaluation}}
$$

---

# 但有一个措辞边界一定要守住

你不能通过这个实验声称：

> **我们证明了 LLM 因为训练数据污染所以历史回测成绩虚高。**

因为 RR 中同时存在至少两个因素：

$$
\text{historical contamination}
+
\text{in-sample strategy selection}.
$$

你无法从这个实验里把它们拆开。

所以最强、同时完全站得住的 claim 是：

> **Retrospective evaluation cannot distinguish historical familiarity and in-sample selection from transferable decision ability.**

这其实比硬证明“training contamination”更好。

你根本不需要证明 GPT 到底背过哪一天 SPX 的走势。

只要论证：

$$
\boxed{
\text{High historical score}
\not\Rightarrow
\text{transferable future competence}
}
$$

而 FORESIGHT-9 就是为右边这个东西提供测试。

这和你现在 Introduction 的论点是一致的。

---

# 这个实验实现起来可以非常便宜

你已经有每个 experiment 在 boundary 时得到的 warmup qualified library，而且 factor 公式都保存下来了。比如 Appendix 中已经列出了这些 warmup factor 及其公式。

所以无需重新调用模型。

对于四个配置：

$$
\begin{aligned}
&AC\text{-terra}\\
&AC\text{-DS}\\
&FM\text{-terra}\\
&FM\text{-DS}
\end{aligned}
$$

取各自在 2026-07-15 已经得到的 final warmup library：

$$
L_c^{2026}
$$

然后：

$$
L_c^{2026}
+
D_{2020:2026}
\rightarrow
\text{factor signals}
\rightarrow
\text{portfolio construction}
\rightarrow
NAV_c^{RR}.
$$

全程 deterministic。

而且尽量复用 FORESIGHT 的：

* portfolio mapper
* long-only constraints
* 10-day cadence
* transaction cost
* factor combination logic
* EW-15 baseline

这样唯一显著变化就是：

$$
\boxed{\text{realized past vs unseen worldlines}}
$$

---

# 这能产生一个比现在 Results 强很多的实验

假设结果类似：

| Configuration | Retrospective Replay | FORESIGHT median | WL wins vs RR rank |
| ------------- | -------------------: | ---------------: | -----------------: |
| FM-terra      |                   #1 |               #1 |                  … |
| AC-terra      |                   #2 |               #3 |                  … |
| AC-DS         |                   #3 |               #2 |                  … |
| FM-DS         |                   #4 |               #4 |                  … |

真正重要的甚至不是收益数字，而是：

### 1. Retrospective ranking 是否 transfer

定义：

$$
R^{RR}
$$

以及每个 worldline：

$$
R^{WL_j}.
$$

然后算：

$$
\rho_j=
\rho_{\text{Spearman}}
(R^{RR},R^{WL_j}).
$$

于是你可以得到：

> The framework ranking suggested by retrospective replay is unstable across alternative futures.

这比现在不断说：

> FM-terra 在 WL5 多少，AC-DS 在 WL8 多少

高级很多。

---

### 2. Historical apparent alpha 是否 transfer

算：

$$
\alpha_c^{RR}
=
R_c^{RR}-R_{EW}^{RR}
$$

然后对于每条 WL：

$$
\alpha_{c,j}^{F}
=
R_{c,j}^{F}-R_{EW,j}.
$$

你真正可以问：

> 一个在历史上看起来 alpha 很强的 agent，在多少 unseen futures 中还能保持 alpha？

例如图可以直接做：

```text
                 Retrospective     Prospective
                 Replay            WL1 ... WL9

FM-terra          ++++++++          + +++ -- + ...
AC-terra          ++++++            -- - -- ...
AC-DS             +++++             - + -- ...
FM-DS             ++++              -- -- ...
```

这比单纯 NAV 图更直击 thesis。

---

# 甚至我觉得这可以直接替换你现在一部分 Results

现在 Results 的第一大发现是：

> framework ranking depends on backbone and worldline. 

加了 RR 后，可以升级成：

## RQ1 — Does retrospective performance predict prospective robustness?

先给 RR leaderboard。

然后：

> No single retrospective ranking remains stable across the nine worldlines.

再展示 rank correlation / heatmap。

这时你的论文论证链第一次真正闭环：

### Motivation

历史回测可能高估 transferable competence。

↓

### Benchmark

构造 unseen counterfactual futures。

↓

### Direct experiment

在历史 replay 中得到一个 ranking。

↓

### Result

该 ranking 在 unseen futures 中失效/显著不稳定。

↓

### Conclusion

retrospective score 不足以评价 adaptive trading agents。

这比现在仅仅展示：

> “不同 WL 排名不一样”

强很多，因为现在 reviewer 还可以说：

> 当然不同测试集排名不一样，这有什么？

加上 RR 后问题变成：

> **我们一直使用的 retrospective evaluation 到底给了我们多少关于 future robustness 的信息？**

这就是 benchmark paper 应该回答的问题。

---

# 一个细节：不要假装 A 是公平的 OOS

甚至可以主动写：

> *Retrospective Replay is deliberately not an out-of-sample test. It reproduces the evaluation regime whose limitations motivate FORESIGHT-9: strategies are selected from the realized historical record and then scored on that same record.*

然后马上接：

> *We do not interpret its return as an estimate of future performance; we ask whether the ranking it induces agrees with rankings under unseen worldlines.*

这段我非常建议写。

因为 reviewer 如果攻击：

> “你这是 in-sample 啊！”

你的回答就是：

> **对，这正是实验变量。**

这样反而会显得设计很清楚。

---

所以我现在会把之前审核意见里的第 16 点正式改成：

> **必须优先考虑增加 Retrospective Replay，而不是 historical OOS benchmark。**

而且这个实验的价值大概率**高于现在的 8,820-window significance analysis**。如果主文篇幅不足，我甚至会选择：

**RR ranking comparison 进正文，window-level bootstrap 降 Appendix。**

这会让 FORESIGHT-9 从“我们做了九条 synthetic market path”真正变成一篇有明确 empirical thesis 的 benchmark 论文。
