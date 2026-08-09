# CHECKLIST — 因子筛选规则与运行审计清单

> 本清单记录因子筛选规则来源、审计状态与后续待办。因子筛选规则**来自 FM（FactorMiner）原版**（论文
> `2602.14670v1.pdf`，仓库内 `FactorMiner/`），AC 与 FM 共用同一份准入合同。

---

## 1. 因子筛选规则（来源：FM 原版）

| 规则 | 取值 | 说明 |
|---|---|---|
| IC 门槛 | `abs(IC) >= 0.007` | 由 FM 原版论文 CSI500 口径 `0.04` 按 15 资产缩放：`0.04 × sqrt(14/499) = 0.0067 ≈ 0.007` |
| ICIR 门槛 | `abs(ICIR) >= 0.084` | 同上缩放：`0.5 × sqrt(14/499) = 0.0838 ≈ 0.084` |
| 相关性门槛 | `abs(Spearman rho) < 0.5` | 与库内任一因子 `|rho| >= 0.5` 即拒绝（FM 原版 Eq.10） |
| 替换规则 | Eq.11 | 候选 `IC>=0.10` 且 `>=1.3×` 被替换因子 IC 且仅与 1 个库内因子冲突时可替换 |
| 库容量 | 30 | 超过 30 按 `q=|IC|×|ICIR|` 保留 best 30 |
| active ensemble | 10 | `q` 降序取 top-10 进组合 |

> 因子筛选/冗余规则代码：`FactorMiner/factorminer/core/factor_library.py`（`check_admission`/`check_replacement`）
> 与 `scheduler/run_pipeline.py`（`_trim_factor_library`）。两版（luna/terra 与 deepseek）此部分代码完全一致，
> 未做过任何因子冲突的隐藏修改。

## 2. 成本门控（交易决策层）

| 规则 | 取值 | 说明 |
|---|---|---|
| 决策频度 | 每 10 交易日 | `decision_cadence_trading_days: 10` |
| 成本门控 | `gross_edge_bps > 3 × migration` | 预期增量收益超过调仓量×3bp 才执行（用户确认的本地语义） |
| 单边迁移成本 | 迁移名义额 × 3bp | `ASSETS.yaml friction_bps: 3`；首次建仓免费 |
| 首次建仓 | 2026-07-16 全投 15 资产 | 无 ensemble 时等权 1/15；cash=0 |
| 落盘字段 | decision/execution/proposed/executed target 等 | 见 `fm_walk_forward.py` decisions |

## 3. 审计状态

### terra（luna 版，`D:\FM acceleration`）— 已审计 ✅
- 脚本：`scripts/audit_factor_library.py`（只读，不改运行状态/指纹）
- 报告：`runtime/state/factor_audit/WL<n>_20260809_191204.json`
- 结果：9 条 WL 全部 `ok`，库内 rho≥0.5 冗余对 = **0**（与 FM 准入"冲突即拒绝"自洽）；
  checkpoint 因子 ID 与顶层库完全一致
- 运行时间线：WL4-9 于 08/03-08/05，WL1-3 于 08/08-08/09，代码与 `reference/1` 快照一致

### deepseek 版（`D:\FM acceleration-deepseek`）— 待跑完后审计 ⏳
- 当前 3 worker 运行中（WL1/2/3 online），跑完 WL1-9 后再执行：
  ```
  cd "D:\FM acceleration-deepseek"
  python scripts/audit_factor_library.py --root "D:\FM acceleration-deepseek\bundle\agent-framework"
  ```
- 审计前注意：ds 版早前出现过 API 限流导致的 0 候选空转窗口，已在 08/09 回滚重跑；
  候选验证（`FM_GEN_MIN_CANDIDATES=38`）已生效，后续窗口 gen=40。

## 4. 待办

- [ ] deepseek 版 9 条 WL 全部跑完后执行审计并回填本节
- [ ] （可选）对 audit 报告的 q 排序与 top-10 ensemble 人工复核
- [ ] 任何因子冗余规则改动需同步更新本表第 1 节（来源仍标注 FM 原版）
