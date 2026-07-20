"""前向走步调度器：2026.07.16 → 2030.12.31 每日推进，驱动 AlphaCrafter + FactorMiner。

核心约束（来自 refer.md 最终方案）：
  - 恢复原生日频（不再按月阻断）。
  - 全资产统一单边 3bps 摩擦（1bp 佣金 + 2bp 滑点）已在两框架内嵌；本调度器在
    「统一账本」里按换手量同步扣费，口径与交易所一致。
  - AlphaCrafter 每月首个交易日读取当月新闻（新闻文件由 prepare_data 预置到每月 1 日，
    AC 的 get_news 按 publish_date ≤ current_date 自动过滤）。
  - 严格防穿越：第 t 日 AC 只见 current_date=t 的数据（其工具内置过滤）；
    FM 只见 panel[:t]（本调度器显式切片）。

两 leg 融合：默认 AlphaCrafter leg 与 FactorMiner leg 各占 50% 净资产，各自输出当日
目标权重 → 融合 → 按 3bps 计摩擦 → 更新统一 NAV。融合比例 --alpha 可调。

运行模式：
  --dry-run（默认）：用确定性占位信号（等权 + 动量）驱动完整日循环，校验时间游标、
    防穿越、摩擦 NAV、每月新闻节奏——无需 API Key、无需真实数据即可跑通。
  --live-ac / --live-fm：分别把对应 leg 切到真实子进程调用
    （AC: ``python -m alphacrafter.main``；FM: ``factorminer ... mine``）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from asset_universe import AC_SYMBOLS, BASELINE_DATE, FORWARD_END, IDS
from friction import COST_BPS

# ---------------------------------------------------------------------------
# 时间轴与防穿越
# ---------------------------------------------------------------------------
def forward_trading_days(panel: pd.DataFrame) -> List[str]:
    """返回 [BASELINE_DATE, FORWARD_END] 内的交易日升序列表。"""
    days = sorted(panel.loc[panel["date"].between(BASELINE_DATE, FORWARD_END), "date"].unique())
    if not days:
        raise ValueError(f"panel 在 {BASELINE_DATE}~{FORWARD_END} 内无交易日，请先 prepare_data。")
    return days.tolist()


def slice_to(panel: pd.DataFrame, t: str) -> pd.DataFrame:
    """防穿越切片：只保留 date ≤ t 的行。"""
    return panel[panel["date"] <= t].copy()


def is_first_trading_day_of_month(t: str, trading_days: List[str]) -> bool:
    """t 是否为其所在月份的第一个交易日（用于日志/新闻节奏核对）。"""
    idx = trading_days.index(t)
    if idx == 0:
        return True
    return trading_days[idx - 1][:7] != t[:7]


# ---------------------------------------------------------------------------
# 占位信号（dry-run 用，保证全链路可验证）
# ---------------------------------------------------------------------------
def _equal_weight() -> Dict[str, float]:
    n = len(IDS)
    return {a: 1.0 / n for a in IDS}


def _momentum_weight(slice_panel: pd.DataFrame) -> Dict[str, float]:
    """20 日动量占位信号：近 20 日涨幅排名 top 1/3 等权做多，其余 0。"""
    rets = {}
    for aid, g in slice_panel.groupby("asset_id"):
        g = g.sort_values("date")
        if len(g) >= 21:
            rets[aid] = g["close"].iloc[-1] / g["close"].iloc[-21] - 1.0
        else:
            rets[aid] = 0.0
    if not rets or max(rets.values()) == min(rets.values()):
        return _equal_weight()
    order = sorted(rets, key=lambda a: rets[a], reverse=True)
    top = order[: max(1, len(order) // 3)]
    w = {a: 0.0 for a in IDS}
    for a in top:
        w[a] = 1.0 / len(top)
    return w


# ---------------------------------------------------------------------------
# 子进程驱动（live 模式）
# ---------------------------------------------------------------------------
def run_ac_live(session: str, ac_config: Path, ac_root: Path) -> Dict[str, float]:
    """调一次 AC Launcher（max_cycles 由其 config 决定），返回当日目标权重。

    权重从 AC session 的 account.json positions 读取（占净资产比例）。失败时回落等权。
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "alphacrafter.main", session, "--config", str(ac_config)],
            cwd=str(ac_root), check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"⚠️  AC 周期失败: {e}，回落等权")
        return _equal_weight()
    # 读 AC 持仓 → 权重（简化：按市值/净资产）
    acct_path = ac_root / "alphacrafter" / "sandbox" / "template_a" / "persistent" / "account.json"
    if not acct_path.exists():
        return _equal_weight()
    acct = json.loads(acct_path.read_text())
    nav = float(acct.get("net_assets", 1.0)) or 1.0
    w = {a: 0.0 for a in IDS}
    for pos in acct.get("positions", []):
        sym = pos.get("symbol")
        if sym in w:
            w[sym] = round(float(pos.get("market_value", 0.0)) / nav, 6)
    return w


def run_fm_live(fm_config: Path, fm_root: Path, panel_slice_path: Path) -> Dict[str, float]:
    """跑一次 FM mining（在 panel[:t] 上），返回当日截面得分 → top 1/3 等权做多。

    真实因子信号需读 FM 输出的因子库并计算；此处给出调用入口与兜底，落地时按
    FM ``top_formulaic_alphas.json`` 计算 Ts_Rank 截面分。失败回落等权。
    """
    try:
        subprocess.run(
            ["factorminer", "--config", str(fm_config), "mine", "--data-path", str(panel_slice_path)],
            cwd=str(fm_root), check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  FM mining 失败: {e}，回落等权")
        return _equal_weight()
    # TODO(落地): 解析 FM 输出 top_formulaic_alphas.json，在 panel[:t] 上计算
    #   各资产截面得分，买入得分最高前 1/3。暂回落等权。
    return _equal_weight()


# ---------------------------------------------------------------------------
# 融合 + 摩擦 NAV
# ---------------------------------------------------------------------------
def blend(ac_w: Dict[str, float], fm_w: Dict[str, float], alpha: float) -> Dict[str, float]:
    """两 leg 线性融合：alpha 为 AC leg 占比。归一化到和为 1（允许留现金）。"""
    merged = {a: alpha * ac_w.get(a, 0.0) + (1 - alpha) * fm_w.get(a, 0.0) for a in IDS}
    tot = sum(merged.values())
    if tot > 1.0:  # 超额满仓按比例缩
        merged = {a: v / tot for a, v in merged.items()}
    return merged


def update_nav(nav: float, w_prev: Dict[str, float], w_new: Dict[str, float],
               prices: Dict[str, float], prev_prices: Dict[str, float]) -> tuple:
    """按当日收益更新 NAV，并对换手扣 3bps 摩擦。返回 (new_nav, turnover, friction)."""
    # 持仓收益（按昨仓 + 今日价格变动）
    gross = sum(w_prev.get(a, 0.0) * (prices.get(a, 0.0) / prev_prices.get(a, prices.get(a, 1.0)) - 1.0)
                for a in IDS)
    turnover = sum(abs(w_new.get(a, 0.0) - w_prev.get(a, 0.0)) for a in IDS) / 2.0  # 单边换手
    friction = turnover * (COST_BPS / 10000.0)
    new_nav = nav * (1.0 + gross - friction)
    return new_nav, turnover, friction


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def run(panel_path: Path, alpha: float, live_ac: bool, live_fm: bool,
        fm_cadence: int, ac_config: Path, fm_config: Path,
        ac_root: Path, fm_root: Path, out_dir: Path) -> None:
    panel = pd.read_parquet(panel_path) if panel_path.suffix in (".parquet", ".pq") else pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    days = forward_trading_days(panel)
    out_dir.mkdir(parents=True, exist_ok=True)

    nav = 1.0e7
    w_prev = {a: 0.0 for a in IDS}
    prev_prices = {a: 1.0 for a in IDS}
    ledger: List[dict] = []
    print(f"▶ 前向走步 {days[0]} → {days[-1]}（{len(days)} 交易日，"
          f"AC leg α={alpha}，FM mining 每 {fm_cadence} 日，摩擦单边 {COST_BPS}bps）")

    slice_cache = out_dir / "_panel_slice"
    slice_cache.mkdir(exist_ok=True)
    for i, t in enumerate(days):
        sl = slice_to(panel, t)
        prices = {a: float(g.sort_values("date")["close"].iloc[-1]) for a, g in sl.groupby("asset_id")}
        prices = {a: prices.get(a, prev_prices.get(a, 1.0)) for a in IDS}

        # --- AlphaCrafter leg（日频；current_date 已由其内部 step 推进，新闻按月自动读）---
        if live_ac:
            ac_w = run_ac_live(f"wf_{t}", ac_config, ac_root)
        else:
            ac_w = _equal_weight()

        # --- FactorMiner leg（按 cadence 重挖；每日用最新因子出权重）---
        if live_fm and (i % fm_cadence == 0):
            sl_path = slice_cache / f"panel_{t}.csv"
            sl.rename(columns={"date": "datetime"}).to_csv(sl_path, index=False)
            fm_w = run_fm_live(fm_config, fm_root, sl_path)
        else:
            fm_w = _momentum_weight(sl)

        # --- 融合 + 摩擦 NAV ---
        w_new = blend(ac_w, fm_w, alpha)
        nav, turnover, friction = update_nav(nav, w_prev, w_new, prices, prev_prices)
        news_day = "📰" if is_first_trading_day_of_month(t, days) else "  "
        ledger.append({
            "date": t, "nav": round(nav, 2), "ret": round(nav / 1.0e7 - 1.0, 6),
            "turnover": round(turnover, 6), "friction_bps": round(friction * 10000.0, 4),
            "gross_position": round(sum(w_new.values()), 4), "monthly_news": is_first_trading_day_of_month(t, days),
        })
        w_prev, prev_prices = w_new, prices
        if i % 20 == 0 or i == len(days) - 1:
            print(f"  {news_day} {t}  NAV={nav:,.0f}  累计={nav/1.0e7-1:+.2%}  换手={turnover:.3f}  摩擦={friction*10000:.2f}bps")

    led_path = out_dir / "nav_ledger.csv"
    pd.DataFrame(ledger).to_csv(led_path, index=False)
    total_friction = sum(r["friction_bps"] for r in ledger)
    news_months = sum(r["monthly_news"] for r in ledger)
    print(f"\n✅ 完成 {len(ledger)} 日。终值 NAV={nav:,.0f}（{nav/1.0e7-1:+.2%}）"
          f"  累计摩擦≈{total_friction:.0f}bps  每月新闻日≈{news_months}")
    print(f"   明细 → {led_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="前向走步调度器 2026.7.16→2030")
    ap.add_argument("--panel", default="fm_workspace/panel.csv", help="全量长表（prepare_data 产物，csv/parquet 均可）")
    ap.add_argument("--out", default="walk_forward_out", help="输出目录（NAV 账本等）")
    ap.add_argument("--alpha", type=float, default=0.5, help="AC leg 占比（1=纯AC，0=纯FM）")
    ap.add_argument("--fm-cadence", type=int, default=5, help="FM 重挖周期（交易日）")
    ap.add_argument("--live-ac", action="store_true", help="真实调用 AlphaCrafter（默认 dry-run）")
    ap.add_argument("--live-fm", action="store_true", help="真实调用 FactorMiner（默认 dry-run）")
    ap.add_argument("--ac-config", default="../AlphaCrafter/config_walkforward.yaml")
    ap.add_argument("--fm-config", default="../FactorMiner/factorminer/configs/default.yaml")
    ap.add_argument("--ac-root", default="../AlphaCrafter")
    ap.add_argument("--fm-root", default="../FactorMiner")
    args = ap.parse_args()
    run(
        panel_path=Path(args.panel), alpha=args.alpha,
        live_ac=args.live_ac, live_fm=args.live_fm, fm_cadence=args.fm_cadence,
        ac_config=Path(args.ac_config), fm_config=Path(args.fm_config),
        ac_root=Path(args.ac_root), fm_root=Path(args.fm_root),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
