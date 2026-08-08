#!/usr/bin/env python
"""
gen_worldline_online.py — 由 9 条世界线的阶段终点表，生成在线阶段合成日频路径

**生成方法严谨可复现，详见 `data-prepare/process.md`**（σ 取法 / GBB 噪声采样 / 种子 / leak / FX 派生 / re-anchor）。

在线阶段 2026-07-17 ~ 各WL末阶段（9 条 WL 均至 2035-12-31）的「未来行情」无法实测；由 wordline-simple/wordline1..9.md
每阶段的资产终点价格插值出日频 close，再拼到 warmup 真实数据后，形成完整 2020-2030 面板，
供 walk_forward 在线滚动（agent 只能看到 ≤ t 的切片，防穿越）。

锚点策略（默认 re-anchor，可关）
  warmup 真实价（2026-07-16）与世界线「估计基线」显著不等（如 SOX 真~11700 vs 估 5800）。
  直接用世界线绝对价会在边界断层。故按世界线「相对路径」缩放/平移到真实价：
    价格类(权益/商品/加密/汇率): real(t) = real_0716 × wl(t) / wl_baseline   （保留世界线涨跌幅）
    收益率/波动率(US10Y/CN10Y/VIX): real(t) = real_0716 + (wl(t) - wl_baseline)（保留 bp/点数变动）
  --no-reanchor 则直接用世界线绝对价（边界有断层，仅对照用）。

插值
  价格类：waypoints 间对数线性（几何漂移，更贴近市场）。
  收益率/VIX：waypoints 间线性（水平）。
  日历：在线阶段统一用 Mon-Fri 工作日（4.5yr≈1170 日），所有资产同日历 → panel 整齐。
  OHLCV：close=插值；open=前一收盘；high/low=close×(1±rng/2)，rng 取 warmup 末 60 日 (high-low)/close 中位；
         volume=warmup 该资产成交量的中位（无原生量的债券/汇率/VIX=0）。

输出（data-prepare/online-worldline/）
  WL<n>_online.csv   仅在线阶段长表(date,asset_id,open,high,low,close,volume,amount)
  WL<n>_full.parquet/csv  warmup(≤2026-07-16, 真实) + 在线(合成) 完整面板（20 资产 = 15 可交易 + 5 信号）
  WAYPOINTS.md       每条世界线解析出的阶段终点（人工核对用）

用法
  python gen_worldline_online.py                 # 9 条世界线全跑，re-anchor
  python gen_worldline_online.py --only 1,3,5
  python gen_worldline_online.py --no-reanchor
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WLDIR = HERE / "wordline-simple"
OUTDIR = HERE / "online-worldline"
WARMUP_PANEL = HERE / "asset-daily-data" / "panel.parquet"
WARMUP_CSV = HERE / "asset-daily-data" / "panel.csv"
BASELINE_DATE = "2026-07-16"
ONLINE_END = "2035-12-31"   # 兜底默认；实际逐 WL 取 max(阶段 end_date)（9 条 WL 均为 2035-12-31）

# 中文名 → asset_id（与 ASSETS.yaml / fetch 口径一致）
ASSET_NAME_MAP = {
    "沪深300": "000300.SH", "标普500": "SPX", "恒生指数": "HSI", "日经225": "N225",
    "斯托克50": "SX5E", "欧洲斯托克50": "SX5E", "SOX": "SOX", "NDX": "NDX",
    "科创50": "000688.SH", "黄金": "XAU", "铜": "COPPER", "原油WTI": "WTI", "原油": "WTI",
    "BTC": "BTC", "ETH": "ETH", "美债10Y": "US10Y", "中债10Y": "CN10Y",
    "美元指数": "DXY", "USD/CNY": "USDCNY", "USD/JPY": "USDJPY", "USD/KRW": "USDKRW",
    "EUR/USD": "EURUSD", "欧元/美元": "EURUSD", "欧元": "EURUSD",
    "VIX": "VIX", "韩国KOSPI": "KOSPI", "KOSPI": "KOSPI",
}
LINEAR_ASSETS = {"US10Y", "CN10Y", "VIX"}  # 收益率/波动率：水平线性插值 + 平移锚定
# 信号汇率：世界线表格未给轨迹时，由 DXY（9 条 WL 全有）按 warmup 实测 β 派生，避免在线阶段 flat。
# EUR 是 DXY 主成分(57.6%)→β 最显著；JPY(13.6%)、CNY(管理汇率)次之。
FX_DERIVE = {"USDCNY", "USDJPY", "EURUSD"}

# === 价格-leads-news（内幕抢跑）参数 ===
# 每个阶段段 [t0=上阶段末, t1=本阶段末] 内，news 在 t_news = t0 + LEAD_TIME_FRAC×段长 处破裂。
# 到 t_news 时价格已完成 LEAD_MOVE_FRAC 比例的移动（抢跑/leak），剩余在 news 后加速反应。
# → 价格先于 news 变动；news 绝不先于价格。命中阶段终点不变。
LEAD_TIME_FRAC = 0.35   # news 在段内 35% 时点破裂（前 35% 时间走 leak）
LEAD_MOVE_FRAC = 0.25   # leak 占该段总移动的 25%（慢 leak）→ 后 65% 时间走 75%（加速反应）

# 线性资产（收益率/波动率）GBB 加性噪声后的下限（防负值；VIX 历史地板 ~9-10）
LINEAR_FLOOR = {"VIX": 9.0, "US10Y": 0.05, "CN10Y": 0.05}


# --------------------------------------------------------------------------- #
def tok_to_date(tok: str) -> str:
    tok = tok.strip().rstrip(".")
    if tok.endswith("H1"):
        return f"{tok[:4]}-06-30"
    if tok.endswith("H2"):
        return f"{tok[:4]}-12-31"
    if "." in tok:
        y, mo = tok.split(".")[:2]
        mo = int(round(float(mo)))
        return f"{int(y):04d}-{mo:02d}-{calendar.monthrange(int(y), mo)[1]}"
    return f"{int(tok):04d}-12-31"


def parse_stage_end(header: str) -> str | None:
    m = re.search(r"[（(]\s*([\d.H]+)\s*-\s*([\d.H]+)\s*[）)]", header)
    return tok_to_date(m.group(2)) if m else None


def num(s: str) -> float | None:
    """'4,608' / '5.50%' / '+120bp' 的「价格水平」列 → float（剥千分位/%/bp 取前缀数字）。"""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("%", "").replace(" ", "")
    if not s:
        return None
    s = s.lstrip("+-")
    m = re.match(r"\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def parse_worldline(path: Path):
    """返回 stages: [{end_date, rows:{asset_id: end_level}, names_seen}]，
    以及 baseline_levels（阶段一第 2 列，世界线估计基线）。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    stages, cur = [], None
    in_table = False
    baseline = {}
    for ln in lines:
        hs = ln.strip()
        if hs.startswith("## 阶段"):
            cur = {"end_date": parse_stage_end(hs), "rows": {}, "header": hs,
                   "idx": len(stages)}
            stages.append(cur)
            in_table = False
            continue
        if cur is None:
            continue
        if hs.startswith("|") and "---" not in hs and "资产" in hs:
            in_table = True
            continue
        if hs.startswith("|") and in_table:
            cells = [c.strip() for c in hs.strip("|").split("|")]
            if len(cells) < 4:
                continue
            name, start_v, end_v = cells[0], cells[1], cells[2]
            aid = ASSET_NAME_MAP.get(name)
            if aid is None:
                continue
            lvl = num(end_v)
            if lvl is None or lvl == 0:
                continue
            cur["rows"][aid] = lvl
            if cur["idx"] == 0:  # 阶段一的「起点」列 = 世界线估计基线（所有资产）
                b = num(start_v)
                if b:
                    baseline[aid] = b
        elif not hs.startswith("|"):
            in_table = False
    return stages, baseline


# --------------------------------------------------------------------------- #
def weekdays(start: str, end: str) -> list[str]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(s, e)]


def interp_waypoints(dates: list[str], wp_dates: list[str], wp_vals: list[float],
                     log: bool) -> np.ndarray:
    """在 (wp_dates, wp_vals) 间插值出 dates 上的值。log=True 用对数线性。"""
    s = pd.Series(wp_vals, index=pd.to_datetime(wp_dates), dtype=float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if log:
        grid = np.interp(pd.DatetimeIndex(dates).astype("int64"),
                         s.index.astype("int64"), np.log(s.values))
        return np.exp(grid)
    grid = np.interp(pd.DatetimeIndex(dates).astype("int64"),
                     s.index.astype("int64"), s.values)
    return grid


def warmup_stats(panel: pd.DataFrame) -> dict:
    """每资产：2026-07-16 收盘(real anchor)、末60日 (high-low)/close 中位(rng)、成交中位(vol)。"""
    stats = {}
    for aid, g in panel.groupby("asset_id"):
        g = g.sort_values("date")
        sub = g[g["date"] <= BASELINE_DATE]
        real_close = float(sub["close"].iloc[-1]) if not sub.empty else float(g["close"].iloc[-1])
        last60 = sub.tail(60)
        hl = (pd.to_numeric(last60["high"], errors="coerce")
              - pd.to_numeric(last60["low"], errors="coerce")) / pd.to_numeric(last60["close"], errors="coerce")
        rng = float(np.nanmedian(hl.replace(0, np.nan))) if hl.notna().any() else 0.005
        vol = float(pd.to_numeric(sub["volume"], errors="coerce").median()) if not sub.empty else 0.0
        stats[aid] = {"real_close": real_close, "rng": max(rng, 0.0005), "vol": max(vol, 0.0)}
    return stats


def compute_fx_betas(panel: pd.DataFrame):
    """回归各信号汇率日 log-return 对 DXY 日 log-return：β = cov(r_fx, r_dxy)/var(r_dxy)。
    返回 {aid: β} 与 DXY 的真实 2026-07-16 收盘。FX_DERIVE 中资产在 WL 无轨迹时用 β×DXY 累计对数收益派生。"""
    p = panel.copy()
    p["close"] = pd.to_numeric(p.groupby("asset_id")["close"].transform(
        lambda c: pd.to_numeric(c, errors="coerce")), errors="coerce")
    p["lr"] = p.groupby("asset_id")["close"].transform(lambda c: np.log(c).diff())
    pivot = p.pivot_table(index="date", columns="asset_id", values="lr")
    dxy = pivot.get("DXY")
    betas = {}
    if dxy is not None:
        for aid in FX_DERIVE:
            if aid in pivot.columns:
                pair = pd.concat([dxy.rename("dxy"), pivot[aid].rename("fx")], axis=1).dropna()
                var = float(pair["dxy"].var())
                betas[aid] = float(pair["fx"].cov(pair["dxy"]) / var) if var > 0 else 0.0
    dxy_real = float(pd.to_numeric(
        panel[panel["asset_id"] == "DXY"].sort_values("date").pipe(
            lambda g: g[g["date"] <= BASELINE_DATE]["close"]).iloc[-1], errors="coerce"))
    return betas, dxy_real


def compute_realized_vols(panel: pd.DataFrame) -> dict:
    """每资产 warmup 日频已实现波动率 σ：价格类=日 log-return std；线性类(US10Y/CN10Y/VIX)=日差 std。
    用于 GBB 日频噪声幅度（确定性、无 AI）。"""
    sigma = {}
    for aid, g in panel.groupby("asset_id"):
        g = g.sort_values("date")
        c = pd.to_numeric(g["close"], errors="coerce").astype(float)
        sigma[aid] = float(c.diff().std()) if aid in LINEAR_ASSETS else float(np.log(c).diff().std())
    return sigma


def _bridge_noise(days: list[str], wps: list[str], sigma: float, seed: int) -> np.ndarray:
    """逐段离散布朗桥噪声（len=len(days)）。每段 [wps[k], wps[k+1]] 独立 BB，
    在段边界（= 世界线阶段终点 / leak 航点）处归零 → 终点值严格命中。σ=日频波动率。"""
    noise = np.zeros(len(days))
    if sigma <= 0:
        return noise
    dts = pd.to_datetime(pd.Series(days))
    for k in range(len(wps) - 1):
        t0, t1 = pd.Timestamp(wps[k]), pd.Timestamp(wps[k + 1])
        idx = np.where((dts >= t0) & (dts <= t1))[0]
        m = len(idx)
        if m < 3:
            continue
        rng = np.random.default_rng(zlib.crc32(f"{seed}|{k}".encode()))
        eps = rng.normal(0.0, sigma, size=m - 1)        # m-1 增量
        B = np.concatenate([[0.0], np.cumsum(eps)])     # len=m，B[0]=0
        u = np.arange(m) / (m - 1)
        noise[idx] += B - B[-1] * u                     # BB：u=0,1 处为 0（B[0]=0→bb[0]=0）
    return noise


# --------------------------------------------------------------------------- #
def _insert_leak_waypoints(wps, wpd, ltf, lmf, log):
    """每段 [t0,v0]→[t1,v1] 插入一个'抢跑航点'：news 在 t_news=t0+ltf×段长 破裂，
    此时价格已到 v_leak（按 lmf 向终点插值）。interp_waypoints 过该点 → leak 后加速。
    log=True 时 v_leak 在对数空间内插（v0×(v1/v0)^lmf），与 log-linear 插值一致。"""
    if ltf <= 0 or len(wps) < 2:
        return wps, wpd
    nwps, nwpd = [wps[0]], [wpd[0]]
    for k in range(len(wps) - 1):
        t0, t1 = pd.Timestamp(wps[k]), pd.Timestamp(wps[k + 1])
        v0, v1 = float(wpd[k]), float(wpd[k + 1])
        dur = (t1 - t0).days
        if dur <= 1:
            nwps.append(wps[k + 1]); nwpd.append(wpd[k + 1]); continue
        t_news = t0 + pd.Timedelta(days=int(round(dur * ltf)))
        v_leak = v0 * (v1 / v0) ** lmf if log else v0 + lmf * (v1 - v0)
        nwps.append(t_news.strftime("%Y-%m-%d")); nwpd.append(v_leak)
        nwps.append(wps[k + 1]); nwpd.append(wpd[k + 1])
    return nwps, nwpd


def _asset_path(aid, stages, baseline, stats, reanchor, days,
                lead_time_frac=0.0, lead_move_frac=0.0,
                sigma=0.0, seed=0, noise=False):
    """单资产在线 close 路径。有世界线轨迹→插值+reanchor；返回 (path, used_worldline: bool)。
    lead_time_frac>0：每段插入抢跑航点（价格先于 news）。
    noise & sigma>0：叠加几何布朗桥日频噪声（段边界归零→命中终点与 leak）。"""
    wps, wpd = [], []
    if aid in baseline:
        wps.append(BASELINE_DATE); wpd.append(baseline[aid])
        seen = {BASELINE_DATE}
        for st in stages:
            if aid in st["rows"] and st["end_date"] and st["end_date"] not in seen:
                seen.add(st["end_date"]); wps.append(st["end_date"]); wpd.append(st["rows"][aid])
    st_ = stats.get(aid, {"real_close": 1.0})
    if wps:
        log = aid not in LINEAR_ASSETS
        wps, wpd = _insert_leak_waypoints(wps, wpd, lead_time_frac, lead_move_frac, log)
        raw = interp_waypoints(days, wps, wpd, log=log)
        if reanchor:
            wl_base = wpd[0]
            path = st_["real_close"] * raw / wl_base if log else st_["real_close"] + (raw - wl_base)
        else:
            path = raw
        if noise and sigma > 0:
            nz = _bridge_noise(days, wps, sigma, seed)   # wps 含 leak+终点 → 边界归零
            if log:
                path = path * np.exp(nz)
            else:
                path = path + nz
                floor = LINEAR_FLOOR.get(aid, 0.0)       # 线性资产防负
                if floor:
                    path = np.maximum(path, floor)
        return np.asarray(path, dtype=float), True
    return None, False


def build_online(stages, baseline, assets: list[str], stats, reanchor: bool,
                 online_end: str, fx_betas: dict, dxy_real: float, derive_fx: bool,
                 lead_time_frac: float = 0.0, lead_move_frac: float = 0.0,
                 vol_map: dict | None = None, wl_num: int = 0, noise: bool = False):
    """返回 online DataFrame[date,asset_id,ohlcv,amount]。
    在线阶段逐 WL 动态终点；信号汇率无世界线轨迹时由 DXY 按 β 派生（derive_fx=True）。
    lead_time_frac>0：每段插入抢跑航点（价格先于 news）。
    noise & vol_map：叠加 GBB 日频噪声（σ=warmup 实现波动率，端点归零→命中阶段终点与 leak）。"""
    day0 = (pd.Timestamp(BASELINE_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    days = weekdays(day0, online_end)
    vol_map = vol_map or {}

    def _seed(aid):
        return zlib.crc32(f"{wl_num}|{aid}".encode())

    # DXY 在线路径（9 条 WL 全有）→ 其累计对数收益供派生缺失汇率
    dxy_path, _ = _asset_path("DXY", stages, baseline, stats, reanchor, days,
                              lead_time_frac, lead_move_frac,
                              sigma=vol_map.get("DXY", 0.0), seed=_seed("DXY"), noise=noise)
    dxy_logret = np.log(dxy_path / dxy_real) if dxy_path is not None else None
    rows = []
    for aid in assets:
        path, used = _asset_path(aid, stages, baseline, stats, reanchor, days,
                                 lead_time_frac, lead_move_frac,
                                 sigma=vol_map.get(aid, 0.0), seed=_seed(aid), noise=noise)
        st_ = stats.get(aid, {"real_close": 1.0, "rng": 0.005, "vol": 0.0})
        if path is None:
            if derive_fx and aid in FX_DERIVE and dxy_logret is not None:
                β = fx_betas.get(aid, 0.0)
                path = st_["real_close"] * np.exp(β * dxy_logret)   # 由 DXY 派生，连续锚定
            else:
                path = np.full(len(days), st_["real_close"])        # flat hold
        rng, vol = st_["rng"], st_["vol"]
        close = path
        open_ = np.concatenate([[st_["real_close"]], close[:-1]])
        if aid in LINEAR_ASSETS:
            high = np.maximum(open_, close) + rng * close / 2
            low = np.minimum(open_, close) - rng * close / 2
        else:
            high = np.maximum(open_, close) * (1 + rng / 2)
            low = np.minimum(open_, close) * (1 - rng / 2)
        low = np.maximum(low, close * 1e-6)
        for i, d in enumerate(days):
            rows.append({"date": d, "asset_id": aid, "open": open_[i], "high": high[i],
                         "low": low[i], "close": close[i], "volume": vol})
    df = pd.DataFrame(rows)
    df["amount"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="逗号分隔世界线号(1-9)")
    ap.add_argument("--no-reanchor", action="store_true", help="用世界线绝对价(边界断层)")
    ap.add_argument("--no-derive-fx", action="store_true",
                    help="关闭：世界线无轨迹的信号汇率(EURUSD/USDJPY/USDCNY)保持 flat（默认由 DXY 按 β 派生）")
    ap.add_argument("--no-lead", action="store_true",
                    help="关闭价格-leads-news（默认 ON：每段插入抢跑航点，价格先于 news 变动）")
    ap.add_argument("--lead-time-frac", type=float, default=LEAD_TIME_FRAC,
                    help=f"news 在段内的破裂时点（占段长比例，默认 {LEAD_TIME_FRAC}）")
    ap.add_argument("--lead-move-frac", type=float, default=LEAD_MOVE_FRAC,
                    help=f"news 破裂前已完成的移动比例（leak，默认 {LEAD_MOVE_FRAC}）")
    ap.add_argument("--no-noise", action="store_true",
                    help="关闭 GBB 日频噪声（默认 ON：σ=warmup 实现波动率，端点归零命中阶段终点）")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()
    lead_time_frac = 0.0 if args.no_lead else args.lead_time_frac
    lead_move_frac = 0.0 if args.no_lead else args.lead_move_frac

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(WARMUP_PANEL) if WARMUP_PANEL.exists() else pd.read_csv(WARMUP_CSV)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    from asset_spec import BENCHMARK_ASSET_IDS
    assets = BENCHMARK_ASSET_IDS
    stats = warmup_stats(panel)
    fx_betas, dxy_real = compute_fx_betas(panel)
    vol_map = compute_realized_vols(panel) if not args.no_noise else {}
    if not args.no_noise:
        print(f"GBB 噪声 σ（warmup 实现波动率）：{ {k: round(v,4) for k,v in vol_map.items()} }")
    if not args.no_derive_fx:
        print(f"信号汇率 vs DXY 实测 β（派生用）：{ {k: round(v,3) for k,v in fx_betas.items()} }；DXY real={dxy_real:.2f}")
    print(f"warmup 面板 {len(panel)} 行, {panel['asset_id'].nunique()} 资产；"
          f"在线阶段 {BASELINE_DATE}+1 ~ 各WL末阶段（逐WL动态，兜底上限 {ONLINE_END}）\n")

    wls = [int(x) for x in re.findall(r"\d", args.only)] if args.only else list(range(1, 10))
    wp_lines = ["# WAYPOINTS — 各世界线解析的阶段终点（机器解析，供人工核对）", "",
                f"> re-anchor={'OFF' if args.no_reanchor else 'ON'}；"
                f"derive-fx={'OFF' if args.no_derive_fx else 'ON（无轨迹汇率由 DXY 按 β 派生）'}；"
                f"price-leads-news={'OFF' if args.no_lead else f'ON（news@段内{lead_time_frac:.0%}处破裂，先走{lead_move_frac:.0%}leak）'}；"
                f"GBB噪声={'OFF' if args.no_noise else 'ON（σ=warmup实现波动率，端点归零命中终点）'}", ""]

    for n in wls:
        path = WLDIR / f"wordline{n}.md"
        if not path.exists():
            print(f"WL{n}: 文件不存在，跳过"); continue
        stages, baseline = parse_worldline(path)
        # 逐 WL 动态终点：取该世界线末阶段真实结束日（不再用全局 2030-12-31）
        wl_end = max((st["end_date"] for st in stages if st["end_date"]), default=ONLINE_END)
        online = build_online(stages, baseline, assets, stats,
                              reanchor=not args.no_reanchor, online_end=wl_end,
                              fx_betas=fx_betas, dxy_real=dxy_real,
                              derive_fx=not args.no_derive_fx,
                              lead_time_frac=lead_time_frac, lead_move_frac=lead_move_frac,
                              vol_map=vol_map, wl_num=n, noise=not args.no_noise)
        online.to_csv(outdir / f"WL{n}_online.csv", index=False)

        # stage_news.json：每阶段的触发 news，dated 在 news_date（价格 leak 之后）。
        # build_inputs 可据此注入对齐的月度新闻（news 滞后价格抢跑）。
        prev_end = BASELINE_DATE
        news_records = []
        for st in stages:
            if not st.get("end_date"):
                continue
            t0, t1 = pd.Timestamp(prev_end), pd.Timestamp(st["end_date"])
            dur = (t1 - t0).days
            news_date = (t0 + pd.Timedelta(days=int(round(dur * lead_time_frac)))).strftime("%Y-%m-%d") if dur > 1 else st["end_date"]
            title = st["header"].split("：", 1)[1].split("（")[0].strip() if "：" in st["header"] else st["header"]
            news_records.append({
                "stage_end": st["end_date"], "news_date": news_date,
                "title": title, "stage_header": st["header"],
                "asset_endpoints": st["rows"],
            })
            prev_end = st["end_date"]
        (outdir / f"WL{n}_stage_news.json").write_text(
            json.dumps(news_records, ensure_ascii=False, indent=2), encoding="utf-8")
        full = pd.concat([panel[panel["asset_id"].isin(assets)], online], ignore_index=True)
        full = full.sort_values(["asset_id", "date"]).reset_index(drop=True)
        try:
            full.to_parquet(outdir / f"WL{n}_full.parquet", index=False)
        except Exception as e:
            print(f"  (parquet 写入跳过: {e})")
        full.to_csv(outdir / f"WL{n}_full.csv", index=False)

        # endpoints 报告
        wp_lines.append(f"## WL{n}　阶段终点（{len(stages)} 阶段）　前向终点 = {wl_end}")
        for st in stages:
            wp_lines.append(f"- 阶段结束 {st['end_date']}：{st['header'].split('：',1)[1].split('（')[0]}")
        # 抽样：SOX/SPX/BTC/US10Y 在线首末
        samp = online[online["asset_id"].isin(["SPX", "SOX", "BTC", "US10Y", "CN10Y", "VIX"])]
        if not samp.empty:
            wp_lines.append("  抽样(在线首日/末日 close):")
            for aid, g in samp.groupby("asset_id"):
                g = g.sort_values("date")
                wp_lines.append(f"    {aid}: {g['date'].iloc[0]}={g['close'].iloc[0]:.2f} → "
                                f"{g['date'].iloc[-1]}={g['close'].iloc[-1]:.2f}")
        wp_lines.append("")
        print(f"WL{n}: {len(stages)} 阶段, 在线 {len(online)} 行 ({online['date'].min()}~{online['date'].max()}), "
              f"full {len(full)} 行")

    (outdir / "WAYPOINTS.md").write_text("\n".join(wp_lines), encoding="utf-8")
    print(f"\n✅ 完成 {len(wls)} 条世界线 → {outdir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
