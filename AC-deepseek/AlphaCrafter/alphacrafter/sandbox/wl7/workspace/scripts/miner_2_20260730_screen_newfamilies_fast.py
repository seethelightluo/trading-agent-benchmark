"""miner_2 cycle 6: FAST batch screen of new factor families (optimized).
Universe: 15 tradable cross-asset instruments. Window 2020-01-01..2026-07-15.
Gate: |IC|>=0.007 and |ICIR|>=0.084 @ h=10.
Library signals computed ONCE; per-candidate corr window capped for speed.
"""
from __future__ import annotations
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner_2_lib as lib

panel = lib.load_panel()
rets = panel.pct_change()
eps = 1e-12

# ---------- price-only candidates (vectorized on union calendar) ----------
C = {}
C["near_high_120"] = panel / panel.rolling(120, min_periods=60).max() - 1.0
C["near_high_250"] = panel / panel.rolling(250, min_periods=120).max() - 1.0
C["up_day_ratio_60"] = (rets > 0).rolling(60, min_periods=30).mean()
C["range_pos_20d"] = (panel - panel.rolling(20, min_periods=10).min()) / (
    panel.rolling(20, min_periods=10).max() - panel.rolling(20, min_periods=10).min() + eps)
neg = rets.where(rets < 0, 0.0)
C["downside_dev_20"] = (neg.pow(2)).rolling(20, min_periods=10).mean().apply(np.sqrt)
C["max_dd_60"] = (panel / panel.rolling(60, min_periods=30).max() - 1.0).rolling(60, min_periods=30).min()
r5 = panel / panel.shift(5) - 1.0
v20 = rets.rolling(20, min_periods=10).std()
C["rev_5d_vol"] = -(r5 / (v20 + eps))
up = rets.where(rets > 0, np.nan).rolling(20, min_periods=10).mean()
dn = rets.where(rets < 0, np.nan).rolling(20, min_periods=10).mean()
C["up_down_ratio_20"] = up / (dn.abs() + eps)

# ---------- OHLC/volume candidates (per-asset calendar) ----------
closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
for s in lib.WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= lib.MAX_VISIBLE].set_index("date").sort_index()
    closes[s] = df["close"].astype(float)
    opens[s] = df["open"].astype(float)
    highs[s] = df["high"].astype(float)
    lows[s] = df["low"].astype(float)
    vols[s] = df["volume"].astype(float)


def build(cols_map, fn):
    out = {}
    for s in panel.columns:
        out[s] = fn(cols_map[s])
    return pd.DataFrame(out, index=panel.index)


def f_range_pos(c):
    h, l = c["high"].dropna(), c["low"].dropna()
    cl = c["close"].dropna()
    idx = cl.index.intersection(h.index).intersection(l.index)
    pos = (cl[idx] - l[idx]) / (h[idx] - l[idx] + eps)
    return pos.rolling(20, min_periods=10).mean()


def f_overnight(c):
    cl, o = c["close"].dropna(), c["open"].dropna()
    gap = o / cl.shift(1) - 1.0
    return gap.rolling(20, min_periods=10).mean()


def f_intraday(c):
    cl, o = c["close"].dropna(), c["open"].dropna()
    return (cl / o - 1.0).rolling(20, min_periods=10).mean()


def f_corr_ret_vol(c):
    cl, v = c["close"].dropna(), c["vol"].dropna()
    r = cl.pct_change()
    lv = np.log(v + eps)
    m = pd.concat([r, lv], axis=1).dropna()
    return m.iloc[:, 0].rolling(20, min_periods=10).corr(m.iloc[:, 1])


def f_dollar_vol_trend(c):
    cl, v = c["close"].dropna(), c["vol"].dropna()
    dv = (cl * v).rolling(20, min_periods=10).mean()
    return dv / dv.shift(60) - 1.0


for s in lib.WATCH:
    closes[s] = pd.Series(closes[s], name="close")
    opens[s] = pd.Series(opens[s], name="open")
    highs[s] = pd.Series(highs[s], name="high")
    lows[s] = pd.Series(lows[s], name="low")
    vols[s] = pd.Series(vols[s], name="vol")

C["range_pos_ohlc_20"] = build({s: {"high": highs[s], "low": lows[s], "close": closes[s]} for s in lib.WATCH}, f_range_pos)
C["overnight_ret_20"] = build({s: {"close": closes[s], "open": opens[s]} for s in lib.WATCH}, f_overnight)
C["intraday_ret_20"] = build({s: {"close": closes[s], "open": opens[s]} for s in lib.WATCH}, f_intraday)
C["corr_ret_vol_20"] = build({s: {"close": closes[s], "vol": vols[s]} for s in lib.WATCH}, f_corr_ret_vol)
C["dollar_vol_trend_20x60"] = build({s: {"close": closes[s], "vol": vols[s]} for s in lib.WATCH}, f_dollar_vol_trend)

# ---------- shared forward returns + library signals (once) ----------
horizons = (1, 2, 3, 5, 10, 20)
fwd = {h: lib.fwd_returns(panel, h) for h in horizons}
libs = lib.library_signals(panel)
LIB_IDS = list(libs.keys())


def fast_library_corr(factor):
    per = {}
    for fid, lf in libs.items():
        cs = []
        common = factor.index.intersection(lf.index)
        for dt in common[-400:]:
            f = factor.loc[dt]
            g = lf.loc[dt]
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            if int(m.sum()) >= lib.MIN_ASSETS:
                cs.append(pd.Series(f[m]).corr(pd.Series(g[m]), method="spearman"))
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


def run(name, factor, direction_override=None):
    factor = factor.reindex(panel.index).loc[:lib.FACTOR_LAST]
    if factor.notna().sum().sum() < 100:
        print(f"=== {name}: insufficient data ===")
        return None
    res = {"name": name, "factor_rows": len(factor), "n_assets": panel.shape[1]}
    ic_by_h = {h: lib.rank_ic_series(factor, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = direction_override if direction_override is not None else (
        float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0)
    ic_by_h = {h: ic * direction for h, ic in ic_by_h.items()}
    for h in horizons:
        ic = ic_by_h[h]
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean())
        res[f"n_dates_h{h}"] = int(len(ic))
    res["direction"] = direction
    valid = factor.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= lib.MIN_ASSETS).mean())
    res["turnover_10d_rank"] = lib.turnover_10d_rank(factor)
    max_corr, per = fast_library_corr(factor)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in horizons}
    gate_ic = abs(res["ic_h10"]) >= lib.ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= lib.ADMISSION["icir"]
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "pass": bool(gate_ic and gate_icir)}
    print(f"=== {name} === h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} "
          f"hit={res['hit_h10']:.3f} cov={res['coverage_asset_days']:.3f} "
          f"turn={res['turnover_10d_rank']:.2f} maxcorr={res['max_abs_library_correlation']:.3f} "
          f"decay={res['decay_ic_by_horizon']} -> {'PASS' if gate_ic and gate_icir else 'FAIL'}")
    return res


RESULTS = {}
for name, f in C.items():
    try:
        RESULTS[name] = run(name, f)
    except Exception as e:
        print(f"=== {name}: ERROR {type(e).__name__}: {e} ===")

json.dump(RESULTS, open("scripts/miner_2_cycle6_results.json", "w"), indent=1, default=str)
print("\nSAVED scripts/miner_2_cycle6_results.json")
print("\n===== SUMMARY =====")
for name, r in RESULTS.items():
    if r is None:
        continue
    p = r["admission_gate"]["pass"]
    print(f"{name:<26} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} -> {'PASS' if p else 'FAIL'}")
