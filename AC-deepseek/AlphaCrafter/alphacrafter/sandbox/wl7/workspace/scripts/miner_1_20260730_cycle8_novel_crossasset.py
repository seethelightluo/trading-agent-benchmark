"""miner_1 cycle 8: NOVEL cross-asset / regime-sensitivity factor families.
Universe: 15 tradable cross-asset instruments (window 2020-01-01..2026-07-15).
Admission gates (benchmark contract): |IC|>=0.007 and |ICIR|>=0.084 @ h=10.

Families:
 A) Cross-asset beta (sensitivity to tradable leaders): BTC, XAU, COPPER, WTI, USDJPY
 B) Volatility term structure / candle shape: vol_term_ratio_5x60, candle_body_eff_20,
    tail_freq_20
 C) Correlation regime shifts: spx_corr_change_60x250, vix_corr_change_60x250
 D) Conditional momentum (dollar regime interaction), breadth: mom_dxy_cond_20x60,
    up_day_ratio_20
 E) Volume dynamics: vol_autocorr_20, volume_vol_of_vol_20x60

Library correlation audit against ALL 10 currently effective library factors.
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
import miner_2_lib as lib

EPS = 1e-12
t0 = time.time()

panel = lib.load_panel()
macro = lib.load_macro()
rets = panel.pct_change()
mkt_r = panel.mean(axis=1).pct_change()
print(f"panel {panel.index[0].date()}..{panel.index[-1].date()} assets={panel.shape[1]} rows={len(panel)} ({time.time()-t0:.1f}s)", flush=True)

# volume panel (union calendar)
vol_panel = None
try:
    vols = {}
    for a in lib.WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= lib.MAX_VISIBLE].set_index("date").sort_index()
        vols[a] = df["volume"].astype(float)
    vol_panel = pd.DataFrame(vols, index=panel.index)
    print("volume panel ok", flush=True)
except Exception as e:
    print("volume load failed:", e, flush=True)


def rolling_beta(x: pd.DataFrame, m: pd.Series, win=60, minp=30) -> pd.DataFrame:
    cov = x.rolling(win, min_periods=minp).cov(m)
    var = m.rolling(win, min_periods=minp).var()
    return cov / var.replace(0, np.nan)


C = {}

# ---------- Family A: cross-asset beta (sensitivity to tradable leaders) ----------
for leader, name in [("BTC", "btc_beta_60d"), ("XAU", "xau_beta_60d"),
                     ("COPPER", "copper_beta_60d"), ("WTI", "wti_beta_60d"),
                     ("USDJPY", "usdjpy_beta_60d")]:
    if leader == "USDJPY":
        lr = macro["USDJPY"].pct_change()
    else:
        lr = panel[leader].pct_change()
    b = rolling_beta(rets, lr, 60, 30)
    b[leader] = 1.0  # benchmark's beta to itself
    C[name] = b

# ---------- Family B: vol term structure / candle shape / tail ----------
v5 = rets.rolling(5, min_periods=3).std()
v60 = rets.rolling(60, min_periods=30).std()
C["vol_term_ratio_5x60"] = v5 / (v60 + EPS)

# candle body efficiency: mean((close-open)/(high-low)) over 20d (per-asset calendar)
candle_cols = {}
for a in lib.WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= lib.MAX_VISIBLE].set_index("date").sort_index()
    body = (df["close"] - df["open"]) / (df["high"] - df["low"] + EPS)
    candle_cols[a] = body.rolling(20, min_periods=10).mean()
C["candle_body_eff_20"] = pd.DataFrame(candle_cols, index=panel.index)

# tail frequency: share of days in 20d with |ret| > 1.5 * 60d rolling std
thr = 1.5 * rets.rolling(60, min_periods=30).std()
C["tail_freq_20"] = (rets.abs() > thr).rolling(20, min_periods=10).mean()

# ---------- Family C: correlation regime shifts ----------
spx_r = panel["SPX"].pct_change()
corr60 = rets.rolling(60, min_periods=30).corr(spx_r)
corr250 = rets.rolling(250, min_periods=120).corr(spx_r)
C["spx_corr_change_60x250"] = corr60 - corr250
vixr = macro["VIX"].pct_change()
vcorr60 = rets.rolling(60, min_periods=30).corr(vixr)
vcorr250 = rets.rolling(250, min_periods=120).corr(vixr)
C["vix_corr_change_60x250"] = vcorr60 - vcorr250

# ---------- Family D: conditional momentum / breadth ----------
m20 = panel.shift(5) / panel.shift(25) - 1.0
dxy_chg60 = macro["DXY"] / macro["DXY"].shift(60) - 1.0
C["mom_dxy_cond_20x60"] = m20 * np.sign(dxy_chg60)
C["up_day_ratio_20"] = (rets > 0).rolling(20, min_periods=10).mean()

# ---------- Family E: volume dynamics ----------
if vol_panel is not None:
    vchg = vol_panel.pct_change()
    # lag-1 autocorrelation of volume changes (rolling 20d, vectorized approx)
    y = vchg.shift(1)
    n = vchg.rolling(20, min_periods=10).count()
    sx = vchg.rolling(20, min_periods=10).sum()
    sy = y.rolling(20, min_periods=10).sum()
    sxy = (vchg * y).rolling(20, min_periods=10).sum()
    sxx = (vchg * vchg).rolling(20, min_periods=10).sum()
    syy = (y * y).rolling(20, min_periods=10).sum()
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    C["vol_autocorr_20"] = num / (den + EPS)
    C["volume_vol_of_vol_20x60"] = vchg.rolling(20, min_periods=10).std().rolling(60, min_periods=30).std()

print(f"candidates built: {list(C.keys())} ({time.time()-t0:.1f}s)", flush=True)

# ---------- shared forward returns ----------
horizons = (1, 2, 3, 5, 10, 20)
fwd = {h: lib.fwd_returns(panel, h) for h in horizons}

# ---------- library signals: ALL 10 effective factors ----------
def library_signals_all():
    out = {}
    out["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    out["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    v = rets.rolling(20, min_periods=10).std()
    out["vol_of_vol20x60"] = v.rolling(60, min_periods=30).std()
    out["max_ret_20d"] = rets.rolling(20, min_periods=10).max()
    dd = rets.clip(upper=0).rolling(20, min_periods=10).std()
    out["downside_vol_ratio_20"] = -(dd / (v + EPS))
    out["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    out["beta_ew_60d"] = rolling_beta(rets, mkt_r, 60, 30)
    out["vix_beta_cond_60x20"] = -rolling_beta(rets, vixr, 60, 30) * (macro["VIX"] / macro["VIX"].shift(20) - 1.0)
    mom20 = panel.shift(5) / panel.shift(25) - 1.0
    out["vol_adj_mom_20x60"] = mom20 / (v + EPS)
    if vol_panel is not None:
        out["amihud_20"] = (rets.abs() / vol_panel.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    return out

libs = library_signals_all()
LIB_IDS = list(libs.keys())
print(f"library signals: {LIB_IDS}", flush=True)


def library_corr_full(factor: pd.DataFrame) -> tuple:
    per = {}
    common = factor.index.intersection(panel.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-500:]:
            f = factor.loc[dt]
            g = lf.loc[dt].reindex(f.index)
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            m = m.reindex(f.index).fillna(False)
            if int(m.sum()) >= lib.MIN_ASSETS:
                cs.append(spearmanr(f[m], g[m])[0])
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


def run(name: str, factor: pd.DataFrame):
    factor = factor.reindex(panel.index).loc[:lib.FACTOR_LAST]
    if factor.notna().sum().sum() < 100:
        print(f"=== {name}: insufficient data ===", flush=True)
        return None
    res = {"name": name, "factor_rows": len(factor), "n_assets": panel.shape[1]}
    ic_by_h = {h: lib.rank_ic_series(factor, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
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
    max_corr, per = library_corr_full(factor)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in horizons}
    gate_ic = abs(res["ic_h10"]) >= lib.ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= lib.ADMISSION["icir"]
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "pass": bool(gate_ic and gate_icir)}
    flag = "PASS" if (gate_ic and gate_icir) else "FAIL"
    print(f"=== {name} === h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} "
          f"hit={res['hit_h10']:.3f} cov={res['coverage_asset_days']:.3f} "
          f"turn={res['turnover_10d_rank']:.2f} maxcorr={res['max_abs_library_correlation']:.3f} "
          f"decay={res['decay_ic_by_horizon']} -> {flag}", flush=True)
    return res


RESULTS = {}
for name, f in C.items():
    try:
        RESULTS[name] = run(name, f)
    except Exception as e:
        print(f"=== {name}: ERROR {type(e).__name__}: {e} ===", flush=True)

import json
json.dump(RESULTS, open("scripts/miner_1_cycle8_results.json", "w"), indent=1, default=str)
print(f"\nSAVED scripts/miner_1_cycle8_results.json ({time.time()-t0:.1f}s)", flush=True)
print("\n===== SUMMARY =====", flush=True)
for name, r in RESULTS.items():
    if r is None:
        continue
    p = r["admission_gate"]["pass"]
    print(f"{name:<26} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"maxcorr={r['max_abs_library_correlation']:.3f} turn={r['turnover_10d_rank']:.2f} "
          f"-> {'PASS' if p else 'FAIL'}", flush=True)
