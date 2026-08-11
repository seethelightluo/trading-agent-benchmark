"""miner_3 screening round 3 (2026-07-30): OHLC price-structure + cross-sectional candidates.
Universe: 15 tradable cross-asset instruments. Warm-up validation window: factor dates
2020-01-01..2026-07-15, data visible through 2026-07-29.
Admission gate (h=10): |IC| >= 0.007, |ICIR| >= 0.084. Also compute max_abs_library_correlation.
"""
from __future__ import annotations
import sys, io, zlib, base64, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner3_lib import WATCH, MIN_ASSETS_PER_DATE, fwd_returns, rank_ic_series, turnover_10d_rank

WARM_END = "2026-07-15"
MAX_VISIBLE = "2026-07-29"

# ---------- OHLCV panel ----------
_OHLCV = {}
def _ohlcv(sym):
    if sym in _OHLCV:
        return _OHLCV[sym]
    df = get_stock_daily_data(sym, days=4000)
    if df is not None and len(df):
        df = df.set_index("date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[:MAX_VISIBLE]
        _OHLCV[sym] = df
    else:
        _OHLCV[sym] = None
    return _OHLCV[sym]

def panel_col(name):
    cols = {}
    for s in WATCH:
        df = _ohlcv(s)
        if df is not None and name in df.columns:
            cols[s] = df[name].astype(float)
    return pd.DataFrame(cols).sort_index()

close = panel_col("close")
open_ = panel_col("open")
high = panel_col("high")
low = panel_col("low")
print(f"close panel: {close.shape}, {close.index.min().date()} .. {close.index.max().date()}")

# ---------- candidate factor definitions ----------
def per_asset(fn):
    def wrapper():
        cols = {}
        for a in close.columns:
            s = close[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=close.index)
    return wrapper

def cand_stochastic_pos_20():
    # mean closing location inside daily range over 20d, centered (0..1 -> -0.5..+0.5)
    def f(s):
        o, h, l = open_[s.name].reindex(s.index), high[s.name].reindex(s.index), low[s.name].reindex(s.index)
        rng = (h - l).replace(0, np.nan)
        pos = (s - l) / rng - 0.5
        return pos.rolling(20).mean()
    return per_asset(f)()

def cand_overnight_ret_20():
    # mean overnight gap return (open/prev_close - 1) over 20d
    def f(s):
        o = open_[s.name].reindex(s.index)
        gap = o / s.shift(1) - 1.0
        return gap.rolling(20).mean()
    return per_asset(f)()

def cand_intraday_ret_20():
    # mean intraday return (close/open - 1) over 20d
    def f(s):
        o = open_[s.name].reindex(s.index)
        intr = s / o - 1.0
        return intr.rolling(20).mean()
    return per_asset(f)()

def cand_overnight_vol_20():
    # std of overnight gap returns over 20d
    def f(s):
        o = open_[s.name].reindex(s.index)
        gap = o / s.shift(1) - 1.0
        return gap.rolling(20).std()
    return per_asset(f)()

def cand_intra_overnight_vol_ratio_20():
    # intraday vol / overnight vol over 20d (vol structure split)
    def f(s):
        o = open_[s.name].reindex(s.index)
        gap = o / s.shift(1) - 1.0
        intr = s / o - 1.0
        ov = gap.rolling(20).std()
        iv = intr.rolling(20).std()
        return iv / ov.replace(0, np.nan)
    return per_asset(f)()

def cand_rel_vol_cs_20():
    # 20d realized vol relative to cross-sectional median vol
    v = close.pct_change().rolling(20).std()
    med = v.median(axis=1)
    return v.sub(med, axis=0)

def cand_vol_adj_mom_60x20():
    # 60d momentum scaled by 20d vol (risk-adjusted trend)
    mom = close / close.shift(60) - 1.0
    vol = close.pct_change().rolling(20).std()
    return mom / vol.replace(0, np.nan)

def cand_rel_skew_20():
    # 20d skewness minus cross-sectional median skewness
    sk = close.pct_change().rolling(20).skew()
    med = sk.median(axis=1)
    return sk.sub(med, axis=0)

def cand_up_frac_20():
    # fraction of up days over 20d minus 0.5
    up = (close.pct_change() > 0).astype(float).rolling(20).mean() - 0.5
    return up

def cand_us10y_beta_cond_60x20():
    # beta to US10Y daily changes (60d) x 20d US10Y move; NaN for yield assets themselves
    us10 = close["US10Y"].dropna().pct_change()
    cols = {}
    for a in close.columns:
        if a in ("US10Y", "CN10Y"):
            cols[a] = np.nan
            continue
        s = close[a].dropna()
        r = s.pct_change()
        v = us10.reindex(s.index)
        z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
        us10_20 = (us10 / us10.shift(20) - 1.0).reindex(s.index)
        cols[a] = (-beta * us10_20).reindex(close.index)
    return pd.DataFrame(cols, index=close.index)

def cand_range_pos_vol_20():
    # closing location (0..1) multiplied by 20d vol: high-conviction trend positioning
    def f(s):
        o, h, l = open_[s.name].reindex(s.index), high[s.name].reindex(s.index), low[s.name].reindex(s.index)
        rng = (h - l).replace(0, np.nan)
        pos = (s - l) / rng - 0.5
        vol = s.pct_change().rolling(20).std()
        return pos.rolling(20).mean() * vol
    return per_asset(f)()

# ---------- library (existing 9 effective factors) ----------
def build_library():
    lib = {}
    ret = close.pct_change()
    mkt = ret.mean(axis=1)
    cols = {}
    for a in close.columns:
        s = ret[a].dropna(); m = mkt.reindex(s.index)
        z = pd.concat([s.rename("r"), m.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = pd.DataFrame(cols, index=close.index)
    lib["rel_mom_20d_skip5"] = (close.shift(5) / close.shift(25) - 1.0).sub(
        (close.shift(5) / close.shift(25) - 1.0).median(axis=1), axis=0)
    lib["mom_120d_skip5"] = close.shift(5) / close.shift(125) - 1.0
    lib["mom_10d_skip5"] = close.shift(5) / close.shift(15) - 1.0
    lib["vol_of_vol20x60"] = close.pct_change().rolling(20).std().rolling(60).std()
    def dsvr(s):
        r = s.pct_change()
        tot = r.rolling(20).std()
        dn = r.where(r < 0, 0.0).rolling(20).std()
        return -(dn / tot)
    lib["downside_vol_ratio_20"] = per_asset(dsvr)()
    lib["max_ret_20d"] = per_asset(lambda s: s.pct_change().rolling(20).max())()
    vix = None
    try:
        from alphacrafter.sim.utils import get_index_daily_data
        vdf = get_index_daily_data("VIX", days=4000)
        vix = vdf.set_index("date")["close"].astype(float)
        vix = vix[~vix.index.duplicated(keep="last")].sort_index().loc[:MAX_VISIBLE]
    except Exception:
        pass
    if vix is not None:
        cols = {}
        for a in close.columns:
            s = close[a].dropna(); r = s.pct_change()
            v = vix.pct_change().reindex(s.index)
            z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
            v20 = (vix / vix.shift(20) - 1.0).reindex(s.index)
            cols[a] = (-beta * v20).reindex(close.index)
        lib["vix_beta_cond_60x20"] = pd.DataFrame(cols, index=close.index)
    # amihud_20 needs volume (skip exact; approximate as NaN column -> ignored by corr)
    return lib

LIB = build_library()
print("library:", list(LIB.keys()))

def corr_with_lib(factor):
    fx = factor.stack().rename("f")
    out = {}
    for fid, lf in LIB.items():
        both = pd.concat([fx, lf.stack().rename("l")], axis=1).dropna()
        if len(both) >= 200:
            out[fid] = float(np.corrcoef(both["f"], both["l"])[0, 1])
    return out

def validate(name, factor, horizons=(5, 10, 20), direction_override=None):
    fw = factor.loc[:WARM_END]
    if fw.shape[0] < 200:
        print(f"=== {name}: TOO SHORT ({fw.shape[0]} rows) ===")
        return None
    fwd = {h: fwd_returns(close, h) for h in horizons}
    ic_by_h = {h: rank_ic_series(fw, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = direction_override if direction_override is not None else (
        float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0)
    ic_by_h = {h: ic * direction for h, ic in ic_by_h.items()}
    out = {"name": name, "direction": direction}
    for h in horizons:
        ic = ic_by_h[h]
        out[f"ic_h{h}"] = float(ic.mean())
        out[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        out[f"hit_h{h}"] = float((ic > 0).mean()) if len(ic) else float("nan")
        out[f"n_dates_h{h}"] = int(len(ic))
    valid = fw.notna()
    out["coverage_asset_days"] = float(valid.mean().mean())
    out["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    out["turnover_10d_rank"] = turnover_10d_rank(fw)
    corrs = corr_with_lib(factor.loc[:MAX_VISIBLE])
    out["max_abs_library_correlation"] = max((abs(v) for v in corrs.values()), default=float("nan"))
    out["library_corrs"] = {k: round(v, 3) for k, v in corrs.items()}
    print(f"=== {name} ===")
    print(f"  window {fw.index.min().date()}..{fw.index.max().date()} rows={len(fw)}")
    print(f"  direction={direction:+.3f} | " + "  ".join(
        f"h{h}: IC={out[f'ic_h{h}']:+.4f} ICIR={out[f'icir_h{h}']:+.4f} hit={out[f'hit_h{h}']:.3f}" for h in horizons))
    print(f"  coverage_asset_days={out['coverage_asset_days']:.3f} dates_ge8={out['coverage_dates_ge8']:.3f} "
          f"turnover={out['turnover_10d_rank']:.2f} max_abs_lib_corr={out['max_abs_library_correlation']:.3f}")
    print(f"  lib_corrs={out['library_corrs']}")
    return out

CANDIDATES = {
    "stochastic_pos_20": cand_stochastic_pos_20,
    "overnight_ret_20": cand_overnight_ret_20,
    "intraday_ret_20": cand_intraday_ret_20,
    "overnight_vol_20": cand_overnight_vol_20,
    "intra_overnight_vol_ratio_20": cand_intra_overnight_vol_ratio_20,
    "rel_vol_cs_20": cand_rel_vol_cs_20,
    "vol_adj_mom_60x20": cand_vol_adj_mom_60x20,
    "rel_skew_20": cand_rel_skew_20,
    "up_frac_20": cand_up_frac_20,
    "us10y_beta_cond_60x20": cand_us10y_beta_cond_60x20,
    "range_pos_vol_20": cand_range_pos_vol_20,
}

RESULTS = {}
for name, fn in CANDIDATES.items():
    try:
        f = fn()
        RESULTS[name] = validate(name, f)
    except Exception as e:
        print(f"=== {name}: ERROR {e} ===")

print("\n===== SUMMARY (h10 gate |IC|>=0.007, |ICIR|>=0.084) =====")
for name, r in RESULTS.items():
    if r is None:
        continue
    passed = abs(r["ic_h10"]) >= 0.007 and abs(r["icir_h10"]) >= 0.084
    lowcorr = r["max_abs_library_correlation"] < 0.5
    print(f"{name:<28} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"maxcorr={r['max_abs_library_correlation']:.3f} -> {'PASS' if passed else 'FAIL'} {'corr-ok' if lowcorr else 'CORR-HI'}")
