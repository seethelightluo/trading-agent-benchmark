"""miner_3 screening round 5 (2026-07-30): new factor families.
Trend efficiency (Kaufman ER), vol term structure, serial autocorrelation,
return distribution tails (kurtosis / vol-burst), drawdown-vol, macro-beta
conditional (DXY), co-skewness, EW correlation, momentum-quality interaction,
upside semi-vol, range/Parkinson efficiency, volume z-score & volume trend.
Per-asset calendar-aware computation; validation window = warm-up ..2026-07-15.
Admission gate (h=10): |IC|>=0.007, |ICIR|>=0.084; max_abs_library_corr < 0.5.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (WATCH, load_panel, load_macro, fwd_returns,
                        rank_ic_series, turnover_10d_rank, MIN_ASSETS_PER_DATE)

WARM_END = "2026-07-15"
MAX_VISIBLE = "2026-07-29"

panel = load_panel().loc[:MAX_VISIBLE]
macro = load_macro()
print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets, range "
      f"{panel.index.min().date()}..{panel.index.max().date()}")

# ---------------------------------------------------------------- helpers
def per_asset(fn, p=None):
    p = panel if p is None else p
    cols = {}
    for a in p.columns:
        s = p[a].dropna()
        cols[a] = fn(s)
    return pd.DataFrame(cols, index=p.index)

def ret_panel():
    return per_asset(lambda s: s.pct_change())

R = ret_panel()
EW = R.mean(axis=1)

def rolling_beta(asset_ret, mkt_ret, w):
    z = pd.concat([asset_ret.rename("r"), mkt_ret.rename("m")], axis=1).dropna()
    return z["r"].rolling(w).cov(z["m"]) / z["m"].rolling(w).var()

def rolling_corr(asset_ret, mkt_ret, w):
    z = pd.concat([asset_ret.rename("r"), mkt_ret.rename("m")], axis=1).dropna()
    return z["r"].rolling(w).corr(z["m"])

def rolling_coskew(asset_ret, mkt_ret, w):
    z = pd.concat([asset_ret.rename("r"), mkt_ret.rename("m")], axis=1).dropna()
    mu_r = z["r"].rolling(w).mean(); mu_m = z["m"].rolling(w).mean()
    var_m = z["m"].rolling(w).var()
    return ((z["r"] - mu_r) * (z["m"] - mu_m) ** 2).rolling(w).mean() / (z["m"].rolling(w).std() ** 3)

# ---------------------------------------------------------------- candidates
def cand_eff_ratio_20():
    def f(s):
        mom = (s - s.shift(20)).abs()
        path = s.diff().abs().rolling(20).sum()
        return mom / path.replace(0, np.nan)
    return per_asset(f)

def cand_eff_ratio_60():
    def f(s):
        mom = (s - s.shift(60)).abs()
        path = s.diff().abs().rolling(60).sum()
        return mom / path.replace(0, np.nan)
    return per_asset(f)

def cand_vol_term_20x60():
    def f(s):
        v20 = s.pct_change().rolling(20).std()
        v60 = s.pct_change().rolling(60).std()
        return np.log(v20 / v60.replace(0, np.nan))
    return per_asset(f)

def cand_autocorr_5_60():
    def f(s):
        r = s.pct_change()
        return r.rolling(60).apply(lambda x: pd.Series(x).autocorr(lag=5), raw=False)
    return per_asset(f)

def cand_kurtosis_20():
    return per_asset(lambda s: s.pct_change().rolling(20).kurt())

def cand_vol_burst_20():
    def f(s):
        r = s.pct_change()
        sig = r.rolling(60).std()
        return (r.abs() > 1.5 * sig).astype(float).rolling(20).mean()
    return per_asset(f)

def cand_dd_vol_60():
    def f(s):
        dd = s / s.rolling(60).max() - 1.0
        vol = s.pct_change().rolling(20).std()
        return dd / vol.replace(0, np.nan)
    return per_asset(f)

def cand_dxy_beta_cond_60x10():
    dxy = macro.get("DXY")
    if dxy is None:
        return pd.DataFrame(index=panel.index, columns=panel.columns)
    def f(s):
        r = s.pct_change()
        d = dxy.pct_change().reindex(s.index)
        z = pd.concat([r.rename("r"), d.rename("d")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["d"]) / z["d"].rolling(60).var()
        dmom = (dxy / dxy.shift(10) - 1.0).reindex(s.index)
        return (beta * dmom).reindex(panel.index)
    cols = {a: f(panel[a].dropna()) for a in panel.columns}
    return pd.DataFrame(cols, index=panel.index)

def cand_coskew_ew_60():
    cols = {}
    for a in panel.columns:
        r = R[a].dropna()
        m = EW.reindex(r.index)
        cols[a] = rolling_coskew(r, m, 60)
    return pd.DataFrame(cols, index=panel.index)

def cand_corr_ew_60():
    cols = {}
    for a in panel.columns:
        r = R[a].dropna()
        m = EW.reindex(r.index)
        cols[a] = rolling_corr(r, m, 60)
    return pd.DataFrame(cols, index=panel.index)

def cand_mom_quality_20():
    def f(s):
        mom = s.shift(5) / s.shift(25) - 1.0
        er = (s - s.shift(20)).abs() / s.diff().abs().rolling(20).sum().replace(0, np.nan)
        return mom * er
    return per_asset(f)

def cand_vix_beta_60():
    vix = macro.get("VIX")
    if vix is None:
        return pd.DataFrame(index=panel.index, columns=panel.columns)
    cols = {}
    for a in panel.columns:
        r = R[a].dropna()
        v = vix.pct_change().reindex(r.index)
        cols[a] = rolling_beta(r, v, 60)
    return pd.DataFrame(cols, index=panel.index)

def cand_semiv_up_ratio_20():
    def f(s):
        r = s.pct_change()
        up = r.where(r > 0, 0.0)
        return up.rolling(20).std() / r.rolling(20).std().replace(0, np.nan)
    return per_asset(f)

def cand_range_vol_ratio_20():
    def f(s):
        h = high[a].reindex(s.index); l = low[a].reindex(s.index)
        park = np.log(h / l).rolling(20).std()
        cc = s.pct_change().rolling(20).std()
        return park / cc.replace(0, np.nan)
    high = panel  # placeholder replaced in loop below
    return None

def cand_volume_z_20():
    def f(s):
        v = s.rolling(20).mean()
        return (v - v.rolling(60).mean()) / v.rolling(60).std().replace(0, np.nan)
    return per_asset(f, vol_panel())

def cand_volume_trend_5x20():
    def f(s):
        v = s.rolling(5).mean() / s.rolling(20).mean().replace(0, np.nan)
        return v
    return per_asset(f, vol_panel())

VOL = None
def vol_panel():
    global VOL
    if VOL is None:
        import alphacrafter.sim.utils as U
        cols = {}
        for a in WATCH:
            df = U.get_stock_daily_data(a, days=4000)
            if df is not None and len(df) and df["volume"].astype(float).gt(0).mean() > 0.5:
                v = df.set_index("date")["volume"].astype(float)
                v = v[~v.index.duplicated(keep="last")].sort_index().loc[:MAX_VISIBLE]
                cols[a] = v
        VOL = pd.DataFrame(cols).sort_index()
        print("volume panel assets:", list(VOL.columns))
    return VOL

# range_vol_ratio needs OHLC; build properly
def cand_range_vol_ratio_20():
    import alphacrafter.sim.utils as U
    cols = {}
    for a in WATCH:
        df = U.get_stock_daily_data(a, days=4000)
        if df is None or not len(df):
            continue
        d = df.set_index("date")[["open", "close", "high", "low"]].astype(float)
        d = d[~d.index.duplicated(keep="last")].sort_index().loc[:MAX_VISIBLE]
        s = d["close"].dropna()
        h, l = d["high"].reindex(s.index), d["low"].reindex(s.index)
        park = np.log(h / l).rolling(20).std()
        cc = s.pct_change().rolling(20).std()
        cols[a] = (park / cc.replace(0, np.nan)).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)

# ---------------------------------------------------------------- library
def build_library():
    lib = {}
    lib["mom_10d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(15) - 1.0)
    lib["mom_120d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(125) - 1.0)
    lib["rel_mom_20d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)
    lib["rel_mom_20d_skip5"] = lib["rel_mom_20d_skip5"].sub(
        lib["rel_mom_20d_skip5"].median(axis=1), axis=0)
    lib["beta_ew_60d"] = pd.DataFrame(
        {a: rolling_beta(R[a].dropna(), EW.reindex(R[a].dropna().index), 60)
         for a in panel.columns}, index=panel.index)
    lib["vol_of_vol20x60"] = per_asset(
        lambda s: s.pct_change().rolling(20).std().rolling(60).std())
    lib["max_ret_20d"] = per_asset(lambda s: s.pct_change().rolling(20).max())
    lib["downside_vol_ratio_20"] = per_asset(
        lambda s: -(s.pct_change().where(s.pct_change() < 0, 0.0).rolling(20).std()
                    / s.pct_change().rolling(20).std()))
    vix = macro.get("VIX")
    if vix is not None:
        cols = {}
        for a in panel.columns:
            r = R[a].dropna()
            v = vix.pct_change().reindex(r.index)
            beta = rolling_beta(r, v, 60)
            vix20 = (vix / vix.shift(20) - 1.0).reindex(r.index)
            cols[a] = (-beta * vix20)
        lib["vix_beta_cond_60x20"] = pd.DataFrame(cols, index=panel.index)
    vp = vol_panel()
    if len(vp.columns) > 3:
        cols = {}
        for a in vp.columns:
            v = vp[a].dropna()
            amihud = (R[a].reindex(v.index).abs() / v).rolling(20).mean()
            cols[a] = amihud.reindex(panel.index)
        lib["amihud_20"] = pd.DataFrame(cols, index=panel.index)
    return lib

LIB = build_library()
print("library factors:", list(LIB.keys()))

def corr_with_lib(factor):
    fx = factor.stack().rename("f")
    out = {}
    for fid, lf in LIB.items():
        both = pd.concat([fx, lf.stack().rename("l")], axis=1).dropna()
        if len(both) >= 200:
            out[fid] = float(np.corrcoef(both["f"], both["l"])[0, 1])
    return out

# ---------------------------------------------------------------- validation
def validate(name, factor):
    fw = factor.loc[:WARM_END]
    fwd = {h: fwd_returns(panel, h) for h in (5, 10, 20)}
    ic_by_h = {h: rank_ic_series(fw, fwd[h]) for h in fwd}
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    ic_by_h = {h: ic * direction for h, ic in ic_by_h.items()}
    out = {"name": name, "direction": direction}
    for h in fwd:
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
    print(f"  dir={direction:+.2f} | " + "  ".join(
        f"h{h}: IC={out[f'ic_h{h}']:+.4f} ICIR={out[f'icir_h{h}']:+.4f} hit={out[f'hit_h{h}']:.3f} n={out[f'n_dates_h{h}']}"
        for h in (5, 10, 20)))
    print(f"  cov_asset={out['coverage_asset_days']:.3f} cov_dates={out['coverage_dates_ge8']:.3f} "
          f"turn={out['turnover_10d_rank']:.2f} maxcorr={out['max_abs_library_correlation']:.3f} corrs={out['library_corrs']}")
    return out

CANDIDATES = {
    "eff_ratio_20": cand_eff_ratio_20,
    "eff_ratio_60": cand_eff_ratio_60,
    "vol_term_20x60": cand_vol_term_20x60,
    "autocorr_5_60": cand_autocorr_5_60,
    "kurtosis_20": cand_kurtosis_20,
    "vol_burst_20": cand_vol_burst_20,
    "dd_vol_60": cand_dd_vol_60,
    "dxy_beta_cond_60x10": cand_dxy_beta_cond_60x10,
    "coskew_ew_60": cand_coskew_ew_60,
    "corr_ew_60": cand_corr_ew_60,
    "mom_quality_20": cand_mom_quality_20,
    "vix_beta_60": cand_vix_beta_60,
    "semiv_up_ratio_20": cand_semiv_up_ratio_20,
    "range_vol_ratio_20": cand_range_vol_ratio_20,
    "volume_z_20": cand_volume_z_20,
    "volume_trend_5x20": cand_volume_trend_5x20,
}

RESULTS = {}
for name, fn in CANDIDATES.items():
    try:
        RESULTS[name] = validate(name, fn())
    except Exception as e:
        print(f"=== {name}: ERROR {type(e).__name__}: {e} ===")

print("\n===== SUMMARY (h10 gate |IC|>=0.007, |ICIR|>=0.084, corr<0.5) =====")
for name, r in RESULTS.items():
    if r is None:
        continue
    passed = abs(r["ic_h10"]) >= 0.007 and abs(r["icir_h10"]) >= 0.084
    lowcorr = r["max_abs_library_correlation"] < 0.5
    print(f"{name:<24} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"maxcorr={r['max_abs_library_correlation']:.3f} -> {'PASS' if passed else 'FAIL'} "
          f"{'corr-ok' if lowcorr else 'CORR-HI'}")
