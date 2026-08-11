"""miner_3 screening round 4 (2026-07-30): fixed per-asset-calendar computation.
All per-asset moments computed on each asset's own calendar BEFORE concat to union panel.
Candidates: skew/up-pressure/stochastic variants + wick asymmetry + composite trend quality.
Admission gate (h=10): |IC| >= 0.007, |ICIR| >= 0.084; max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner3_lib import WATCH, MIN_ASSETS_PER_DATE, fwd_returns, rank_ic_series, turnover_10d_rank

WARM_END = "2026-07-15"
MAX_VISIBLE = "2026-07-29"

_OHLCV = {}
def _ohlcv(sym):
    if sym in _OHLCV:
        return _OHLCV[sym]
    df = get_stock_daily_data(sym, days=4000)
    if df is not None and len(df):
        df = df.set_index("date")
        df = df[~df.index.duplicated(keep="last")].sort_index().loc[:MAX_VISIBLE]
        _OHLCV[sym] = df
    else:
        _OHLCV[sym] = None
    return _OHLCV[sym]

def panel_of(name):
    """Union panel of per-asset series computed on each asset's own calendar."""
    cols = {}
    for s in WATCH:
        df = _ohlcv(s)
        if df is not None and name in df.columns:
            cols[s] = df[name].astype(float)
    return pd.DataFrame(cols).sort_index()

close = panel_of("close")
open_ = panel_of("open")
high = panel_of("high")
low = panel_of("low")

def per_asset(fn):
    """fn(s: per-asset close Series, sname) -> Series on asset calendar; returns union panel."""
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        cols[a] = fn(s, a)
    return pd.DataFrame(cols, index=close.index)

def rets_of(s, sname):
    return s.pct_change()

# ---------------- candidates ----------------
def cand_rel_skew_20():
    def f(s, a):
        return s.pct_change().rolling(20).skew()
    sk = per_asset(f)
    med = sk.median(axis=1)
    return sk.sub(med, axis=0)

def cand_up_frac_20():
    def f(s, a):
        return (s.pct_change() > 0).astype(float).rolling(20).mean() - 0.5
    return per_asset(f)

def cand_up_frac_60():
    def f(s, a):
        return (s.pct_change() > 0).astype(float).rolling(60).mean() - 0.5
    return per_asset(f)

def cand_stochastic_pos_10():
    def f(s, a):
        o, h, l = open_[a].reindex(s.index), high[a].reindex(s.index), low[a].reindex(s.index)
        rng = (h - l).replace(0, np.nan)
        pos = (s - l) / rng - 0.5
        return pos.rolling(10).mean()
    return per_asset(f)

def cand_stochastic_pos_60():
    def f(s, a):
        o, h, l = open_[a].reindex(s.index), high[a].reindex(s.index), low[a].reindex(s.index)
        rng = (h - l).replace(0, np.nan)
        pos = (s - l) / rng - 0.5
        return pos.rolling(60).mean()
    return per_asset(f)

def cand_overnight_ret_60():
    def f(s, a):
        o = open_[a].reindex(s.index)
        gap = o / s.shift(1) - 1.0
        return gap.rolling(60).mean()
    return per_asset(f)

def cand_intraday_ret_60():
    def f(s, a):
        o = open_[a].reindex(s.index)
        intr = s / o - 1.0
        return intr.rolling(60).mean()
    return per_asset(f)

def cand_intraday_ret_20_demean():
    def f(s, a):
        o = open_[a].reindex(s.index)
        return (s / o - 1.0).rolling(20).mean()
    raw = per_asset(f)
    med = raw.median(axis=1)
    return raw.sub(med, axis=0)

def cand_wick_ratio_20():
    # lower-wick share minus upper-wick share, averaged over 20d (bullish rejection asymmetry)
    def f(s, a):
        o, h, l = open_[a].reindex(s.index), high[a].reindex(s.index), low[a].reindex(s.index)
        rng = (h - l).replace(0, np.nan)
        lower = (o - l) / rng
        upper = (h - o) / rng
        return (lower - upper).rolling(20).mean()
    return per_asset(f)

def cand_dd_vol_20x20():
    # vol-scaled distance below 20d high (negative when below high)
    def f(s, a):
        dd = s / s.rolling(20).max() - 1.0
        vol = s.pct_change().rolling(20).std()
        return dd / vol.replace(0, np.nan)
    return per_asset(f)

def cand_trend_quality_20():
    # composite: average of stochastic_pos_20 and up_frac_20 (z-scored per date)
    sp = cand_stochastic_pos_10  # placeholder replaced below
    def f1(s, a):
        o, h, l = open_[a].reindex(s.index), high[a].reindex(s.index), low[a].reindex(s.index)
        rng = (h - l).replace(0, np.nan)
        return ((s - l) / rng - 0.5).rolling(20).mean()
    def f2(s, a):
        return (s.pct_change() > 0).astype(float).rolling(20).mean() - 0.5
    a = per_asset(f1)
    b = per_asset(f2)
    za = a.sub(a.median(axis=1), axis=0).div(a.std(axis=1).replace(0, np.nan), axis=0)
    zb = b.sub(b.median(axis=1), axis=0).div(b.std(axis=1).replace(0, np.nan), axis=0)
    return (za + zb) / 2.0

# ---------------- library ----------------
def build_library():
    lib = {}
    def pa(fn):
        cols = {}
        for a in close.columns:
            cols[a] = fn(close[a].dropna(), a)
        return pd.DataFrame(cols, index=close.index)
    lib["rel_mom_20d_skip5"] = pa(lambda s, a: s.shift(5) / s.shift(25) - 1.0)
    lib["rel_mom_20d_skip5"] = lib["rel_mom_20d_skip5"].sub(lib["rel_mom_20d_skip5"].median(axis=1), axis=0)
    lib["mom_120d_skip5"] = pa(lambda s, a: s.shift(5) / s.shift(125) - 1.0)
    lib["mom_10d_skip5"] = pa(lambda s, a: s.shift(5) / s.shift(15) - 1.0)
    lib["beta_ew_60d"] = pa(lambda s, a: s.pct_change().rolling(60).mean())
    # beta vs EW market (computed on per-asset calendars is complex; approximate with union rets)
    ret = pa(lambda s, a: s.pct_change())
    mkt = ret.mean(axis=1)
    cols = {}
    for a in close.columns:
        s = ret[a].dropna(); m = mkt.reindex(s.index)
        z = pd.concat([s.rename("r"), m.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = pd.DataFrame(cols, index=close.index)
    lib["vol_of_vol20x60"] = pa(lambda s, a: s.pct_change().rolling(20).std().rolling(60).std())
    lib["max_ret_20d"] = pa(lambda s, a: s.pct_change().rolling(20).max())
    lib["downside_vol_ratio_20"] = pa(lambda s, a: -(s.pct_change().where(s.pct_change() < 0, 0.0).rolling(20).std()
                                                   / s.pct_change().rolling(20).std()))
    vix = None
    try:
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

def validate(name, factor, horizons=(5, 10, 20)):
    fw = factor.loc[:WARM_END]
    fwd = {h: fwd_returns(close, h) for h in horizons}
    ic_by_h = {h: rank_ic_series(fw, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
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
    print(f"  dir={direction:+.2f} | " + "  ".join(
        f"h{h}: IC={out[f'ic_h{h}']:+.4f} ICIR={out[f'icir_h{h}']:+.4f} hit={out[f'hit_h{h}']:.3f} n={out[f'n_dates_h{h}']}" for h in horizons))
    print(f"  cov_asset={out['coverage_asset_days']:.3f} cov_dates={out['coverage_dates_ge8']:.3f} "
          f"turn={out['turnover_10d_rank']:.2f} maxcorr={out['max_abs_library_correlation']:.3f} corrs={out['library_corrs']}")
    return out

CANDIDATES = {
    "rel_skew_20": cand_rel_skew_20,
    "up_frac_20": cand_up_frac_20,
    "up_frac_60": cand_up_frac_60,
    "stochastic_pos_10": cand_stochastic_pos_10,
    "stochastic_pos_60": cand_stochastic_pos_60,
    "overnight_ret_60": cand_overnight_ret_60,
    "intraday_ret_60": cand_intraday_ret_60,
    "intraday_ret_20_demean": cand_intraday_ret_20_demean,
    "wick_ratio_20": cand_wick_ratio_20,
    "dd_vol_20x20": cand_dd_vol_20x20,
    "trend_quality_20": cand_trend_quality_20,
}

RESULTS = {}
for name, fn in CANDIDATES.items():
    try:
        RESULTS[name] = validate(name, fn())
    except Exception as e:
        print(f"=== {name}: ERROR {e} ===")

print("\n===== SUMMARY (h10 gate |IC|>=0.007, |ICIR|>=0.084, corr<0.5) =====")
for name, r in RESULTS.items():
    if r is None:
        continue
    passed = abs(r["ic_h10"]) >= 0.007 and abs(r["icir_h10"]) >= 0.084
    lowcorr = r["max_abs_library_correlation"] < 0.5
    print(f"{name:<26} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"maxcorr={r['max_abs_library_correlation']:.3f} -> {'PASS' if passed else 'FAIL'} {'corr-ok' if lowcorr else 'CORR-HI'}")
