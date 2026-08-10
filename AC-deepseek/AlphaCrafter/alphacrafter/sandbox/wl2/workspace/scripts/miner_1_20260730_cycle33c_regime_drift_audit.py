"""miner_1 cycle 33c: regime-drift audit for close_loc_asym family before persistence.

The full-sample gate passes (IC=-0.03, ICIR=-0.12) but year splits show sign flip
in 2025. Check:
  1. Yearly regime splits for close_loc_asym_20 and close_loc_asym_20x60.
  2. Rolling 252d IC series (sign stability) for each variant.
  3. Recent 12M IC/ICIR (live-relevant window) and hit ratio.
  4. Half-sample A/B (first half vs second half) to quantify stationarity.
  5. If a variant is robust (consistent sign across regimes incl. recent), persist
     with full provenance; otherwise record drift and do NOT persist blindly.
"""
import sys, json, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor, turnover_rank,
                         load_asset)

panel = load_panel()
union_idx = panel.index

# rebuild library (same as 33b)
close = panel
lib = {}
lib["mom20_volproxy60"] = per_asset(close, lambda s: (s.shift(5) / s.shift(25) - 1.0) / (1.0 + (s.shift(5) / s.shift(65) - 1.0).abs()))
def calmness_20(s):
    return s.pct_change().abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan, raw=True)
lib["calmness_20"] = per_asset(close, calmness_20)
lib["vol_of_vol20x60"] = per_asset(close, lambda s: s.pct_change().rolling(20).std().rolling(60).std())
lib["mom_10d_skip5"] = per_asset(close, lambda s: s.shift(5) / s.shift(15) - 1.0)
lib["mom_120d_skip5"] = per_asset(close, lambda s: s.shift(5) / s.shift(125) - 1.0)
dxy = macro_series("DXY"); usdjpy = macro_series("USDJPY"); vix = macro_series("VIX")
spx = macro_series("SPX")
def beta_cond(asset_close, driver_close, w=60, m=20):
    dcs = driver_close.reindex(asset_close.index).ffill()
    ar, dr = asset_close.pct_change(), dcs.pct_change()
    df = pd.concat([ar.rename("a"), dr.rename("d")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=max(int(w * 0.5), 15)).cov(df["d"]) / (df["d"].rolling(w, min_periods=max(int(w * 0.5), 15)).var() + 1e-12)
    return b * (dcs / dcs.shift(m) - 1.0).reindex(b.index)
lib["dxy_beta_cond_60x20"] = per_asset(close, beta_cond, dxy, 60, 20)
lib["usdjpy_beta_cond_120x60"] = per_asset(close, beta_cond, usdjpy, 120, 60)
lib["vix_beta_cond_60x20"] = per_asset(close, beta_cond, vix, 60, 20)
def downbeta_spx_60(s):
    sp = spx.reindex(s.index).ffill()
    ar, sr = s.pct_change(), sp.pct_change()
    df = pd.concat([ar.rename("a"), sr.rename("s")], axis=1).dropna()
    neg = df[df["s"] < 0]
    return neg["a"].rolling(60, min_periods=15).cov(neg["s"]) / (neg["s"].rolling(60, min_periods=15).var() + 1e-12)
lib["downbeta_spx_60"] = per_asset(close, downbeta_spx_60)
def lagbeta_spx_60(s):
    sp = spx.reindex(s.index).ffill()
    ar, sr = s.pct_change(), sp.pct_change().shift(1)
    df = pd.concat([ar.rename("a"), sr.rename("s")], axis=1).dropna()
    return df["a"].rolling(60, min_periods=15).cov(df["s"]) / (df["s"].rolling(60, min_periods=15).var() + 1e-12)
lib["lagbeta_spx_60"] = per_asset(close, lagbeta_spx_60)

frames = {a: load_asset(a) for a in TRADABLES}
def own_series(a, col):
    s = frames[a][col].astype(float)
    s.index = pd.to_datetime(frames[a]["date"].values)
    return s

def reindex_to_panel(series_dict):
    out = {}
    for a, s in series_dict.items():
        s.index = pd.to_datetime(s.index)
        out[a] = s.reindex(union_idx)
    return pd.DataFrame(out, index=union_idx)

def close_loc_asym(s_close, s_high, s_low, w, minp=None):
    if minp is None:
        minp = max(int(w * 0.5), 10)
    ratio = (s_high - s_close) / ((s_close - s_low) + 1e-12)
    return ratio.rolling(w, min_periods=minp).mean()

def build(w):
    d = {}
    for a in TRADABLES:
        c, hi, lo = own_series(a, "close"), own_series(a, "high"), own_series(a, "low")
        d[a] = close_loc_asym(c, hi, lo, w)
    return reindex_to_panel(d)

s60 = build(60); s20 = build(20)
spread = s20 - s60

fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(panel, h)
ret10 = fwd_cache["10"]

cands = {"close_loc_asym_60": s60, "close_loc_asym_20": s20, "close_loc_asym_20x60": spread}

print("=" * 100)
print("CYCLE 33c REGIME-DRIFT AUDIT")
print("=" * 100)

for name, sig in cands.items():
    ic_ser = compute_ic(sig, ret10, 8).dropna()
    years = ic_ser.index.year
    print("\n--- %s (full-sample IC=%.4f, ICIR=%.4f)" % (name, ic_ser.mean(),
          (ic_ser.mean() / ic_ser.std()) if ic_ser.std() > 0 else 0.0))
    parts = []
    for y in sorted(set(years)):
        sub = ic_ser[years == y]
        parts.append("%s: ic=%.4f icir=%.4f n=%d" % (y, sub.mean(),
                     (sub.mean() / sub.std()) if sub.std() > 0 else 0.0, len(sub)))
    print("   yearly:", "; ".join(parts))
    # half-sample
    half = len(ic_ser) // 2
    h1, h2 = ic_ser.iloc[:half], ic_ser.iloc[half:]
    print("   half1: ic=%.4f icir=%.4f | half2: ic=%.4f icir=%.4f" % (
        h1.mean(), (h1.mean()/h1.std()) if h1.std() > 0 else 0.0,
        h2.mean(), (h2.mean()/h2.std()) if h2.std() > 0 else 0.0))
    # recent 252d
    rec = ic_ser.iloc[-252:]
    print("   last252: ic=%.4f icir=%.4f hit=%.3f n=%d" % (
        rec.mean(), (rec.mean()/rec.std()) if rec.std() > 0 else 0.0,
        (rec < 0).mean(), len(rec)))
    # last 126d
    rec2 = ic_ser.iloc[-126:]
    print("   last126: ic=%.4f icir=%.4f hit=%.3f n=%d" % (
        rec2.mean(), (rec2.mean()/rec2.std()) if rec2.std() > 0 else 0.0,
        (rec2 < 0).mean(), len(rec2)))

print("\n" + "=" * 100)
print("RECENT-12M FULL VALIDATION (sign-agnostic vs direction-consistent):")
print("=" * 100)
for name, sig in cands.items():
    m = validate_factor(sig, panel, library=lib, fwd_cache=fwd_cache)
    to = turnover_rank(sig, step=10)
    print("[%s] full IC=%.4f ICIR=%.4f maxlib=%.3f turn10=%.3f" % (
        name, m["ic"], m["icir"], m.get("max_abs_library_correlation", float("nan")), to))
