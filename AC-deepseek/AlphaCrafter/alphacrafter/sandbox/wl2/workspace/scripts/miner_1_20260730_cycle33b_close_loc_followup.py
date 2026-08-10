"""miner_1 cycle 33b: focused follow-up on close_loc_asym_60 family + fixes.

Cycle33 uncovered ONE genuine library-passing candidate: close_loc_asym_60
(IC=-0.0294, ICIR=-0.1112, maxlib=0.349, direction negative).
This script:
  1. Re-validates close_loc_asym_60 against the FULL ACTIVE 10-JSON library
     (calmness_20, downbeta_spx_60, dxy_beta_cond_60x20, lagbeta_spx_60,
      mom20_volproxy60, mom_10d_skip5, mom_120d_skip5, usdjpy_beta_cond_120x60,
      vix_beta_cond_60x20, vol_of_vol20x60) with explicit turnover.
  2. Tests close_loc_asym_20 and 20x60 spread variants.
  3. Fixes semi_vol_asym_60 (per-asset min_periods bug) and re-tests it.
  4. Reports regime (yearly) splits for the winner.
  5. If winner passes gate: writes JSON + npy artifact (raw, 2398x15) and
     appends to factor_library_audit.jsonl.
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

# ---------------------------------------------------------------------------
# Rebuild ACTIVE library (10 JSONs) exactly as persisted signals
# ---------------------------------------------------------------------------
close = panel
lib = {}
# mom20_volproxy60
lib["mom20_volproxy60"] = per_asset(close, lambda s: (s.shift(5) / s.shift(25) - 1.0) / (1.0 + (s.shift(5) / s.shift(65) - 1.0).abs()))
# calmness_20
def calmness_20(s):
    return s.pct_change().abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan, raw=True)
lib["calmness_20"] = per_asset(close, calmness_20)
# vol_of_vol20x60
lib["vol_of_vol20x60"] = per_asset(close, lambda s: s.pct_change().rolling(20).std().rolling(60).std())
# mom_10d_skip5
lib["mom_10d_skip5"] = per_asset(close, lambda s: s.shift(5) / s.shift(15) - 1.0)
# mom_120d_skip5
lib["mom_120d_skip5"] = per_asset(close, lambda s: s.shift(5) / s.shift(125) - 1.0)
# macro drivers
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

# ---------------------------------------------------------------------------
# Candidate definitions from raw OHLC
# ---------------------------------------------------------------------------
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

cands = {}
c60 = {}
for a in TRADABLES:
    c, hi, lo = own_series(a, "close"), own_series(a, "high"), own_series(a, "low")
    c60[a] = close_loc_asym(c, hi, lo, 60)
cands["close_loc_asym_60"] = reindex_to_panel(c60)

c20 = {}
for a in TRADABLES:
    c, hi, lo = own_series(a, "close"), own_series(a, "high"), own_series(a, "low")
    c20[a] = close_loc_asym(c, hi, lo, 20)
cands["close_loc_asym_20"] = reindex_to_panel(c20)

# spread: 20d minus 60d (recent intraday asymmetry relative to baseline)
cs = {}
for a in TRADABLES:
    c, hi, lo = own_series(a, "close"), own_series(a, "high"), own_series(a, "low")
    cs[a] = close_loc_asym(c, hi, lo, 20) - close_loc_asym(c, hi, lo, 60)
cands["close_loc_asym_20x60"] = reindex_to_panel(cs)

# semi_vol_asym fixed: use explicit min_periods and fill zeros
for a in TRADABLES:
    c = own_series(a, "close")
    r = c.pct_change()
    dd = r.clip(upper=0).abs()
    ud = r.clip(lower=0)
    dstd = dd.rolling(60, min_periods=30).std()
    ustd = ud.rolling(60, min_periods=30).std()
    c60[a] = dstd / (ustd + 1e-12)
cands["semi_vol_asym_60_fixed"] = reindex_to_panel(c60)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(panel, h)

print("=" * 100)
print("CYCLE 33b FOCUSED - close_loc_asym family + semi_vol fix")
print("=" * 100)

results = {}
for name, sig in cands.items():
    m = validate_factor(sig, panel, library=lib, fwd_cache=fwd_cache)
    to = turnover_rank(sig, step=10)
    m["turnover_10d_rank"] = round(to, 3) if to == to else None
    results[name] = m
    ic = abs(m["ic"]); icir = abs(m["icir"])
    passed = (ic >= 0.007) and (icir >= 0.084) and (m.get("max_abs_library_correlation", 9) < 0.5) and (m.get("turnover_10d_rank") is not None and m["turnover_10d_rank"] <= 0.5)
    print("[%s] IC=%s ICIR=%s hit=%s n=%s turn_10d=%s maxlib=%s => %s"
          % (name, m["ic"], m["icir"], m.get("ic_hit_ratio"), m.get("n_ic_dates"),
             m.get("turnover_10d_rank"), m.get("max_abs_library_correlation"),
             "PASS" if passed else "FAIL"))
    if m.get("library_pairwise_corr"):
        top = sorted(m["library_pairwise_corr"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print("     top-lib-corr:", [(k, v) for k, v in top])

# Regime split for close_loc_asym_60
sig = cands["close_loc_asym_60"]
ret10 = fwd_cache["10"]
ic_ser = compute_ic(sig, ret10, 8).dropna()
years = ic_ser.index.year
regime_parts = []
for y in sorted(set(years)):
    sub = ic_ser[years == y]
    regime_parts.append("%s: ic=%.4f icir=%.4f n=%d" % (y, sub.mean(), (sub.mean() / sub.std()) if sub.std() > 0 else 0.0, len(sub)))
print("REGIME_SPLITS close_loc_asym_60:", "; ".join(regime_parts))

# Direction sanity: verify sign stability by year
print("YEAR_SIGNS:", [(y, ("NEG" if ic_ser[years == y].mean() < 0 else "POS")) for y in sorted(set(years))])

# decay table for the main candidate
print("DECAY close_loc_asym_60:")
for h in (1, 2, 3, 5, 10, 20):
    r = fwd_cache[str(h)]
    ic_h = compute_ic(sig, r, 8).dropna()
    print("   h=%2d  ic=%.4f  icir=%.4f  n=%d" % (h, ic_h.mean(), (ic_h.mean()/ic_h.std()) if ic_h.std() > 0 else 0.0, len(ic_h)))
