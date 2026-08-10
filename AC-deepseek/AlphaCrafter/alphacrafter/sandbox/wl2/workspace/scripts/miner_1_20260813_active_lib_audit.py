"""miner_1 (2026-08-13) audit of the CURRENT ACTIVE factor library and re-validation
of my 3 gate-passing candidates (ma_accel_20_60, bollinger_pos_20, ret_skew_10) against
the ACTIVE library only (ensemble 8 + any other EFFECTIVE+admitted JSONs).

Why: the earlier screen's max_lib_corr used ALL factors/*.signal.npy (including evicted
range_pos_120d etc.). The post-Miner gate evicts on conflict with ACTIVE admitted factors.
This script recomputes pairwise rho vs the active library from real signal artifacts.
"""
import json, glob, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner_3_20260813_lib import (ASSETS, GRID, load_asset, asset_series, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix, summarize,
                                  decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
                                  coverage_stats, safe_div)

# ---------- 1. Build ACTIVE library: EFFECTIVE + admitted JSONs in factors/ root ----------
def scan_active_library():
    active = {}
    for f in sorted(glob.glob("factors/*.json")):
        base = os.path.basename(f)
        if base.endswith(".bak") or base == "factor_ensemble.json":
            continue
        # skip timestamp-suffixed duplicates if a plain version exists
        if ".2026" in base and os.path.exists("factors/" + base.split(".2026")[0] + ".json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        v = d.get("validation", {})
        if v.get("status") != "EFFECTIVE":
            continue
        adm = d.get("benchmark_admission", {})
        if not adm.get("admitted_at") and "admitted" not in str(adm).lower():
            # some admitted files may lack the field; keep if status EFFECTIVE + artifact
            pass
        fid = d.get("factor_id", base.replace(".json", ""))
        active[fid] = (f, d)
    return active

def load_artifact_signal(d):
    """Return (n_dates, 15) signal matrix from artifact spec (npy or embedded panel)."""
    sa = d.get("signal_artifact", None)
    if isinstance(sa, dict) and sa.get("format") == "daily_panel":
        cols = sa.get("columns")
        vals = np.array(sa.get("values"), dtype=float)  # (T,15)
        return vals, cols
    if isinstance(sa, str) and os.path.exists(sa):
        return np.load(sa, allow_pickle=True), None
    if isinstance(sa, str):
        p = "factors/" + sa
        if os.path.exists(p):
            return np.load(p, allow_pickle=True), None
    # fallback: artifact_provenance path
    prov = d.get("artifact_provenance", {})
    if isinstance(prov, dict):
        for k in ("path", "artifact_path", "file"):
            if k in prov and os.path.exists(prov[k]):
                return np.load(prov[k], allow_pickle=True), None
    return None, None

active = scan_active_library()
print("ACTIVE library scan: %d EFFECTIVE JSONs in root" % len(active))
for fid, (f, d) in sorted(active.items()):
    adm = d.get("benchmark_admission", {})
    print("  %-28s admitted_at=%s" % (fid, adm.get("admitted_at", "?")))

# ---------- 2. Load active signals ----------
active_signals = {}
missing = []
for fid, (f, d) in active.items():
    sig, cols = load_artifact_signal(d)
    if sig is None or sig.shape[0] < 100:
        missing.append(fid)
        continue
    active_signals[fid] = sig
print("Active signals loaded: %d; missing: %s" % (len(active_signals), missing))

# ---------- 3. Recompute my 3 candidates ----------
series = asset_series()
fwd_by_h = fwd_by_horizon_dict(series)

def f_ma_accel(df, fs=20, fl=60, minp=10, accel=20):
    ma_f = df['close'].rolling(fs, min_periods=minp).mean()
    ma_l = df['close'].rolling(fl, min_periods=minp).mean()
    level = ma_f / ma_l - 1.0
    return level - level.shift(accel)

def f_bollinger(df, w=20, minp=10, k=2.0):
    ma = df['close'].rolling(w, min_periods=minp).mean()
    sd = df['close'].rolling(w, min_periods=minp).std()
    return (df['close'] - ma) / (k * sd)

def f_ret_skew(df, w=10, minp=5):
    return df['ret'].rolling(w, min_periods=minp).skew()

builders = {
    'ma_accel_20_60': (f_ma_accel, {}),
    'bollinger_pos_20': (f_bollinger, {}),
    'ret_skew_10': (f_ret_skew, {}),
}

def mean_daily_rho(a, b, min_assets=8):
    """Mean daily cross-sectional Spearman rho between two (T,15) raw signal matrices."""
    ar = cross_sectional_rank(a)
    br = cross_sectional_rank(b)
    rows = min(ar.shape[0], br.shape[0])
    rhos = []
    for t in range(rows):
        x, y = ar[t], br[t]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < min_assets:
            continue
        xs = pd.Series(x[ok]); ys = pd.Series(y[ok])
        c = xs.rank().corr(ys.rank())
        if np.isfinite(c):
            rhos.append(c)
    return float(np.mean(rhos)) if rhos else 0.0

print("\n=== CANDIDATE RE-VALIDATION vs ACTIVE LIBRARY ===")
cand_results = {}
for name, (fn, kw) in builders.items():
    d = {}
    for s, df in series.items():
        try:
            d[s] = fn(df, **kw)
        except Exception:
            d[s] = pd.Series(np.nan, index=df.index)
    mat = to_grid(d)
    ics = spearman_ic_matrix(mat, fwd_by_h[10])
    summ = summarize(ics, np.array(GRID), name, 10)
    cov_ad, cov_d8 = coverage_stats(mat)
    rank = cross_sectional_rank(mat)
    turn = turnover_10d_rank(rank)
    dec = decay_curve(mat, fwd_by_h)
    # pairwise vs ACTIVE library
    pw = {}
    for fid, sig in active_signals.items():
        pw[fid] = round(mean_daily_rho(mat, sig), 4)
    mx_name = max(pw, key=lambda k: abs(pw[k])) if pw else None
    mx_abs = abs(pw[mx_name]) if mx_name else 0.0
    # pairwise vs ALL root artifacts (provenance)
    pw_all = {}
    for f in sorted(glob.glob("factors/*.signal.npy")):
        arr = np.load(f, allow_pickle=True)
        fid = os.path.basename(f).replace(".signal.npy", "")
        pw_all[fid] = round(mean_daily_rho(mat, arr), 4)
    mx_all_name = max(pw_all, key=lambda k: abs(pw_all[k])) if pw_all else None
    mx_all = abs(pw_all[mx_all_name]) if mx_all_name else 0.0

    ic, icir = summ['ic'], summ['icir']
    passed = abs(ic) >= 0.0070 and abs(icir) >= 0.0840
    conflict = mx_abs >= 0.5
    print("\n[%s] IC=%.4f ICIR=%.3f hit=%.3f n=%d cov=%.3f/%.3f turn=%.3f PASS=%s" %
          (name, ic, icir, summ['hit'], summ['n_ic_dates'], cov_ad, cov_d8, turn, passed))
    print("  active-lib pairwise:", json.dumps(pw))
    print("  max_ACTIVE_corr=%s (%.3f) | max_ALL_artifacts_corr=%s (%.3f) | rho-conflict-vs-active=%s"
          % (mx_name, mx_abs, mx_all_name, mx_all, conflict))
    print("  regime:", json.dumps(summ['regime']))
    print("  decay:", json.dumps(dec))
    cand_results[name] = {
        'ic': round(ic, 4), 'icir': round(icir, 3), 'hit': round(summ['hit'], 3),
        'n': summ['n_ic_dates'], 'cov_ad': round(cov_ad, 3), 'cov_d8': round(cov_d8, 3),
        'turnover': round(turn, 3), 'pass': passed,
        'max_active_corr': round(mx_abs, 4), 'max_active_name': mx_name,
        'max_all_corr': round(mx_all, 4), 'max_all_name': mx_all_name,
        'active_pairwise': pw, 'regime': summ['regime'], 'decay': dec,
        'conflict_vs_active': conflict,
    }

with open('scripts/miner_1_20260813_activelib_results.json', 'w') as f:
    json.dump(cand_results, f, indent=1)
print("\nSaved scripts/miner_1_20260813_activelib_results.json")
