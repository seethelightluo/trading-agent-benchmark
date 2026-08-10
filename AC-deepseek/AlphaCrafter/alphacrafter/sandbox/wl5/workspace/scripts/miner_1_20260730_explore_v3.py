"""miner_1: Explore 9 NEW cross-asset factor families (2026-07-30 cycle) - v2 with min_periods.

Universe: 15 tradable cross-asset benchmarks, visible window through 2026-07-29.
All rolling ops use min_periods so sparse per-market calendars do not NaN-poison signals.
Gate: abs(IC10) >= 0.007 and abs(ICIR10) >= 0.084 on >=8-instrument cross-sections.
"""
import sys, json, os, zlib, base64, io
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split)

VIS = "2026-07-29"
H = 10

close = closes_panel(VIS)
macro = macro_closes(VIS)
ret = close.pct_change()
vix = macro["VIX"]
us10y = close["US10Y"]
btc = close["BTC"]

def down_semidev(s, w, mp):
    return np.sqrt((s.clip(upper=0.0) ** 2).rolling(w, min_periods=mp).mean())

def up_semidev(s, w, mp):
    return np.sqrt((s.clip(lower=0.0) ** 2).rolling(w, min_periods=mp).mean())

def rolling_beta(x, y, w, mp):
    out = {}
    for a in x.columns:
        pair = pd.concat([x[a].rename("a"), y.rename("y")], axis=1).dropna()
        b = (pair["a"].rolling(w, min_periods=mp).cov(pair["y"])
             / pair["y"].rolling(w, min_periods=mp).var())
        out[a] = b
    return pd.DataFrame(out).reindex(x.index)

cands = {}

# F1 trend efficiency: |net move| / path length over 60d
cands["eff_ratio_60"] = ((close - close.shift(60)).abs()
                         / ret.abs().rolling(60, min_periods=45).sum())

# F2 downside-risk-scaled momentum (20d, skip 5)
mom20 = close.shift(5) / close.shift(25) - 1.0
cands["mom_dvol_20x60"] = mom20 / down_semidev(ret, 60, 45).clip(lower=1e-6)

# F3 vol asymmetry: down semi-dev / up semi-dev (60d)
cands["vol_asym_60"] = down_semidev(ret, 60, 45) / up_semidev(ret, 60, 45).clip(lower=1e-6)

# F4 autocorrelation shift: AR(1) last 20d minus AR(1) prior 60d window
ac20 = ret.rolling(20, min_periods=10).apply(
    lambda z: pd.Series(z).autocorr(1) if len(z) >= 10 else np.nan, raw=False)
ac60 = ret.rolling(60, min_periods=30).apply(
    lambda z: pd.Series(z).autocorr(1) if len(z) >= 30 else np.nan, raw=False)
cands["acorr_shift_20x60"] = ac20 - ac60

# F5 US10Y rate beta (yield changes)
cands["us10y_beta_60"] = rolling_beta(ret, us10y.pct_change(), 60, 40)

# F6 momentum gated by low-VIX percentile (risk-on conditioning)
vix_pct = vix.rolling(120, min_periods=90).rank(pct=True)
gate = (1.0 - vix_pct).reindex(close.index).ffill()
cands["mom_vixgate_20x120"] = mom20 * gate

# F7 vol acceleration: 20d/60d realized vol
cands["vol_accel_20x60"] = (ret.rolling(20, min_periods=15).std()
                            / ret.rolling(60, min_periods=45).std().clip(lower=1e-9))

# F8 BTC beta
cands["btc_beta_60"] = rolling_beta(ret, btc.pct_change(), 60, 40)

# F9 60d return skewness
cands["ret_skew_60"] = ret.rolling(60, min_periods=45).skew()

fr = forward_returns(close, H)
results = {}
for fid, sig in cands.items():
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        results[fid] = {"gate_pass": False, "reason": "insufficient IC dates",
                        "valid_entries": int(sig.notna().sum().sum())}
        continue
    m["regime"] = regime_split(ic)
    results[fid] = m

# max abs corr vs quarantined library artifacts (provenance only; active library is empty)
qdir = "factors/quarantine"
lib_sigs = {}
for fn in sorted(os.listdir(qdir)):
    if not fn.endswith(".json") or fn.endswith(".reason.json"):
        continue
    d = json.load(open(f"{qdir}/{fn}"))
    art = d.get("validation", {}).get("signal_artifact")
    if not art:
        continue
    try:
        dec = zlib.decompress(base64.b64decode(art["data"])).decode("utf-8")
        s = pd.read_csv(io.StringIO(dec), index_col=0)
        s.index = pd.to_datetime(s.index)
        lib_sigs[d["factor_id"]] = s.reindex(close.index)
    except Exception as e:
        print("skip lib artifact", fn, e)

for fid, m in results.items():
    if "ic" not in m:
        continue
    ics = ic_series(cands[fid], fr, min_valid=8)
    best = 0.0
    for lf, ls in lib_sigs.items():
        lic = ic_series(ls, fr, min_valid=8)
        pair = pd.concat([ics.rename("a"), lic.rename("b")], axis=1).dropna()
        if len(pair) >= 30:
            r = pair["a"].corr(pair["b"])
            if np.isfinite(r):
                best = max(best, abs(float(r)))
    m["max_abs_library_correlation"] = round(best, 4)
    m["gate_pass"] = bool(abs(m["ic"]) >= 0.007 and abs(m.get("icir") or 0) >= 0.084)

print("=" * 100)
for fid, m in sorted(results.items()):
    if "ic" not in m:
        print(f"[{fid}] -> {m}")
        continue
    print(f"[{fid}] ic10={m['ic']:.4f} icir10={m['icir']} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov_asset={m['coverage_asset_days']:.2f} cov_ge8={m['coverage_dates_ge8']:.2f} "
          f"turn={m['turnover_10d_rank']} rho_lib={m['max_abs_library_correlation']} "
          f"GATE={'PASS' if m['gate_pass'] else 'fail'}")
    print(f"    decay={m['decay_ic_by_horizon']}")
    print(f"    regimes={ {k: (v['ic'], v['icir'], v['n']) for k, v in m['regime'].items()} }")

with open("scripts/miner_1_20260730_explore_v3_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("regime",)}
               for k, v in results.items()}, f, indent=1, default=str)
print("saved results json")
