"""miner_1: 2027-10-15 factor scan batch A.
Rebuild-free: uses scripts/panel_cache.pkl (through 2027-10-14).
Scans candidate families, computes daily cross-sectional Spearman IC/ICIR/hit
across horizons 1,2,3,5,10 on full sample (2021+) and last-365d drift window.
Also revalidates the current effective library for drift context.
"""
import numpy as np
import pandas as pd
import pickle, json

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]; O = panel["open"]; H = panel["high"]; L = panel["low"]
V = panel["vol"]; M = panel["macro"]
ret = C.pct_change()

FULL0 = pd.Timestamp("2021-01-01")
gate_ic, gate_icir = 0.0070, 0.0840

def daily_ic_series(f, h):
    fwd = C.shift(-h) / C - 1.0
    ics = []
    for dt in f.index:
        ff, rr = f.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            ics.append((dt, ic))
    return pd.Series({d: v for d, v in ics})

def evaluate(f, label, horizons=(1, 2, 3, 5, 10)):
    rows = []
    for h in horizons:
        s = daily_ic_series(f, h)
        if len(s) < 30:
            continue
        s = s[s.index >= FULL0]
        ic = float(s.mean()); sd = float(s.std(ddof=1))
        icir = ic / sd if sd > 0 else 0.0
        hit = float((s > 0).mean()) if ic > 0 else float((s < 0).mean())
        cut = s.index.max() - pd.Timedelta(days=365)
        s12 = s[s.index >= cut]
        ic12 = float(s12.mean()); icir12 = float(ic12 / s12.std(ddof=1)) if len(s12) > 5 and s12.std(ddof=1) > 0 else 0.0
        cov = float(f.notna().mean().mean())
        rows.append((h, ic, icir, hit, len(s), ic12, icir12, cov))
    return rows

def summarize(name, f):
    rows = evaluate(f, name)
    if not rows:
        print(f"{name:32s} no data"); return None
    best = max(rows, key=lambda r: abs(r[2]))
    h, ic, icir, hit, n, ic12, icir12, cov = best
    ok = "PASS" if abs(ic) >= gate_ic and abs(icir) >= gate_icir else "fail"
    print(f"{name:32s} h={h:2d} IC={ic:+8.5f} ICIR={icir:+8.5f} hit={hit:5.3f} n={n:5d} IC12m={ic12:+8.5f} ICIR12m={icir12:+7.5f} cov={cov:5.3f} {ok}")
    return dict(name=name, h=h, ic=ic, icir=icir, hit=hit, n=n, ic12=ic12, icir12=icir12, cov=cov)

print("=" * 120)
print("LIBRARY DRIFT REVALIDATION (full sample 2021+, last-365d window)")
print("=" * 120)
lib = {}
lib["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
lib["mom_10d_skip5"] = C.shift(5) / C.shift(15) - 1.0
for nd in (1, 2, 3, 5):
    lib[f"rev_{nd}d"] = -(C.shift(nd) / C - 1.0)
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    lib[f"nclv_{nd}d"] = (C - lo) / (hi - lo) - 0.5
lib["id_rev_1d"] = -(C / O - 1.0)
rng = (H - L).replace(0, np.nan)
lib["nbody_1d"] = (C - L) / rng - 0.5
def vol_series(x, w):
    return x.pct_change().rolling(w).std()
lib["vol_of_vol20x60"] = vol_series(C.rolling(20).std(), 60)
vix = M["VIX"]; vix_ret = vix.pct_change()
cov60 = C.pct_change().rolling(60).cov(vix_ret); var60 = vix_ret.rolling(60).var()
beta_vix = cov60 / var60
cond = (vix > vix.rolling(20).mean()).astype(float)
lib["vix_beta_cond_60x20"] = beta_vix * cond

libres = {}
for name, f in lib.items():
    libres[name] = summarize(name, f)
json.dump({k: v for k, v in libres.items() if v}, open("scripts/miner1_20271015_lib_drift.json", "w"), indent=1, default=float)

print()
print("=" * 120)
print("CANDIDATE SCAN BATCH A (new ideas)")
print("=" * 120)
cand = {}

# A1: short-horizon momentum / continuation family (regime flip evidence)
cand["mom_5d_skip1"] = C.shift(1) / C.shift(6) - 1.0
cand["mom_10d_skip2"] = C.shift(2) / C.shift(12) - 1.0
cand["mom_20d_skip5"] = C.shift(5) / C.shift(25) - 1.0
cand["mom_60d_skip5"] = C.shift(5) / C.shift(65) - 1.0

# A2: trend efficiency (Kaufman ER): |C-C[-n]| / sum(|ret|, n)
for nd in (10, 20, 60):
    num = (C - C.shift(nd)).abs()
    den = ret.abs().rolling(nd).sum()
    cand[f"er_{nd}d"] = num / den.replace(0, np.nan)

# A3: vol-scaled momentum (risk-adjusted)
vol10 = ret.rolling(10).std()
vol20 = ret.rolling(20).std()
cand["mom_20d_skip5_vs"] = (C.shift(5) / C.shift(25) - 1.0) / vol10.replace(0, np.nan)
cand["mom_60d_skip5_vs"] = (C.shift(5) / C.shift(65) - 1.0) / vol20.replace(0, np.nan)

# A4: drawdown/recovery position: distance from rolling max
for nd in (20, 60):
    cand[f"dd_{nd}d"] = C / C.rolling(nd).max() - 1.0   # negative = below high

# A5: longer close-location-value
for nd in (10, 20):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    cand[f"nclv_{nd}d"] = (C - lo) / (hi - lo) - 0.5

# A6: vol regime: vol10/vol60
cand["vol_ratio_10x60"] = vol10 / vol20.rolling(60).mean().replace(0, np.nan)

# A7: cross-sectional relative strength (asset vs panel mean, 20d)
cand["relstr_20d"] = C.shift(5) / C.shift(25) - (C.shift(5) / C.shift(25)).mean(axis=1)

# A8: overnight gap reversal: -(open_t/close_{t-1} - 1)
cand["gap_rev_1d"] = -(O / C.shift(1) - 1.0)

# A9: upper/lower wick ratio
body = (C - O).abs()
wup = H - np.maximum(O, C)
wlo = np.minimum(O, C) - L
rng2 = (H - L).replace(0, np.nan)
cand["wick_up_1d"] = wup / rng2
cand["wick_lo_1d"] = wlo / rng2

# A10: macro-conditional reversal (VIX calm) and momentum (VIX stress)
vix_med = vix.rolling(252).median()
calm = (vix <= vix_med).astype(float)
stress = (vix > vix_med).astype(float)
cand["rev_2d_calm"] = -(C.shift(2) / C - 1.0) * calm
cand["mom_20d_skip5_stress"] = (C.shift(5) / C.shift(25) - 1.0) * stress

# A11: 10d momentum of vol (vol trend)
cand["vol_trend_10x60"] = vol10 / vol10.shift(10) - 1.0

res = {}
for name, f in cand.items():
    r = summarize(name, f)
    if r:
        res[name] = r
json.dump(res, open("scripts/miner1_20271015_scanA_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/miner1_20271015_scanA_results.json")
