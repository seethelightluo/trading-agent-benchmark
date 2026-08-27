"""miner_2 revalidation 2035-08-30. Re-validate effective library factors + explore new candidates.
Universe: 15 cross-asset instruments, macro signals observation-only.
"""
import sys, json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from miner_2_lib import build, cols, fwd_panel, ev, turnover10, ADMIT_IC, ADMIT_ICIR, sig, pack

END = pd.Timestamp("2035-08-29")
df = build(END)
c = cols(df, "close")
ret = c.pct_change()
vix_r = df["VIX__close"].pct_change()
dxy_r = df["DXY__close"].pct_change()
usdjpy_r = df["USDJPY__close"].pct_change()
eurusd_r = df["EURUSD__close"].pct_change()

print(f"Close panel: {c.shape[0]} dates x {c.shape[1]} assets; {c.index[0].date()} to {c.index[-1].date()}")
fp = fwd_panel(c, 10)

def rollout_beta(asset_r, macro_r, win):
    out = {}
    for s in asset_r.columns:
        j = pd.concat([asset_r[s].rename("a"), macro_r.rename("m")], axis=1).dropna()
        out[s] = j["a"].rolling(win).cov(j["m"]) / j["m"].rolling(win).var()
    return pd.DataFrame(out, index=asset_r.index)

# =================== BUILD EXISTING FACTORS ===================
F = {}
F["beta_VIX_60"] = rollout_beta(ret, vix_r, 60)
F["kaufman_eff_20d"] = c.diff(20).abs() / c.diff().abs().rolling(20).sum().replace(0, np.nan)
F["mom_120d_skip5"] = c.shift(5) / c.shift(125) - 1.0
bb = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
F["bb_width_20d"] = bb
F["cny_beta_60"] = rollout_beta(ret, dxy_r, 60)
v20 = c.pct_change().rolling(20).std()
F["vol_z_20d"] = (v20 - v20.rolling(60).mean()) / v20.rolling(60).std().replace(0, np.nan)
F["ac1_120d"] = ret.rolling(120).apply(lambda x: x.autocorr(1) if len(x) >= 30 else np.nan, raw=False)
F["mom_10d_skip5"] = c.shift(5) / c.shift(15) - 1.0
F["dxy_corr_change_20_60"] = ret.rolling(20).corr(dxy_r) - ret.rolling(60).corr(dxy_r)
F["skew_20d"] = ret.rolling(20).skew()
F["kurt_20d"] = ret.rolling(20).apply(pd.DataFrame.kurtosis, raw=False)
hi60 = c.rolling(60).apply(lambda x: x.values.argmax() if len(x)==60 else np.nan, raw=True)
F["days_since_high_60"] = 60.0 - hi60 - 1.0

vroc20 = df["VIX__close"].pct_change(20)
vixroc = pd.DataFrame(index=c.index, columns=c.columns, dtype=float)
safe = ["XAU", "US10Y", "CN10Y"]
for s in c.columns:
    vixroc[s] = vroc20 if s in safe else -vroc20
F["vix_roc_20d"] = vixroc

# streak_len_14
rpos = (ret > 0).astype(float)
rneg = (ret < 0).astype(float)
F["streak_len_14"] = rpos * rpos.rolling(14).sum() - rneg * rneg.rolling(14).sum()

# Also build vix_beta_cond_60x20
vix_beta_cond = {}
for s in c.columns:
    j = pd.concat([ret[s].rename("a"), vix_r.rename("v")], axis=1).dropna()
    b60 = j["a"].rolling(60).cov(j["v"]) / j["v"].rolling(60).var()
    b20 = j["a"].rolling(20).cov(j["v"]) / j["v"].rolling(20).var()
    vix_beta_cond[s] = b60 - b20
F["vix_beta_cond_60x20"] = pd.DataFrame(vix_beta_cond, index=c.index)

# rng_pos_20d
hi_data = df[[f"{a}__high" for a in c.columns]]
lo_data = df[[f"{a}__low" for a in c.columns]]
hi_p = pd.DataFrame({a: hi_data[f"{a}__high"] for a in c.columns}, index=c.index)
lo_p = pd.DataFrame({a: lo_data[f"{a}__low"] for a in c.columns}, index=c.index)
rng_p = hi_p - lo_p
F["rng_pos_20d"] = rng_p.rolling(20).mean() / c

# =================== REVALIDATE ===================
print(f"\n{'Factor':25s} {'IC':>8s} {'ICIR':>8s} {'Hit':>5s} {'n_IC':>5s} {'Cov':>5s} {'T/O':>5s} => Gate")
results = {}
for name, panel in F.items():
    e = ev(panel, fp, mv=8, min_n=20)
    if e["n_ic_dates"] < 20:
        status = "skip-insuf"
    else:
        ic_pass = abs(e["ic"]) >= ADMIT_IC
        icir_pass = abs(e["icir"]) >= ADMIT_ICIR
        status = "PASS" if (ic_pass and icir_pass) else "FAIL"
    turo = turnover10(panel) if e["n_ic_dates"] >= 20 else 0
    print(f"{name:25s} {e['ic']:+8.5f} {e['icir']:+8.5f} {e['hit']:5.3f} {e['n_ic_dates']:5d} {e['cov_date_ge8']:5.2f} {turo:5.2f} => {status}")
    results[name] = {"ic": e["ic"], "icir": e["icir"], "hit": e["hit"], "n_ic": e["n_ic_dates"], "cov": e["cov_date_ge8"], "turnover": turo, "status": status}

# =================== NEW CANDIDATES ===================
print("\n=== NEW CANDIDATE FACTORS ===")

# 1. USDJPY_beta_60: beta vs JPY moves (risk-on/off via carry)
cands = {}
cands["usdjpy_beta_60"] = rollout_beta(ret, usdjpy_r, 60)

# 2. EURUSD_beta_60: beta vs EURUSD
cands["eurusd_beta_60"] = rollout_beta(ret, eurusd_r, 60)

# 3. Mom_reversal_5_20: (short-term reversal - medium-term momentum) / volatility
mom5 = c.shift(5)
mom20 = c.shift(5) / c.shift(25) - 1.0
cands["mom_rev_5_20"] = (mom20 - (c / c.shift(5) - 1.0)) / (v20 + 1e-9)

# 4. Cross_asset_zscore: average z-score across all 15 assets (contrarian signal)
z_all = (c - c.rolling(60).mean()) / c.rolling(60).std().replace(0, np.nan)
cands["cross_z_mean"] = z_all.mean(axis=1).to_frame("mean").join(pd.DataFrame(index=c.index, columns=c.columns)).iloc[:, :0]  # placeholder

# 5. VIX term structure proxy: diff of VIX 20d ROC and 5d ROC
vix_roc5 = df["VIX__close"].pct_change(5)
vix_roc20 = df["VIX