"""miner_2 2034-08-25: revalidate existing library factors on fresh panel (through 2034-08-24)."""
import json, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
GATE_IC = 0.0070
GATE_ICIR = 0.0840
MIN_VALID = 8

p = pickle = pd.read_pickle("scripts/panel_cache_20340825.pkl")
close, open_, high, low, vol, ret, macro = p["close"], p["open"], p["high"], p["low"], p["vol"], p["ret"], p["macro"]
close = close[ASSETS]; open_ = open_[ASSETS]; high = high[ASSETS]; low = low[ASSETS]
macro = macro.reindex(close.index).ffill()

def rolling_min(s, w): return s.rolling(w, min_periods=w).min()
def rolling_max(s, w): return s.rolling(w, min_periods=w).max()

signals = {}
signals["miner2_20260715_id_rev_1d"] = -(close/open_ - 1.0)
signals["miner2_20260715_nbody_1d"] = -(close - open_) / (high - low).replace(0, np.nan)
for nd in [1,2,3,5]:
    num = close - rolling_min(low, nd)
    den = rolling_max(high, nd) - rolling_min(low, nd)
    signals[f"miner2_20260715_nclv_{nd}d"] = -(num / den.replace(0, np.nan))
lnc = np.log(close)
for nd in [1,2,3,5]:
    signals[f"miner2_20260715_rev_{nd}d"] = -(lnc - lnc.shift(nd))
daily_ret = close.pct_change()
signals["miner2_20260715_rev_1d_vs"] = -(lnc - lnc.shift(1)) / daily_ret.rolling(20, min_periods=20).std().replace(0, np.nan)
signals["mom_120d_skip5"] = close.shift(5) / close.shift(125) - 1.0
signals["vol_of_vol20x60"] = daily_ret.rolling(20, min_periods=20).std().rolling(60, min_periods=60).std()
vix_ret = macro["VIX"].pct_change()
beta = daily_ret.rolling(60, min_periods=60).cov(vix_ret) / vix_ret.rolling(60, min_periods=60).var()
vix_20d_chg = macro["VIX"] / macro["VIX"].shift(20) - 1.0
signals["vix_beta_cond_60x20"] = -beta * vix_20d_chg

def daily_ic(sig, fwd):
    dates = sig.index.intersection(fwd.index)
    out = {}
    for dt in dates:
        s = sig.loc[dt]; f = fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() >= MIN_VALID:
            out[dt] = spearmanr(s[m], f[m]).statistic
    return pd.Series(out).sort_index()

def metrics(ic, sig):
    ic = ic.dropna()
    if len(ic) < 20:
        return None
    return {"ic": float(ic.mean()), "icir": float(ic.mean()/ic.std()) if ic.std() > 0 else 0.0,
            "hit": float((ic > 0).mean()), "n": int(len(ic))}

print("=== LIBRARY FACTOR REVALIDATION (panel through 2034-08-24) ===")
fwd = {h: close.shift(-h)/close - 1.0 for h in [1,2,3,5,10]}
rows = []
for fid, sig in signals.items():
    row = {"factor_id": fid}
    for h in [1,2,3,5,10]:
        ic = daily_ic(sig, fwd[h])
        for lab, win in [("full", None), ("2y", 504), ("1y", 252), ("6m", 126)]:
            sub = ic if win is None else ic.tail(win)
            m = metrics(sub, sig)
            if m:
                row[f"ic{h}_{lab}"] = m["ic"]; row[f"icir{h}_{lab}"] = m["icir"]
    rows.append(row)
res = pd.DataFrame(rows).set_index("factor_id")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60)
cols = []
for h in [1,2,3,5,10]:
    cols += [f"ic{h}_full", f"icir{h}_full", f"ic{h}_6m", f"icir{h}_6m"]
print(res[cols].round(4).to_string())
res.to_csv("scripts/miner2_reval_20340825_lib.csv")

# gate check on 1y/6m windows for admission-relevant horizons
print("\n=== GATE CHECK (|IC|>=0.0070 & |ICIR|>=0.0840) ===")
for fid, row in res.iterrows():
    for lab in ["1y", "6m"]:
        for h in [1,2,3,5]:
            ic = row.get(f"ic{h}_{lab}"); icir = row.get(f"icir{h}_{lab}")
            if pd.notna(ic) and pd.notna(icir) and abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR:
                print(f"{fid} {lab} h={h}: IC={ic:.4f} ICIR={icir:.3f} PASS")
