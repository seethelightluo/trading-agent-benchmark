"""Screener cycle 2034-03-10: recompute library factor signals on the current panel,
validate against stored artifacts where possible, compute recent IC/ICIR, regime metrics."""
import json, pickle, glob, os, base64, zlib, io, csv
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

CURRENT = "2034-03-09"
ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]

p = pickle.load(open("scripts/panel_cache_20340310.pkl","rb"))
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
# vix_beta_cond_60x20
vix_ret = macro["VIX"].pct_change()
asset_ret = close.pct_change()
beta = asset_ret.rolling(60, min_periods=60).cov(vix_ret) / vix_ret.rolling(60, min_periods=60).var()
vix_20d_chg = macro["VIX"] / macro["VIX"].shift(20) - 1.0
signals["vix_beta_cond_60x20"] = -beta * vix_20d_chg

# ---- validate against stored artifacts ----
def load_artifact(path):
    d = json.load(open(path))
    sa = d.get("validation", {}).get("signal_artifact") or d.get("signal_artifact")
    if sa is None: return None
    raw = base64.b64decode(sa["data_b64"]); txt = zlib.decompress(raw).decode()
    rows = list(csv.reader(io.StringIO(txt))); cols = rows[0]
    sig = pd.DataFrame(rows[1:], columns=cols)
    sig["date"] = pd.to_datetime(sig["date"]); sig = sig.set_index("date")
    for c in ASSETS:
        sig[c] = pd.to_numeric(sig[c], errors="coerce")
    return sig[ASSETS][~sig.index.duplicated(keep="last")]

print("=== artifact validation (spearman corr on overlap) ===")
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f or "evicted" in f: continue
    d = json.load(open(f)); fid = d.get("factor_id")
    art = load_artifact(f)
    if art is None: continue
    rec = signals[fid]
    common = art.index.intersection(rec.index)
    if len(common) < 100: 
        print(fid, "overlap too small", len(common)); continue
    cors = []
    for a in ASSETS:
        x = art.loc[common, a]; y = rec.loc[common, a]
        m = x.notna() & y.notna()
        if m.sum() > 50:
            cors.append(spearmanr(x[m], y[m]).statistic)
    print(f"{fid}: overlap={len(common)} median_spearman={np.median(cors):.4f}")

# ---- forward returns ----
fwd1 = close.shift(-1)/close - 1.0
fwd5 = close.shift(-5)/close - 1.0
fwd10 = close.shift(-10)/close - 1.0

def daily_ic(sig, fwd):
    dates = sig.index.intersection(fwd.index)
    out = {}
    for dt in dates:
        s = sig.loc[dt]; f = fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() >= 8:
            out[dt] = spearmanr(s[m], f[m]).statistic
    return pd.Series(out).sort_index()

print("\n=== recent IC/ICIR (through", CURRENT, ") ===")
rows = []
for fid, sig in signals.items():
    row = {"factor_id": fid}
    for h, fwd in [("ic1", fwd1), ("ic5", fwd5), ("ic10", fwd10)]:
        ic = daily_ic(sig, fwd)
        for lab, win in [("20", 20), ("60", 60)]:
            sub = ic.tail(win)
            if len(sub) == 0:
                row[f"{h}_{lab}"] = np.nan; row[f"{h}ir_{lab}"] = np.nan
            else:
                row[f"{h}_{lab}"] = sub.mean()
                sd = sub.std(ddof=1)
                row[f"{h}ir_{lab}"] = sub.mean()/(sd+1e-12) if sd > 0 else 0.0
        sub120 = ic.tail(120)
        row[f"{h}_120"] = sub120.mean() if len(sub120) else np.nan
    rows.append(row)
res = pd.DataFrame(rows).set_index("factor_id")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(res.round(4).to_string())
res.to_csv("scripts/screener_ic_20340310.csv")

# ---- regime metrics ----
print("\n=== regime metrics (as of", CURRENT, ") ===")
eqw = close.mean(axis=1)
r20 = (eqw.iloc[-1]/eqw.iloc[-21]-1); r60 = (eqw.iloc[-1]/eqw.iloc[-61]-1); r120 = (eqw.iloc[-1]/eqw.iloc[-121]-1)
print(f"eqw 20d {r20*100:.2f}%  60d {r60*100:.2f}%  120d {r120*100:.2f}%")
ma20 = close.rolling(20).mean().iloc[-1]; ma60 = close.rolling(60).mean().iloc[-1]; last = close.iloc[-1]
print("above MA20:", int((last>ma20).sum()), "/15   above MA60:", int((last>ma60).sum()), "/15")
d20 = daily_ret.tail(20)
print("20d mean daily ret (eqw):", f"{(d20.mean(axis=1).mean()*100):.4f}%")
disp = d20.std(axis=1)
print("20d avg cross-sectional dispersion:", f"{disp.mean()*100:.3f}%  last:", f"{disp.iloc[-1]*100:.3f}%")
annvol = daily_ret.tail(20).std()*np.sqrt(252)
print("20d ann vol by asset:", {a: f"{v*100:.1f}%" for a,v in annvol.sort_values(ascending=False).items()})
vix = macro["VIX"].iloc[-1]
print(f"VIX last {vix:.1f}  10d ago {macro['VIX'].iloc[-11]:.1f}  60d ago {macro['VIX'].iloc[-61]:.1f}  120d ago {macro['VIX'].iloc[-121]:.1f}")
print("\n20d asset returns:")
print((close.iloc[-1]/close.iloc[-21]-1).sort_values(ascending=False).round(4).to_string())
print("\n60d asset returns:")
print((close.iloc[-1]/close.iloc[-61]-1).sort_values(ascending=False).round(4).to_string())
