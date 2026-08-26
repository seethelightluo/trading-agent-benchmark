#!/usr/bin/env python
"""Miner_1: validate passing candidates on recent sub-period (2023-07+), compute turnover, and
max_abs_library_correlation against existing effective factor signal panels (where loadable)."""
import pandas as pd, numpy as np, json, os, glob

CUR = "2031-01-22"
SPLIT = "2023-07-01"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA = "../persistent/stock_data/"

def load_series(assets, val="close"):
    out = {}
    for a in assets:
        df = pd.read_csv(DATA+a+".csv")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[df["date"] <= CUR]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        out[a] = df[val]
    return pd.DataFrame(out).sort_index()

px = load_series(ASSETS, "close")
hi = load_series(ASSETS, "high")
lo = load_series(ASSETS, "low")
r = px.pct_change()

def forward_ret(px_, h):
    return px_.shift(-h)/px_ - 1.0

def daily_ic(signal, fwd, min_instr=8):
    ics = []
    for dt in fwd.index:
        if dt not in signal.index: continue
        s = signal.loc[dt]; f = fwd.loc[dt]
        m = s.notna() & f.notna() & np.isfinite(s) & np.isfinite(f)
        if m.sum() >= min_instr:
            try:
                ic = np.corrcoef(s[m], f[m])[0,1]
                if np.isfinite(ic): ics.append(ic)
            except Exception: pass
    ics = np.array(ics)
    if len(ics)==0: return dict(ic=np.nan, icir=np.nan, hit=np.nan, ndates=0)
    ic=ics.mean(); std=ics.std()
    return dict(ic=ic, icir=ic/std if std>0 else 0.0, hit=(ics>0).mean(), ndates=len(ics))

def build_candidates(px_, hi_, lo_, r_):
    cand = {}
    cand["dist_ma250"] = px_/px_.rolling(250).mean()-1.0
    cand["vol_std_120"] = (px_-px_.rolling(120).mean())/px_.rolling(120).mean()
    def cap_ratio(x):
        pos=x[x>0].sum(); neg=-x[x<0].sum()
        return pos/neg if neg>0 else 0.0
    cand["capture_ratio_20"] = r_.rolling(20).apply(cap_ratio, raw=True)
    cand["rng_pos_hl_20"] = (px_-lo_.rolling(20).min())/(hi_.rolling(20).max()-lo_.rolling(20).min())
    cand["mom20_skip10"] = px_.shift(10)/px_.shift(30)-1.0
    cand["mom60_skip20"] = px_.shift(20)/px_.shift(80)-1.0
    cand["mom10_skip2"] = px_.shift(2)/px_.shift(12)-1.0
    cand["kurt_10d"] = r_.rolling(10).kurt()
    cand["ac5_20d"] = r_.rolling(20).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1] if len(x)>=8 else np.nan, raw=True)
    return cand

cand = build_candidates(px, hi, lo, r)
print("=== Recent sub-period validation (>= %s) ===" % SPLIT)
recent_results = {}
for name, sig in cand.items():
    sub = sig[sig.index >= SPLIT]
    fwd = forward_ret(px.loc[sub.index], 10)
    res = daily_ic(sub, fwd)
    recent_results[name] = res
    if np.isfinite(res['ic']):
        print(f"{name:<20} ic={res['ic']:>8.4f} icir={res['icir']:>8.3f} hit={res['hit']:>6.3f} ndates={res['ndates']}")

print("\n=== Library correlation (full-period, compute inline from factor expressions for existing set) ===")
# Existing effective factors recompute (expression-based) to measure pairwise max abs corr
vix = pd.read_csv("../persistent/index_data/VIX.csv"); vix.columns=[str(c).strip().lower() for c in vix.columns]
vix = vix[vix["date"]<=CUR]; vix["date"]=pd.to_datetime(vix["date"]); vix=vix.set_index("date").sort_index()["close"]
dxy = pd.read_csv("../persistent/index_data/DXY.csv"); dxy.columns=[str(c).strip().lower() for c in dxy.columns]
dxy = dxy[dxy["date"]<=CUR]; dxy["date"]=pd.to_datetime(dxy["date"]); dxy=dxy.set_index("date").sort_index()["close"]

library = {}
library["beta_VIX_60"] = px.apply(lambda c: (pd.concat([c.pct_change(), vix.pct_change()],axis=1,keys=['a','v']).dropna()['a'].rolling(60).cov(pd.concat([c.pct_change(), vix.pct_change()],axis=1,keys=['a','v']).dropna()['v'])/pd.concat([c.pct_change(), vix.pct_change()],axis=1,keys=['a','v']).dropna()['v'].rolling(60).var()).reindex(px.index))
# kaufman_eff_20d
library["kaufman_eff_20d"] = px.apply(lambda c: (abs(c-c.shift(20))/(c-c.shift()).abs().rolling(20).sum()))
# mom_120d_skip5
library["mom_120d_skip5"] = px.shift(5)/px.shift(125)-1.0
# bb_width_20d
library["bb_width_20d"] = 4*px.rolling(20).std()/px.rolling(20).mean()
# mom_10d_skip5
library["mom_10d_skip5"] = px.shift(5)/px.shift(15)-1.0
# vol_z_20d (volume)
VOL = {a: load_series([a],"volume")[a] for a in ASSETS}
volpdf = pd.DataFrame(VOL).reindex(px.index)
library["vol_z_20d"] = (volpdf-volpdf.rolling(20).mean())/volpdf.rolling(20).std()
# ac1_120d
library["ac1_120d"] = r.rolling(120).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1] if len(x)>=8 else np.nan, raw=True)
# skew_20d
library["skew_20d"] = r.rolling(20).skew()
# cny_beta_60
usdcny = pd.read_csv("../persistent/index_data/USDCNY.csv"); usdcny.columns=[str(c).strip().lower() for c in usdcny.columns]
usdcny = usdcny[usdcny["date"]<=CUR]; usdcny["date"]=pd.to_datetime(usdcny["date"]); usdcny=usdcny.set_index("date").sort_index()["close"]
def beta_cny(c):
    j=pd.concat([c.pct_change(), usdcny.pct_change()],axis=1,keys=['a','v']).dropna()
    return (j['a'].rolling(60).cov(j['v'])/j['v'].rolling(60).var()).reindex(px.index)
library["cny_beta_60"] = px.apply(lambda c: beta_cny(c))
# dxy_corr_change_20_60
def corr_win(c, win):
    j=pd.concat([c.pct_change(), dxy.pct_change()],axis=1,keys=['a','v']).dropna()
    return j['a'].rolling(win).corr(j['v']).reindex(px.index)
library["dxy_corr_change_20_60"] = px.apply(lambda c: corr_win