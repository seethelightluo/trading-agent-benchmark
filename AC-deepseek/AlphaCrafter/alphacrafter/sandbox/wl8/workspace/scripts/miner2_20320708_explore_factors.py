"""
miner2 factor exploration - 2032-07-08
Current regime: VIX=47.49, NAV~$1.55M, fallback ensemble active.
Test: risk_adj_mom_20, skew_corrected_mom, vix_beta_signed, vol_ratio_20_60, cross_zscore
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
OBS = {"DXY","USDCNY","USDJPY","EURUSD","VIX"}
MIN_ASSETS = 8; IC_GATE = 0.0070; ICIR_GATE = 0.0840
print("MINER2 FACTOR EXPLORATION - 2032-07-08\n")

# Load data
data = {}
for a in ASSETS:
    df = get_stock_daily_data(a, days=500)
    if df is not None and len(df) > 30:
        df = df.sort_values("date")
        data[a] = {k: df[k].values.astype(float) for k in ["close","volume","open","high","low"]}
        data[a]["date"] = pd.DatetimeIndex(df["date"])
macro = {}
for n in OBS:
    df = get_index_daily_data(n, days=500)
    if df is not None and len(df) > 30:
        df = df.sort_values("date")
        macro[n] = {"close": df["close"].values.astype(float), "date": pd.DatetimeIndex(df["date"])}

# Align dates
ds = [set(data[a]["date"]) for a in ASSETS if a in data]
cd = sorted(ds[0].intersection(*ds[1:]))
cd_idx = pd.DatetimeIndex(cd)
print(f"Dates: {len(cd_idx)} ({cd_idx[0].date()} to {cd_idx[-1].date()})")

close = pd.DataFrame(index=cd_idx, columns=ASSETS, dtype=float)
for a in ASSETS:
    if a in data:
        close[a] = pd.Series(data[a]["close"], index=data[a]["date"]).reindex(cd_idx)

frozen = [a for a in ASSETS if a in close and close[a].dropna().std() < 1e-6]
print(f"Frozen: {frozen}")

# Macro
vix = pd.Series(macro["VIX"]["close"], index=macro["VIX"]["date"]).reindex(cd_idx)
dxy = pd.Series(macro["DXY"]["close"], index=macro["DXY"]["date"]).reindex(cd_idx)

def fwd_ret(close, h):
    return close.shift(-h) / close - 1.0

def rank_ic_spearman(fv, rv):
    m = fv.notna() & rv.notna()
    if m.sum() < MIN_ASSETS: return np.nan
    r,_ = spearmanr(fv[m], rv[m])
    return r

def ic_series(fp, fr):
    return pd.Series([rank_ic_spearman(fp.loc[d], fr.loc[d]) for d in fp.index
                      if rank_ic_spearman(fp.loc[d], fr.loc[d]) is not None],
                     index=[d for d in fp.index if rank_ic_spearman(fp.loc[d], fr.loc[d]) is not None])

def ic_series_h(fp, fr):
    ics = []
    for dt in fp.index:
        ic = rank_ic_spearman(fp.loc[dt], fr.loc[dt])
        if not np.isnan(ic): ics.append(ic)
    return pd.Series(ics, index=fp.index[:len(ics)])

def compute_panel(fn, macro_sers=None, **kw):
    fp = pd.DataFrame(np.nan, index=cd_idx, columns=ASSETS)
    for a in ASSETS:
        if a not in close.columns: continue
        if a in frozen: fp[a] = 0.0; continue
        try:
            v = fn(close[a], macro_sers, **kw)
            if isinstance(v, pd.Series): fp[a] = v.reindex(close[a].index)
        except: pass
    return fp

def validate(fp, close, name, horizons=(1,2,3,5,10,20), adm=10):
    print(f"\n{'='*50}\nVALIDATE: {name}\n{'='*50}")
    cov = fp.notna().sum().sum() / fp.size
    ge8 = (fp.notna().sum(axis=1) >= MIN_ASSETS).mean()
    to = fp.rank(axis=1).diff(10).abs().mean(axis=1).dropna().mean()
    print(f"Coverage: {cov:.4f}, ge8: {ge8:.4f}, turnover: {to:.4f}")
    dec = {}
    for h in horizons:
        fr = fwd_ret(close, h)
        ic = ic_series_h(fp, fr)
        dec[h] = float(ic.mean()) if len(ic)>0 else np.nan
        print(f"  h={h:2d}: IC={dec[h]:+.6f} ({len(ic)}d)")
    ms = ic_series_h(fp, fwd_ret(close, adm))
    icv = float(ms.mean()) if len(ms)>0 else np.nan
    icirv = float(ms.mean()/ms.std()) if len(ms)>3 else np.nan
    hit = float((ms>0).mean()) if icv>0 else float((ms<0).mean())
    print(f"\nAdm h={adm}: IC={icv:+.6f}(gate>{IC_GATE}) ICIR={icirv:+.6f}(gate>{ICIR_GATE}) hit={hit:.4f}")
    p = abs(icv)>=IC_GATE and abs(icirv)>=ICIR_GATE
    print(f"PASSED: {p}")
    return {"id":name,"ic":icv,"icir":icirv,"hit":hit,"n":len(ms),"cov":cov,"ge8":ge8,"to":to,"dec":dec,"pass":p}

# ---- F1: risk_adj_mom_20 ----
def f1(c, m, **kw):
    r = c.pct_change(20)
    v = c.pct_change().rolling(20).std()
    return r/(v+1e-8)
fp1 = compute_panel(f1)
r1 = validate(fp1, close, "risk_adj_mom_20")

# ---- F2: skew_corrected_mom ----
def f2(c, m, **kw):
    r = c.pct_change(20)
    sk = c.pct_change().rolling(20).skew()
    return r * (1 - sk.abs())
fp2 = compute_panel(f2)
r2 = validate(fp2, close, "skew_corrected_mom")

# ---- F3: vix_beta_signed ----
def f3(c, m, **kw):
    if m is None: return pd.Series(0.0, index=c.index)
    v = m.reindex(c.index)
    cv = pd.concat([c.rename("c"), v.rename("v")], axis=1).dropna()
    b = cv["c"].pct_change().rolling(60).cov(cv["v"].pct_change()) / cv["v"].pct_change().rolling(60).var()
    s20 = v.pct_change(20).fillna(0).map(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    return (b * s20).reindex(c.index)
fp3 = compute_panel(f3, macro_sers=vix)
r3 = validate(fp3, close, "vix_beta_signed")

# ---- F4: vol_ratio_20_60 ----
def f4(c, m, **kw):
    v20 = c.pct_change().rolling(20).std()
    v60 = c.pct_change().rolling(60).std()
    return v20/(v60+1e-8)
fp4 = compute_panel(f4)
r4 = validate(fp4, close, "vol_ratio_20_60")

# ---- F5: cross_zscore_10d ----
def f5(c, m, **kw):
    r = c.pct_change(