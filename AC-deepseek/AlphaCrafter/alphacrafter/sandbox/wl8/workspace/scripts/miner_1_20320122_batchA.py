"""miner_1 research cycle 2032-01-22.
Explore candidate per-asset cross-asset factors on the 15-asset universe.
Validation through 2032-01-21 (visible through previous completed day). No lookahead.
Admission gates: |IC|>=0.0070, |ICIR|>=0.0840, horizon=10d.
Prints # dates used and instruments used.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
END = pd.Timestamp("2032-01-21")
START = pd.Timestamp("2020-01-02")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
HORIZON = 10
MIN_ASSETS = 8

def load_panel():
    closes, vols, opens, highs, lows = {},{},{},{},{}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        closes[a]=df["close"].astype(float); vols[a]=df["volume"].astype(float)
        opens[a]=df["open"].astype(float); highs[a]=df["high"].astype(float); lows[a]=df["low"].astype(float)
    return (pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(opens), pd.DataFrame(highs), pd.DataFrame(lows))

def load_macro():
    out={}
    for k in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
        df=pd.read_csv(f"{INDEX_DIR}/{k}.csv", parse_dates=["date"])
        df=df[df["date"]<=END].set_index("date").sort_index()
        df=df[~df.index.duplicated(keep="last")]
        out[k]=df["close"].astype(float)
    return out

close, vol, open_, high, low = load_panel()
macro = load_macro()
print(f"panel dates: {len(close)}  assets: {len(ASSETS)}  last_date: {close.index[-1].date()}")

def dense_per_asset():
    d={}
    for a in ASSETS:
        idx=close[a].dropna().index
        d[a]={"close":close[a].reindex(idx),"vol":vol[a].reindex(idx),"open":open_[a].reindex(idx),"high":high[a].reindex(idx),"low":low[a].reindex(idx)}
    return d
dense=dense_per_asset()

def factor_panel(fn, **params):
    out={}
    for a in ASSETS:
        dc=dense[a]
        try:
            s=fn(dc, **params)
            out[a]=pd.Series(np.asarray(s).ravel(), index=dc["close"].index).reindex(close.index)
        except Exception as e:
            out[a]=pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)

def eval_factor(name, panel, print_res=True):
    fwd = close.pct_change(HORIZON).shift(-HORIZON)
    ics=[]; dates=[]
    for dt in panel.index[(panel.index>=START)&(panel.index<=END)]:
        fv=panel.loc[dt]; rv=fwd.loc[dt]
        m=fv.notna()&rv.notna()
        if m.sum()<MIN_ASSETS: continue
        ic,_=spearmanr(fv[m], rv[m])
        if np.isfinite(ic): ics.append(ic); dates.append(dt)
    arr=np.array(ics)
    mu=float(arr.mean()); sd=float(arr.std(ddof=1)) if len(arr)>1 else 0.0
    icir=mu/sd if sd>0 else 0.0
    hit=float((arr>0).mean()) if mu>=0 else float((arr<0).mean())
    # coverage and turnover
    total=int(panel.notna().sum().sum()); cells=int(panel.size)
    cov_ad=total/cells if cells else 0.0
    sub=panel.dropna(how="all"); rows=sub.iloc[::10].rank(axis=1)
    chg=[]; prev=None
    for _,r in rows.iterrows():
        r=r.dropna()
        if prev is not None:
            both=prev.index.intersection(r.index)
            if len(both)>=MIN_ASSETS: chg.append(float((r[both]-prev[both]).abs().mean()))
        prev=r
    to