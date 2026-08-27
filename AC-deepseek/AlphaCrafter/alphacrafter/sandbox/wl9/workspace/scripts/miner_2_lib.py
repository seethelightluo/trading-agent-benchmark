"""Shared factor-mining helper for miner_2.
Reusable data loading, factor panel building, IC/ICIR/turnover/coverage/decay
evaluation, and read-back of persisted library signal artifacts.
Universe: 15 intentionally tradable cross-asset instruments (all valid).
"""
import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P = Path("../persistent"); SD = P / "stock_data"; ID = P / "index_data"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
LIBS = [p.name for p in Path("factors").glob("*.json")
        if "_deprecated" not in p.name and p.name != "factor_ensemble.json"]

ADMIT_IC = 0.0070
ADMIT_ICIR = 0.0840

def build(end=None):
    """Wide DataFrame: asset OHLCV + macro closes. End exclusive bound."""
    o = {}
    for a in ASSETS:
        f = SD/(a+".csv")
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
            d.columns=[str(c).lower() for c in d.columns]
            av=[c for c in ["open","high","low","close","volume"] if c in d.columns]
            d = d[d["date"]<=end].set_index("date")[av].astype(float)
            o[a]=d
    m={}
    for x in MACRO:
        d = pd.read_csv(ID/(x+".csv"), parse_dates=["date"]).sort_values("date")
        d.columns=[str(c).lower() for c in d.columns]
        m[x]=d[d["date"]<=end].set_index("date")["close"].astype(float)
    al=set()
    for d in o.values(): al.update(d.index)
    for d in m.values(): al.update(d.index)
    df=pd.DataFrame(index=pd.DatetimeIndex(sorted(al)))
    for a,d in o.items():
        for c in d.columns: df[f"{a}__{c}"]=d[c]
    for x,d in m.items(): df[f"{x}__close"]=d
    return df

def cols(df,name): return pd.DataFrame({a:df[f"{a}__{name}"] for a in ASSETS}).dropna(axis=1,how="all")

def fwd_ret(df, h=10):
    """h-day forward return per asset."""
    c = cols(df,"close")
    fr = c.shift(-h) / c - 1.0
    return fr

def ev(panel, fp, mv=8, min_n=12, start=None, end=None):
    """Spearman IC of factor panel vs forward-return panel; daily IC series."""
    ics=[]; ns=[]
    if start: panel=panel[panel.index>=pd.Timestamp(start)]
    if end: panel=panel[panel.index<=pd.Timestamp(end)]
    for t in (panel.index):
        if t not in fp.index: continue
        f=np.asarray(panel.loc[t],dtype=float); r=np.asarray(fp.loc[t],dtype=float)
        v=~(np.isnan(f)|np.isnan(r))
        if v.sum()>=mv:
            if np.std(f[v])>0 and np.std(r[v])>0:
                rho,_=spearmanr(f[v],r[v])
                if not np.isnan(rho): ics.append(rho); ns.append(int(v.sum()))
    ia=np.array(ics)
    if len(ia)<min_n:
        return dict(ic=0.0,icir=0.0,n=0,hit=0.0,cov_panel=0.0,n_assets=0,n_dates=0)
    ic=float(ia.mean()); s=float(ia.std(ddof=1))
    cov=float(np.nanmean((~np.isnan(panel)).sum(axis=1)))
    cov_date=float(np.nanmean((~np.isnan(panel)).sum(axis=1)>=mv))
    return dict(ic=ic, icir=float(ic/s if s>1e-10 else 0.0), n=len(ia),
                hit=float((ia>0).mean()), cov_panel=cov, cov_date_ge8=cov_date,
                n_assets=int(panel.shape[1]), n_dates=int(panel.shape[0]),
                n_ic_dates=len(ia))

def turnover10(panel):
    r = panel.rank(axis=1)
    return float(r.diff(10).abs().mean(axis=1).sum(axis=0))

def decay(panel, fp10, maxh=20):
    out={}
    mx=panel.shape[0]
    for h in [1,2,3,5,10,20]:
        if h>mx: continue
        fr=fwd_ret(fp10, h)
        # fp10 is already a forward-return panel built from same base; reuse base below
        e=ev(panel, fr, mv=8, min_n=12)
        out[h]=round(e["ic"],4)
    return out

def fwd_panel(close_panel, h):
    return close_panel.shift(-h)/close_panel - 1.0

def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=pd.DatetimeIndex(dt),columns=rows[0][1:])

def mlc(panel):
    """max_abs_library_correlation: mean across dates of |spearman| vs other effective lib factors."""
    mx=0.0; best=None
    for f in LIBS:
        lp=sig(json.load(open("factors/"+f)).get("validation",{}).get("signal_artifact",{}))
        if lp is None: continue
        c=panel.index.intersection(pd.DatetimeIndex(lp.index))
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[]
        for i in range(a.shape[0]):
            x=a.iloc[i].values; y=bb.iloc[i].values
            v=~(np.isnan(x)|np.isnan(y))
            if len(np.unique(x[v]))>1 and len(np.unique(y[v]))>1:
                rs.append(spearmanr(x[v],y[v])[0])
        if rs and np.nanmean(np.abs(rs))>mx:
            mx=float(np.nanmean(np.abs(rs))); best=f
    return mx, best

def pack(panel):
    """Build base64:zlib:csv signal artifact for a factor panel."""
    import hashlib
    df=panel.copy(); df.insert(0,"date",panel.index.strftime("%Y-%m-%d"))
    raw=df.to_csv(index=False).encode()
    b=base64.b64encode(zlib.compress(raw)).decode()
    return b, hashlib.sha256(raw).hexdigest()