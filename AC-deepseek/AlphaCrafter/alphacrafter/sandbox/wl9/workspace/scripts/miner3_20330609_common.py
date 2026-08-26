import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
P=Path("../persistent"); SD=P/"stock_data"; ID=P/"index_data"
ASSETS=["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO=["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load(end):
    o={}
    for a in ASSETS:
        f=SD/(a+".csv")
        if f.exists():
            d=pd.read_csv(f,parse_dates=["date"]).sort_values("date")
            o[a]=d[d["date"]<=end].set_index("date")[["open","high","low","close","volume"]].astype(float)
    return o
def macro(end):
    o={}
    for mm in MACRO:
        f=ID/(mm+".csv")
        if f.exists():
            d=pd.read_csv(f,parse_dates=["date"]).sort_values("date")
            o[mm]=d[d["date"]<=end].set_index("date")["close"].astype(float)
    return o
def build(end):
    o=load(end); m=macro(end)
    al=set()
    for d in o.values(): al.update(d.index)
    for d in m.values(): al.update(d.index)
    ds=sorted(al)
    df=pd.DataFrame(index=pd.DatetimeIndex(ds))
    for a,d in o.items():
        for c in d.columns: df[f"{a}__{c}"]=d[c]
    for mm,d in m.items(): df[f"{mm}__close"]=d
    return df
def pc(df): return pd.DataFrame({a:df[f"{a}__close"] for a in ASSETS})
def rp_(df): return pc(df).pct_change()
def fw(cp,h): return cp.shift(-h)/cp-1.0
def ev(panel,fp,mv=8):
    ics=[]
    for t in panel.index:
        if t not in fp.index: continue
        f=np.asarray(panel.loc[t],dtype=float); r=np.asarray(fp.loc[t],dtype=float)
        v=~(np.isnan(f)|np.isnan(r))
        if v.sum()>=mv:
            rho,_=spearmanr(f[v],r[v])
            if not np.isnan(rho): ics.append(rho)
    ia=np.array(ics)
    if len(ia)<10: return dict(ic=0,icir=0,n=len(ia))
    ic=ia.mean(); s=ia.std(ddof=1)
    it=ic/s if s>1e-10 else 0.0
    return dict(ic=float(ic),icir=float(it),n=len(ia),hit=float((ia>0).mean()),cov=float(np.nanmean((~np.isnan(panel)).sum(axis=1))))
def t10(panel):
    r=panel.rank(axis=1); return float(r.diff(10).abs().mean(axis=1).mean())
libs=["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json","dxy_corr_change_20_60.json","kaufman_eff_20d.json","kurt_20d.json","mom_10_vixreg.json","mom_10d_skip5.json","mom_120d_skip5.json","rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json","vix_roc_20d.json","vol_z_20d.json"]
def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    hd=rows[0]; dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=dt,columns=hd[1:])
def mlc(panel,libnames=None):
    libnames = libnames or libs
    b=0.0
    for f in libnames:
        lp=sig(f)
        if lp is None: continue
        c=panel.index.intersection(pd.DatetimeIndex(lp.index))
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[spearmanr(a.iloc[i].values,bb.iloc[i].values)[0] for i in range(a.shape[0])]
        rs=[x for x in rs if not np.isnan(x)]
        if rs: b=max(b,abs(float(np.mean(rs))))
    return float(b)