import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
P=Path("../persistent"); SD=P/"stock_data"; ID=P/"index_data"
ASSETS=["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO=["DXY","USDCNY","USDJPY","EURUSD","VIX"]
END="2032-08-19"; START="2026-07-16"
def load():
    o={}
    for a in ASSETS:
        f=SD/(a+".csv")
        if f.exists():
            d=pd.read_csv(f,parse_dates=["date"]).sort_values("date")
            o[a]=d[d["date"]<=END].set_index("date")[["open","high","low","close","volume"]].astype(float)
    return o
def macro():
    o={}
    for mm in MACRO:
        f=ID/(mm+".csv")
        if f.exists():
            d=pd.read_csv(f,parse_dates=["date"]).sort_values("date")
            o[mm]=d[d["date"]<=END].set_index("date")["close"].astype(float)
    return o
def build():
    o=load(); m=macro()
    al=set()
    for d in o.values(): al.update(d.index)
    for d in m.values(): al.update(d.index)
    ds=sorted(x for x in al if START<=x.strftime("%Y-%m-%d")<=END and x.weekday()<5)
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
        fv=panel.loc[t]; rv=fp.loc[t]
        if np.isscalar(fv): continue
        f=fv.values.astype(float); r=rv.values if hasattr(rv,'values') else rv
        r=np.asarray(r,dtype=float)
        v=~(np.isnan(f)|np.isnan(r))
        if v.sum()>=mv:
            rho,_=spearmanr(f[v],r[v])
            if not np.isnan(rho): ics.append(rho)
    ia=np.array(ics)
    if len(ia)<10: return dict(ic=0,icir=0,n=len(ia))
    ic=ia.mean(); s=ia.std(ddof=1)
    return dict(ic=float(ic),icir=float(ic/s if s>1e-10 else 0),n=len(ia))
def t10(panel):
    r=panel.rank(axis=1); return float(r.diff(10).abs().mean(axis=1).mean())
def main():
    np.seterr(all="ignore")
    df=build(); cp=pc(df); rv=rp_(df)
    m={x:df[f"{x}__close"] for x in MACRO}
    f5=fw(cp,5); f10=fw(cp,10); f20=fw(cp,20)
    vi=m["VIX"]; dx=m["DXY"]
    cand={}
    # Distance from 60d mean normalized by vol (z-distance)
    ma60=cp.rolling(60).mean(); sd20=cp.pct_change().rolling(20).std()
    cand["dist_ma60_z"]=((cp/ma60-1)/sd20)
    # RSI14
    delta=cp.diff(); up=delta.clip(lower=0).rolling(14).mean(); dn=(-delta.clip(upper=0)).rolling(14).mean()
    rs=up/(dn+1e-12); cand["rsi_14"]=100-100/(1+rs)
    # downside vol ratio (semi-deviation 20d)
    r=rv
    dneg=r.clip(upper=0); dpos=r.clip(lower=0)
    cand["downside_vol_ratio"]=r.rolling(20).std()/(dneg.rolling(20).std()+1e-9)
    # price acceleration: momentum of momentum (10d mom minus 20d mom normalized)
    cand["price_accel"]= (cp/cp.shift(10)-1) - (cp/cp.shift(20)/cp.shift(10)-1).fillna(0) if False else ((cp/cp.shift(10)-1) - (cp/cp.shift(20)-1))
    # volume-price correlation 20d
    vv=pd.DataFrame({a:df[f"{a}__volume"] for a in ASSETS})
    cand["vol_price_corr_20"]=vv.rolling(20).corr(rv).reindex(rv.index)
    # cross-sectional: asset relative vs cross-sectional mean momentum (market-relative 20d)
    m20=cp/cp.shift(20)-1
    cs_mean=m20.mean(axis=1)
    cand["cs_rel_mom_20"]=m20.sub(cs_mean,axis=0)
    # kurtosis of returns 20d (already have skew; try kurt normalized)
    cand["kurt_20d_new"]=r.rolling(20).kurt()
    # max drawdown based risk: 60d max drawdown magnitude
    roll_max=cp.rolling(60).max(); cand["dd_60"]=(cp/roll_max-1)
    print("n_dates",df.shape[0],"n_assets",len(ASSETS))
    print("factor                     h5_ic  h5_icir  h10_ic h10_icir h20_ic h20_icir   cov  to10")
    for name,p in cand.items():
        cov=float((~p.isna()).mean().mean()); e5=ev(p,f5); e10=ev(p,f10); e20=ev(p,f20)
        to=t10(p)
        print(f"{name:26s} {e5['ic']:7.4f} {e5['icir']:7.3f} {e10['ic']:7.4f} {e10['icir']:7.3f} {e20['ic']:7.4f} {e20['icir']:7.3f} {cov:5.2f} {to:5.2f}")
main()
