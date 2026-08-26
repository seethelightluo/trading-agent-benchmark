import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
P=Path("../persistent"); SD=P/"stock_data"; ID=P/"index_data"
ASSETS=["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO=["DXY","USDCNY","USDJPY","EURUSD","VIX"]
END="2032-10-28"; START="2026-07-16"
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
        f=np.asarray(panel.loc[t],dtype=float); r=np.asarray(fp.loc[t],dtype=float)
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
def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    hd=rows[0]; dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=dt,columns=hd[1:])
def mlc(panel,libs):
    b=0.0
    for f in libs:
        lp=sig(f)
        if lp is None: continue
        c=panel.index.intersection(pd.DatetimeIndex(lp.index))
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[spearmanr(a.iloc[i].values,bb.iloc[i].values)[0] for i in range(a.shape[0])]
        rs=[x for x in rs if not np.isnan(x)]
        if rs: b=max(b,abs(float(np.mean(rs))))
    return float(b)
libs=["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json","dxy_corr_change_20_60.json","kaufman_eff_20d.json","kurt_20d.json","mom_10_vixreg.json","mom_10d_skip5.json","mom_120d_skip5.json","rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json","vix_roc_20d.json","vol_z_20d.json"]
def main():
    np.seterr(all="ignore")
    df=build(); cp=pc(df); rv=rp_(df); vv=pd.DataFrame({a:df[f"{a}__volume"] for a in ASSETS})
    m={x:df[f"{x}__close"] for x in MACRO}
    f5=fw(cp,5); f10=fw(cp,10); f20=fw(cp,20)
    vi=m["VIX"]; dx=m["DXY"]; cny=m["USDCNY"]; jpy=m["USDJPY"]; eur=m["EURUSD"]
    cand={}
    # 1) upside/downside capture skew: ratio of mean positive-day return to mean negative-day return over 60d
    m30=rv.rolling(30).mean()
    up=(rv.clip(lower=0)+1e-9).rolling(30).mean()
    dn=(-rv.clip(upper=0)+1e-9).rolling(30).mean()
    cand["updown_capture_30"]=up/(dn+1e-9)
    # 2) rolling beta to cross-sectional equal-weight market (60d), demeaned
    mkt=cp.mean(axis=1).pct_change()
    mkt=mkt.replace([np.inf,-np.inf],np.nan)
    beta=rv.rolling(60).cov(mkt)/(mkt.rolling(60).var()+1e-12)
    cand["beta_market_60_demean"]=beta.sub(beta.mean(axis=1),axis=0)
    # 3) idiosyncratic vol (residual): vol minus component correlated with market
    mv=mkt.rolling(20).std()
    ivol=rv.rolling(20).std()-beta.rolling(20).std().mul(mv,axis=0).fillna(0)*0+rv.rolling(20).std()-mv.values[:,None]*0
    # simpler: ratio of asset vol to market vol
    cand["vol_ratio_mkt_20"]=rv.rolling(20).std().div(mv,axis=0)
    # 4) autocorr of daily returns over 5d (trend persistence)
    ac5=rv.rolling(5).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1] if len(x)>3 and np.std(x[:-1])>0 and np.std(x[1:])>0 els