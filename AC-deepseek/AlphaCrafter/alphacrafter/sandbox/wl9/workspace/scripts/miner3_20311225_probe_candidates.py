import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
P=Path("../persistent"); SD=P/"stock_data"; ID=P/"index_data"
ASSETS=["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO=["DXY","USDCNY","USDJPY","EURUSD","VIX"]
END="2031-12-24"; START="2026-07-16"
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
def vp_(df): return pd.DataFrame({a:df[f"{a}__volume"] for a in ASSETS})
def fw(cp,h): return cp.shift(-h)/cp-1.0
def ev(panel,fp,mv=8):
    ics=[]
    for t in panel.index:
        f=panel.loc[t].values.astype(float); r=fp.loc[t].values.astype(float)
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
        c=panel.index.intersection(lp.index)
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[spearmanr(a.iloc[i].values,bb.iloc[i].values)[0] for i in range(a.shape[0])]
        rs=[x for x in rs if not np.isnan(x)]
        if rs: b=max(b,abs(float(np.mean(rs))))
    return float(b)
libs=["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json","dxy_corr_change_20_60.json","kaufman_eff_20d.json","kurt_20d.json","mom_10_vixreg.json","mom_10d_skip5.json","mom_120d_skip5.json","rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json","vix_roc_20d.json","vol_z_20d.json"]
def main():
    np.seterr(all="ignore")
    df=build(); cp=pc(df); rv=rp_(df); vv=vp_(df); m={x:df[f"{x}__close"] for x in MACRO}
    f10=fw(cp,10); f5=fw(cp,5); f20=fw(cp,20)
    gap=pd.DataFrame(index=df.index,dtype=float); ovn=pd.DataFrame(index=df.index,dtype=float)
    for a in ASSETS:
        gap[a]=df[f"{a}__close"]/df[f"{a}__open"]-1
        ovn[a]=df[f"{a}__open"].pct_change()
    cand={}
    cand["intraday_mom_20d"]=gap.rolling(20).mean()
    cand["overnight_mom_20d"]=ovn.rolling(20).sum()
    dr=m["DXY"].pct_change(); rr=rv.reindex(dr.index)
    c60=rr.rolling(60).corr(dr); c20=rr.rolling(20).corr(dr)
    cand["dxy_corr_drift_20_60"]=c60.reindex(df.index)-c20.reindex(df.index)
    vi=m["VIX"]
    cand["vix_roc_60d"]=(vi/vi.shift(60)-1).reindex(df.index)
    cand["vix_roc_20d_new"]=(vi/vi.shift(20)-1).reindex(df.index)
    cand["vix_level_60d"]=vi.rolling(60).mean().reindex(df.index)
    print("n_dates",df.shape[0],"n_assets",len(ASSETS))
    print("factor                 h5_ic  h5_icir  h10_ic h10_icir h20_ic h20_icir   cov   mlc  to10")
    for name,p in cand.items():
        cov=float((~p.isna()).mean().mean()); e5=ev(p,f5); e10=ev(p,f10); e20=ev(p,f20)
        ml=mlc(p,libs); to=t10(p)
        print(f"{name:24s} {e5['ic']:7.4f} {e5['icir']:7.3f} {e10['ic']:7.4f} {e10['icir']:7.3f} {e20['ic']:7.4f} {e20['icir']:7.3f} {cov:5.2f} {ml:5.2f} {to:5.2f}")
main()