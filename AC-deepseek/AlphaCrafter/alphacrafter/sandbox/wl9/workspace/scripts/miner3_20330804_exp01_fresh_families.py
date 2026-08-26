import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P=Path("../persistent"); SD=P/"stock_data"; ID=P/"index_data"
ASSETS=["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO=["DXY","USDCNY","USDJPY","EURUSD","VIX"]
END="2033-08-03"
libs=["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json","dxy_corr_change_20_60.json","kaufman_eff_20d.json","kurt_20d.json","mom_10_vixreg.json","mom_10d_skip5.json","mom_120d_skip5.json","rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json","vix_roc_20d.json","vol_z_20d.json"]

def build():
    o={}
    for a in ASSETS:
        f=SD/(a+".csv")
        if f.exists():
            d=pd.read_csv(f,parse_dates=["date"]).sort_values("date")
            o[a]=d[d["date"]<=END].set_index("date")[["open","high","low","close","volume"]].astype(float)
    m={}
    for mm in MACRO:
        f=ID/(mm+".csv")
        if f.exists():
            d=pd.read_csv(f,parse_dates=["date"]).sort_values("date")
            m[mm]=d[d["date"]<=END].set_index("date")["close"].astype(float)
    al=set()
    for d in o.values(): al.update(d.index)
    for d in m.values(): al.update(d.index)
    df=pd.DataFrame(index=pd.DatetimeIndex(sorted(al)))
    for a,d in o.items():
        for c in d.columns: df[a+"__"+c]=d[c]
    for mm,d in m.items(): df[mm+"__close"]=d
    return df

def cols(df,name): return pd.DataFrame({a:df[a+"__"+name] for a in ASSETS}).dropna(axis=1,how="all")
def rp_(df):
    cp=cols(df,"close"); return cp.pct_change()
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
    if len(ia)<10: return dict(ic=0,icir=0,n=len(ia),hit=0,cov=0)
    ic=float(ia.mean()); s=float(ia.std(ddof=1))
    return dict(ic=ic,icir=float(ic/s if s>1e-10 else 0),n=len(ia),
                hit=float((ia>0).mean()),cov=float(np.nanmean((~np.isnan(panel)).sum(axis=1))))
def t10(panel):
    r=panel.rank(axis=1); return float(r.diff(10).abs().mean(axis=1).mean())
def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    hd=rows[0]; dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=dt,columns=hd[1:])
def mlc(panel,libnames=None):
    libnames=libnames or libs; b=0.0
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

def main():
    np.seterr(all="ignore")
    df=build(); cp=cols(df,"close"); rv=rp_(df)
    op=cols(df,"open"); hh=cols(df,"high"); ll=cols(df,"low"); vo=cols(df,"volume")
    m={x:df[x+"__close"] for x in MACRO}
    f5=fw(cp,5); f10=fw(cp,10); f20=fw(cp,20)
    cny=m["USDCNY"]; jpy=m["USDJPY"]; eur=m["EURUSD"]
    cand={}
    up=(rv.clip(lower=0)).rolling(30).std()
    dn=(-rv.clip(upper=0)).rolling(30).std()
    cand["downup_vol_asym_30"]=dn/(up+1e-9)
    mkt=cp.mean(axis=1).pct_change().replace([np.inf,-np.inf],np.nan)
    beta=rv.rolling(60).cov(mkt)/(mkt.rolling(60).var()+1e-12)
    cand["beta_market_60_demean"]=beta.sub(beta.mean(axis=1),axis=0)
    avol=rv.rolling(20).std()
    cand["vol_share_cs_20"]=avol.div(avol.sum(axis=1),axis=0)
    rng=(hh-ll).replace(0,np.nan)
    cand["range_pos_20"]=((cp-ll)/rng).rolling(20).mean()
    cand["overnight_mom_20"]=(cp/op-1).rolling(20).mean()
    dcny=cny.pct_change()
    cand["cny_ret_corr_60"]=rv.rolling(60).corr(dcny)
    dj=jpy.pct_change(); de=eur.pct_change()
    cand["jx_ret_corr_diff_100"]=rv.rolling(100).corr(dj)-rv.rolling(100).corr(de)
    m120=cp.rolling(120).max()
    cand["dd_from_high_120"]=(cp-m120)/m120
    cand["vol_trend_20_60"]=vo.rolling(20).mean()/vo.rolling(60).mean()
    print("n_dates",df.shape[0],"n_assets",len(ASSETS),"end",END)
    print("factor                      h5_ic h5icir h10_ic h10icir h20_ic h20icir  cov   mlc  to10")
    for name,p in cand.items():
        cov=float((~p.isna()).mean().mean())
        e5=ev(p,f5)
        e10=ev(p,f10)
        e20=ev(p,f20)
        ml=mlc(p); to=t10(p)
        print(f"{name:26s} {e5['ic']:6.4f} {e5['icir']:6.3f} {e10['ic']:6.4f} {e10['icir']:6.3f} {e20['ic']:6.4f} {e20['icir']:6.3f} {cov:5.2f} {ml:5.2f} {to:5.2f}")
main()