"""miner_3 cycle 2034-11-23. Explore fresh candidate factor families."""
import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P = Path("../persistent"); SD = P/"stock_data"; ID = P/"index_data"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
libs = ["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json",
        "kaufman_eff_20d.json","mom_10_vixreg.json","mom_10d_skip5.json","mom_120d_skip5.json",
        "rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json",
        "vix_roc_20d.json","vol_z_20d.json","dxy_corr_change_20_60.json","kurt_20d.json"]

def build():
    dd={}
    for a in ASSETS:
        d=pd.read_csv(SD/(a+".csv"),parse_dates=["date"]).sort_values("date")
        d.columns=[str(c).lower() for c in d.columns]
        dd[a]=d.set_index("date")[["open","high","low","close","volume"]].astype(float)
    mm={}
    for x in MACRO:
        d=pd.read_csv(ID/(x+".csv"),parse_dates=["date"]).sort_values("date")
        d.columns=[str(c).lower() for c in d.columns]
        mm[x]=d.set_index("date")["close"].astype(float)
    al=set()
    for d in dd.values(): al.update(d.index)
    for d in mm.values(): al.update(d.index)
    df=pd.DataFrame(index=pd.DatetimeIndex(sorted(al)))
    for a,d in dd.items():
        for c in d.columns: df[f"{a}__{c}"]=d[c]
    for x,d in mm.items(): df[f"{x}__close"]=d
    return df

def cols(df,name): return pd.DataFrame({a:df[f"{a}__{name}"] for a in ASSETS}).dropna(axis=1,how="all")
def fw(cp,h): return cp.shift(-h)/cp-1.0
def ev(panel,fp,mv=8,min_n=12):
    ics=[]
    for t in panel.index:
        if t not in fp.index: continue
        f=np.asarray(panel.loc[t],dtype=float); r=np.asarray(fp.loc[t],dtype=float)
        v=~(np.isnan(f)|np.isnan(r))
        if v.sum()>=mv:
            rho,_=spearmanr(f[v],r[v])
            if not np.isnan(rho): ics.append(rho)
    ia=np.array(ics)
    if len(ia)<min_n: return dict(ic=0.0,icir=0.0,n=0,hit=0.0,cov=0.0)
    ic=float(ia.mean()); s=float(ia.std(ddof=1))
    return dict(ic=ic,icir=float(ic/s if s>1e-10 else 0.0),n=len(ia),hit=float((ia>0).mean()),
                cov=float(np.nanmean((~np.isnan(panel)).sum(axis=1))))
def t10(panel):
    r=panel.rank(axis=1); return float(r.diff(10).abs().mean(axis=1).mean())
def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=dt,columns=rows[0][1:])
def mlc(panel):
    b=0.0
    for f in libs:
        lp=sig(f)
        if lp is None: continue
        c=panel.index.intersection(pd.DatetimeIndex(lp.index))
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[x for x in [spearmanr(a.iloc[i].values,bb.iloc[i].values)[0] for i in range(a.shape[0])] if not np.isnan(x)]
        if rs: b=max(b,abs(float(np.mean(rs))))
    return float(b)

def main():
    np.seterr(all="ignore")
    df=build(); cp=cols(df,"close"); rv=cp.pct_change()
    op=cols(df,"open"); hh=cols(df,"high"); ll=cols(df,"low"); vo=cols(df,"volume")
    m={x:df[f"{x}__close"] for x in MACRO}
    f5=fw(cp,5); f10=fw(cp,10); f20=fw(cp,20)
    ddj=m["USDJPY"].pct_change(); deur=m["EURUSD"].pct_change()
    print(f"Panel {len(cp)} dates x {cp.shape[1]} assets; {cp.index[0]:%Y-%m-%d}..{cp.index[-1]:%Y-%m-%d}",flush=True)

    cand={}
    m60=cp.rolling(60).mean(); v40=rv.rolling(40).std()
    cand["e_z_mean60_vol40"]=((cp-m60)/v40.replace(0,np.nan))
    above20=(cp>cp.rolling(20).mean()).astype(float).rolling(10).mean()
    breadth=above20.mean(axis=1)
    cand["e_breath_disp_10"]=above20.sub(breadth,axis=0)
    vol_std=rv.rolling(20).std(); vma10=vo.rolling(10).mean(); vma40=vo.rolling(40).mean()
    cand["e_volup_quiet"]=(vma10/vma40.replace(0,np.nan)).div(vol_std.replace(0,np.nan))
    gap=op.pct_change(); intr=(hh-ll)/op.replace(0,np.nan)
    cand["e_gap_over_intra_20"]=(gap.abs()/intr.replace(0,np.nan)).rolling(20).mean()
    cand["e_usdjpy_beta_x_trend"]=rv.rolling(60).cov(ddj)/(ddj.rolling(60).var().replace(0,np.nan)+1e-12)
    roll_max=cp.rolling(120).max(); refl=cp/roll_max-1
    cand["e_drawdown_recover_60"]=refl.rolling(60).mean()
    cand["e_eurusd_beta_60"]=rv.rolling(60).cov(deur)/(deur.rolling(60).var().replace(0,np.nan)+1e-12)
    rng=hh-ll
    cand["e_range_ratio_10over40"]=rng.rolling(10).mean()/rng.rolling(40).mean().replace(0,np.nan)

    print("factor                         h5_ic h5ir  h10ic h10ir h20ic h20ir  cov  mlc  to10")
    for name,p in cand.items():
        e5=ev(p,f5); e10=ev(p,f10); e20=ev(p,f20)
        cov=float((~p.isna()).mean().mean()); ml=mlc(p); to=t10(p)
        print(f"{name:30s} {e5['ic']:.4f} {e5['icir']:5.2f} {e10['ic']:.4f} {e10['icir']:5.2f} "
              f"{e20['ic']:.4f} {e20['icir']:5.2f} {cov:5.1f} {ml