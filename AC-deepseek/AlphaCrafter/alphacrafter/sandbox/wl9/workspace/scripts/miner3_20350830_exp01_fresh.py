"""miner_3 cycle 2035-08-30. Explore fresh candidate factor families (cross-asset)."""
import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P = Path("../persistent"); SD = P/"stock_data"; ID = P/"index_data"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
LIBS = ["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json",
        "dxy_corr_change_20_60.json","kaufman_eff_20d.json","mom_10_vixreg.json","mom_10d_skip5.json",
        "mom_120d_skip5.json","rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json",
        "vix_roc_20d.json","vol_z_20d.json","kurt_20d.json"]
END = pd.Timestamp("2035-08-29")

def build():
    dd={}
    for a in ASSETS:
        f=SD/(a+".csv")
        if not f.exists(): continue
        d=pd.read_csv(f,parse_dates=["date"]).sort_values("date")
        d.columns=[str(c).lower() for c in d.columns]
        d=d[d["date"]<=END].set_index("date")[["open","high","low","close","volume"]].astype(float)
        dd[a]=d
    mm={}
    for x in MACRO:
        d=pd.read_csv(ID/(x+".csv"),parse_dates=["date"]).sort_values("date")
        d.columns=[str(c).lower() for c in d.columns]
        mm[x]=d[d["date"]<=END].set_index("date")["close"].astype(float)
    al=set()
    for d in dd.values(): al.update(d.index)
    for d in mm.values(): al.update(d.index)
    df=pd.DataFrame(index=pd.DatetimeIndex(sorted(al)))
    for a,d in dd.items():
        for c in d.columns: df[f"{a}__{c}"]=d[c]
    for x,d in mm.items(): df[f"{x}__close"]=d
    return df

def cols(df,name): return pd.DataFrame({a:df[f"{a}__{name}"] for a in ASSETS if f"{a}__{name}" in df.columns}).dropna(axis=1,how="all")
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
    try: d=json.load(open("factors/"+fname))
    except Exception: return None
    sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=dt,columns=rows[0][1:])
def mlc(panel):
    b=0.0
    for f in LIBS:
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
    ddj=m["USDJPY"].pct_change(); deur=m["EURUSD"].pct_change(); ddxy=m["DXY"].pct_change()
    print(f"Panel {len(cp)} dates x {cp.shape[1]} assets; {cp.index[0]:%Y-%m-%d}..{cp.index[-1]:%Y-%m-%d}",flush=True)

    cand={}
    # 1. Downside deviation ratio: downside vol vs total vol over 40d
    dd_neg=rv.where(rv<0,0.0)
    down_risk=dd_neg.rolling(40).std(); tot_risk=rv.rolling(40).std()
    cand["e_downside_ratio_40"]=down_risk/tot_risk.replace(0,np.nan)
    # 2. Up-move capture: mean up day return / mean abs of all returns (20d)
    up=(rv.where(rv>0,0.0).rolling(20).mean())/rv.abs().rolling(20).mean().replace(0,np.nan)
    cand["e_up_capture_20"]=up
    # 3. Volume dryness / illiquidity (high vol & low volume regime)
    vo_z=(vo-vo.rolling(40).mean())/vo.rolling(40).std().replace(0,np.nan)
    vol20=rv.rolling(20).std()
    cand["e_quiet_highvol_20"]=(-vo_z).div(vol20.replace(0,np.nan)+1e-6)
    # 4. USDCNY beta (China stress) - CN pricing pressure
    dcny=m["USDCNY"].pct_change()
    cand