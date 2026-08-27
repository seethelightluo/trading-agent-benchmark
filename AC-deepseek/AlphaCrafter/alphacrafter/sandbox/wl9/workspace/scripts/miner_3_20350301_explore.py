import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P = Path("../persistent"); SD = P / "stock_data"; ID = P / "index_data"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
LIBS = ["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json",
        "days_since_high_60.json","dxy_corr_change_20_60.json","kaufman_eff_20d.json",
        "kurt_20d.json","mom_10_vixreg.json","mom_10d_skip5.json","mom_120d_skip5.json",
        "rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json",
        "vix_roc_20d.json","vol_z_20d.json"]
END = pd.Timestamp("2035-02-28"); np.seterr(all="ignore")

def build(end):
    o = {}
    for a in ASSETS:
        f = SD/(a+".csv")
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
            d.columns=[str(c).lower() for c in d.columns]
            av=[c for c in ["open","high","low","close","volume"] if c in d.columns]
            d = d[d["date"] <= end].set_index("date")[av].astype(float)
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

def ev(panel, fp, mv=8, min_n=12, start=None):
    ics=[]
    idx=fp.index if start is None else fp.index[fp.index>=pd.Timestamp(start)]
    for t in idx:
        if t not in fp.index or t not in panel.index: continue
        f=np.asarray(panel.loc[t],dtype=float); r=np.asarray(fp.loc[t],dtype=float)
        v=~(np.isnan(f)|np.isnan(r))
        if v.sum()>=mv:
            if np.std(f[v])>0 and np.std(r[v])>0:
                rho,_=spearmanr(f[v],r[v])
                if not np.isnan(rho): ics.append(rho)
    ia=np.array(ics)
    if len(ia)<12: return dict(ic=0.0,icir=0.0,n=0,hit=0.0,cov=0.0)
    ic=float(ia.mean()); s=float(ia.std(ddof=1))
    return dict(ic=ic,icir=float(ic/s if s>1e-10 else 0.0),n=len(ia),
                hit=float((ia>0).mean()),cov=float(np.nanmean((~np.isnan(panel)).sum(axis=1))))

def t10(panel):
    r=panel.rank(axis=1); return float(r.diff(10).abs().mean(axis=1).sum())

def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=pd.DatetimeIndex(dt),columns=rows[0][1:])

def mlc(panel):
    b=0.0
    for f in LIBS:
        lp=sig(f)
        if lp is None: continue
        c=panel.index.intersection(pd.DatetimeIndex(lp.index))
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[spearmanr(a.iloc[i].values,bb.iloc[i].values)[0] for i in range(a.shape[0])]
        rs=[x for x in rs if not np.isnan(x)]
        if rs: b=max(b,abs(float(np.mean(rs))))
    return float(b)

def report(name, panel, fwd, fl=1.0, start=None):
    a=ev(fl*panel, fwd, start=start)
    ok=(abs(a["ic"])>=0.0070 and abs(a["icir"])>=0.084)
    corr=mlc(panel)
    print(f"[{'OK' if ok else '--'}] {name:26s} IC={a['ic']:.4f} ICIR={a['icir']:.4f} n={a['n']:4d} hit={a['hit']:.3f} cov={a['cov']:.2f} tov={t10(panel):.3f} mlc={corr:.2f}", flush=True)
    return a,ok,corr

df=build(END); cp=cols(df,"close"); rv=cp.pct_change()
op=cols(df,"open"); hh=cols(df,"high"); ll=cols(df,"low"); vo=cols(df,"volume")
m={x:df[f"{x}__close"] for x in MACRO}
f10=cp.shift(-10)/cp-1.0
dvy=m["DXY"].pct_change(); vixr=m["VIX"].pct_change()
print(f"Panel {len(cp)} dates x {cp.shape[1]} assets; {cp.index[0]:%Y-%m-%d}..{cp.index[-1]:%Y-%m-%d}", flush=True)

cand={}
ds=rv.clip(upper=0)
dv=(ds**2).rolling(20).mean(); tv=(rv**2).rolling(20).mean()
cand["downside_vol_ratio_20"]=(dv/tv.replace(0,np.nan)).apply(np.sqrt)
ior=(cp-op)/(op.replace(0,np.nan))
cand["intraday_z60_5d"]=(ior.rolling(5).mean()-ior.rolling(60).mean())/ior.rolling(60).std().replace(0,np.nan)
print("=== MULTI-SAMPLE 2021+ (10d horizon) ===", flush=True)
for nm,pv in cand.items():
    a,ok,c=report(nm,pv,f10)
print("=== contra ===", flush=True)
for nm,pv in cand.items():
    report(nm+".c",pv,f10,fl=-1.0)