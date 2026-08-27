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
END = pd.Timestamp("2035-09-12"); np.seterr(all="ignore")

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

def ev(panel, fp, mv=8, min_n=30, start=None):
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
    if len(ia)<min_n: return dict(ic=0.0,icir=0.0,n=0,hit=0.0,cov=0.0)
    ic=float(ia.mean()); s=float(ia.std(ddof=1))
    cov=float(np.nanmean(np.clip((~np.isnan(panel)).sum(axis=1).astype(float),0,15)))
    return dict(ic=ic,icir=float(ic/s if s>1e-10 else 0.0),n=len(ia),
                hit=float((ia>0).mean()),cov=cov)

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
    out={}
    for f in LIBS:
        lp=sig(f)
        if lp is None: continue
        c=panel.index.intersection(pd.DatetimeIndex(lp.index))
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[spearmanr(a.iloc[i].values,bb.iloc[i].values)[0] for i in range(a.shape[0])]
        rs=[x for x in rs if not np.isnan(x)]
        if rs: out[f]=float(np.mean(rs))
    return out

df=build(END); cp=cols(df,"close"); hi=cols(df,"high"); lo=cols(df,"low"); op=cols(df,"open")
rv=cp.pct_change()
fr10=pd.DataFrame(cp.shift(-10)/cp-1.0,index=cp.index,columns=cp.columns)
fr5=pd.DataFrame(cp.shift(-5)/cp-1.0,index=cp.index,columns=cp.columns)
fr20=pd.DataFrame(cp.shift(-20)/cp-1.0,index=cp.index,columns=cp.columns)

# ============ FACTOR A: up/down capture asymmetry (log avg-up / avg-down magnitude, 20d) ============
updays=rv.clip(lower=0); dndays=rv.clip(upper=0)
ud=(updays.rolling(20).mean()/ (dndays.abs().rolling(20).mean()+1e-12))
facA=ud.apply(np.log1p).replace([np.inf,-np.inf],np.nan)
print("=== FACTOR A: log up/down capture ratio (20d) ===", flush=True)
for h,fr in [(5,fr5),(10,fr10),(20,fr20)]:
    a=ev(facA, fr); print(f"  h={h:2d} IC={a['ic']:.4f} ICIR={a['icir']:.4f} n={a['n']} hit={a['hit']:.3f} cov={a['cov']:.1f}", flush=True)
for s in ["2023-01-01","2025-01-01","2027-01-01","2029-01-01","2031-01-01","2033-01-01","2035-01-01"]:
    a=ev(facA, fr10, start=s); print(f"  {s} 10d IC={a['ic']:.4f} ICIR={a['icir']:.4f} n={a['n']}", flush=True)
print(f"  turn10_rank={t10(facA):.3f}", flush=True)

# ============ FACTOR B: avg overnight gap (20d) ============
gap=(op/cp.shift(1)-1.0)
facB=gap.rolling(20).mean()
print("=== FACTOR B: avg 20d overnight gap ===", flush=True)
for h,fd in [(5,fr5),(10,fr10),(20,fr20)]:
    a=ev(facB, fd); print(f"  h={h:2d} IC={a['ic']:.4f} ICIR={a['icir']:.4f} n={a['n']} hit={a['hit']:.3f}", flush=True)

# ============ FACTOR C: body position (close location within day range), 20d avg ============
bodypos=(cp-lo)/((hi-lo)+1e-12)
facC=bodypos.rolling(20).mean()
print("=== FACTOR C: mean body position (20d) ==="
# ============ FACTOR C: body position (close location within day range), 20d avg ============
bodypos=(cp-lo)/((hi-lo)+1e-12)
facC=bodypos.rolling(20).mean()
print("=== FACTOR C: mean body position (20d) ===", flush=True)
for h,fd in [(5,fr5),(10,fr10),(20,fr20)]:
    a=ev(facC, fd); print(f"  h={h:2d} IC={a["ic"]:.4f} ICIR={a["icir"]:.4f} n={a["n"]} hit={a["hit"]:.3f}", flush=True)

# ============ FACTOR D: Price-volume trend (close * volume vs vol-weighted price) ============
vwap=(hi+lo+cp)/3.0
pvt=((cp-vwap)/vwap)*vol.rolling(20).mean().div(vol.rolling(20).mean()+1)
# simpler: volume-weighted price deviation
facD=((cp-vwap)/vwap).rolling(20).mean()
print("=== FACTOR D: price vs VWAP deviation (20d) ===", flush=True)
for h,fd in [(5,fr5),(10,fr10),(20,fr20)]:
    a=ev(facD, fd); print(f"  h={h:2d} IC={a["ic"]:.4f} ICIR={a["icir"]:.4f} n={a["n"]} hit={a["hit"]:.3f}", flush=True)

# library correlation only for promising factors
print("=== Factors computed. Run library correlation on best candidates ===", flush=True)

