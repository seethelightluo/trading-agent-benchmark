"""miner_3 full validation: overnight-minus-intraday return factor.
Universe: 15 tradable cross-asset instruments.
Validates on_minus_id_20 factor: mean(overnight_ret - intraday_ret, 20d).
Overnight = open/prev_close - 1, Intraday = close/open - 1.
Hypothesis: assets with higher overnight relative to intraday (gap up / weak
session) mean-revert over multi-day horizon; this captures a gap-fade pattern.
"""
import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P = Path("../persistent"); SD = P / "stock_data"; ID = P / "index_data"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
LIBS = sorted([f.name for f in Path("factors/").glob("*.json") if f.name != "factor_ensemble.json"])
END = pd.Timestamp("2035-10-10"); np.seterr(all="ignore")

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
    """cross-sectional rank IC evaluator"""
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
    if len(ia)<12: return dict(ic=pi*np.nan,icir=np.nan,n=0,hit=np.nan,cov=0.0)
    ic=float(ia.mean()); s=float(ia.std(ddof=1))
    return dict(ic=ic, icir=float(ic/s if s>1e-10 else 0.0), n=len(ia),
                hit=float((ia>0).mean()), cov=float(np.nanmean((~np.isnan(panel)).sum(axis=1))))

def t10(panel):
    """turnover: mean abs rank change over 10d"""
    r=panel.rank(axis=1)
    v=r.diff(10).abs().mean(axis=1)
    return float(v.mean())

def sig(fname):
    """decode signal artifact from existing factors."""
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=pd.DatetimeIndex(dt),columns=rows[0][1:])

def mlc(panel):
    """max_abs_library_correlation: pairwise rank corr with all existing factor signals"""
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

# Build panel
df = build(END)
close = cols(df, "close")
open_ = cols(df, "open")

# Compute overnight and intraday returns
overnight_ret = open_ / close.shift(1) - 1.0
intraday_ret = close / open_ - 1.0

# Factor: overnight minus intraday, averaged over 20d
on_minus_id = (overnight_ret - intraday_ret).rolling(20, min_periods=12).mean()
fac = on_minus_id.replace([np.inf, -np.inf], np.nan)

print("=== OVERNIGHT MINUS INTRADAY FACTOR (on_minus_id_20) ===", flush=True)

# Horizon decay analysis
print("\n--- Decay by horizon ---")
for H in [1, 2, 3, 5, 10, 15, 20, 30]:
    fwd = close.shift(-H) / close - 1.0
    a = ev(fac, fwd)
    print(f"  H={H:2d}  IC={a['ic']:+.4f}  ICIR={a['icir']:+.4f}  n={a['n']:4d}  hit={a['hit']:.3f}  cov={a['cov']:.2f}", flush=True)

# Recency analysis (H=10)
print("\n--- Recency (H=10) ---")
fwd10 = close.shift(-10) / close - 1.0
for start in ["2022-01-01","