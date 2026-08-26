"""miner_3 cycle 2033-11-10. Visible through 2033-11-09.
(1) Re-validate currently effective library on full + recent window.
(2) Explore fresh candidate factor families (one idea per factor).
Admission gates: abs daily IC >= 0.0070 AND abs ICIR >= 0.084.
Cross-section = 15 tradable assets; min 8 assets per IC date.
"""
import json, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P = Path("../persistent"); SD = P/"stock_data"; ID = P/"index_data"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
END = "2033-11-09"; VIS = pd.Timestamp(END)

libs = ["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json",
        "dxy_corr_change_20_60.json","kaufman_eff_20d.json","kurt_20d.json","mom_10_vixreg.json","mom_10d_skip5.json",
        "mom_120d_skip5.json","rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json",
        "vix_roc_20d.json","vol_z_20d.json"]

def build():
    o={}
    for a in ASSETS:
        d=pd.read_csv(SD/(a+".csv"),parse_dates=["date"]).sort_values("date")
        o[a]=d[d["date"]<=VIS].set_index("date")[["open","high","low","close","volume"]].astype(float)
    m={}
    for x in MACRO:
        d=pd.read_csv(ID/(x+".csv"),parse_dates=["date"]).sort_values("date")
        m[x]=d[d["date"]<=VIS].set_index("date")["close"].astype(float)
    al=set()
    for d in o.values(): al.update(d.index)
    for d in m.values(): al.update(d.index)
    df=pd.DataFrame(index=pd.DatetimeIndex(sorted(al)))
    for a,d in o.items():
        for c in d.columns: df[f"{a}__{c}"]=d[c]
    for x,d in m.items(): df[f"{x}__close"]=d
    return df

def cols(df,name): return pd.DataFrame({a:df[f"{a}__{name}"] for a in ASSETS}).dropna(axis=1,how="all")
def rp_(df):
    cp=cols(df,"close"); return cp.pct_change()
def fw(cp,h): return cp.shift(-h)/cp-1.0
def ev(panel,fp,mv=8,min_n=10):
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
def decay_ic(panel,cp,hors=(1,2,3,5,10,20)):
    out={}
    for h in hors:
        fh=fw(cp,h); out[str(h)]=ev(panel,fh)["ic"]
    return out
def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=dt,columns=rows[0][1:])
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

def fmt(e): return f"IC={e['ic']:+.4f} ICIR={e['icir']:+.3f} n={e['n']:4d} hit={e['hit']:.2f} cov={e['cov']:4.2f}"

def main():
    np.seterr(all="ignore")
    df=build(); cp=cols(df,"close"); rv=rp_(df)
    op=cols(df,"open"); hh=cols(df,"high"); ll=cols(df,"low"); vo=cols(df,"volume")
    m={x:df[f"{x}__close"] for x in MACRO}
    f5=fw(cp,5); f10=fw(cp,10); f20=fw(cp,20)
    recent=pd.Timestamp("2032-11-10")
    print(f"Panel {df.shape[0]} dates x {len(ASSETS)} assets; {df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d}",flush=True)

    beta_win=lambda x,y,w: x.rolling