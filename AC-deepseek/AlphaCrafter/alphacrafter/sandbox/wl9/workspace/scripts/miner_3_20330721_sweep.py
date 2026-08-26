"""miner_3 cycle 2033-07-21. Visible through 2033-07-20. No lookahead.
Revalidate effective library + explore fresh candidates:
 A) hl_gap_20range: (high-low)/close normalized range (range persistence)
 B) hl_range_z: rolling-z of HL range vs medium-term (regime vol trigger)
 C) drawdown_40: close / rolling_max(close,40) (deep drawdown -> bounce)
 D) retrace_90: close/rolling_max_close_90 (distance from high)
E) vol_trend_20_60: short vs long vol ratio (vol ratio regime)
F) mom_60d_skip5: medium momentum
G) cross-asset beta_dxy_60 already in library (skip)
Admission gates: abs daily paper IC>=0.0070, abs ICIR>=0.084 (10d fwd).
Cross-section >=8 names.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-07-20'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(assets,end):
    C,H,L,V={},{},{},{}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=end].sort_values('date').set_index('date')
        C[a]=df['close'].astype(float); H[a]=df['high'].astype(float); L[a]=df['low'].astype(float)
        V[a]=df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan,index=df.index)
    return C,H,L,V
closes,highs,lows,vols=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(closes).dropna(); high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index)
vol=pd.DataFrame(vols).reindex(close.index)
rets=close.pct_change().dropna()
ndx=close.index
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {ndx[0]:%Y-%m-%d}..{ndx[-1]:%Y-%m-%d}",flush=True)

def compute_ic(fv,fwd,min_dates=30,start=None,flip=False):
    f=fv.reindex(fwd.index)
    if flip: f=-f
    ii=fwd.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[]; ok=0
    for d in ii:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),
                hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)
def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0); return float((s.diff()!=0).mean().mean())
def report(name,fv,start=None,flip=False):
    f=fv.reindex(fwd10.index)
    a=compute_ic(f,fwd10,start=start,flip=flip)
    b=compute_ic(f,fwd5,start=start,flip=flip); c=compute_ic(f,fwd20,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}",flush=True)
    return a

print("=== REVALIDATE EFFECTIVE LIBRARY (full) ===",flush=True)
report("mom_10d_skip5",close.shift(5)/close.shift(15)-1.0)
report("mom_120d_skip5",close.shift(5)/close.shift(125)-1.0)
report("kaufman_eff_20d",(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum()))
report("skew_20d",(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3)))
report("bb_width_20d",rets.rolling(20).std(),flip=True)
report("vol_z_20d",rets.rolling(20).std().rank(axis=1),flip=True)

print("\n=== FRESH CANDIDATES (full) ===", flush=True)
# A) HL range fraction of price (range persistence)
hlr_20=(high-low)/close.rolling(20).mean()
report("hlr_20",hlr_20,flip=True)
# B) HL range z-score vs 60d
hlr_z=(hlr_20-hlr_20.rolling(60).mean())/(hlr_20.rolling(60).std().replace(0,np.nan))
report("hlr_z_20x60",hlr_z,flip=True)
# C) drawdown from 40d high -> bounce (factor = -drawdown)
dd40=close/close.rolling(40).max()-1.0
report("retrace_40",dd40,flip=True)
retrace_90=close/close.rolling(90).max()-1.0
report("retrace_90",retrace_90,flip=True)
# E) vol trend: 10d vol / 60d vol (low ratio = quiet -> expansion?)
vt=rets.rolling(10).std().div(rets.rolling(60).std().replace(0,np.nan))
report("vol_trend_10x60",vt,flip=True)
# F) medium momentum 60d skip5
report("mom_60d_skip5",close.shift(5)/close.shift(65)-1
