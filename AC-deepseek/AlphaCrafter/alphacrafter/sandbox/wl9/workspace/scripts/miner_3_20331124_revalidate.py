"""miner_3 cycle 2033-11-24. Visible through 2033-11-23. No lookahead.
Revalidate effective/ensemble library on recent regime + full sample.
Admission gates: abs daily paper IC>=0.0070, abs ICIR>=0.084 (10d fwd).
Cross-section >=8 names of the 15-instrument universe.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-11-23'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(assets,end):
    C,H,L={},{},{}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=end].sort_values('date').set_index('date')
        C[a]=df['close'].astype(float); H[a]=df['high'].astype(float); L[a]=df['low'].astype(float)
    return C,H,L
closes,highs,lows=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(closes).dropna(); high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index)
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
def report(name,fv,start=None,flip=True):
    a=compute_ic(fv,fwd10,start=start,flip=flip)
    b=compute_ic(fv,fwd5,start=start,flip=flip); c=compute_ic(fv,fwd20,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}",flush=True)

RUN_START='2032-01-01'  # recent regime ~2 yrs
print("=== ENSEMBLE/LIBRARY REVALIDATE (recent 2032+) ===",flush=True)
report("mom_10d_skip5",close.shift(5)/close.shift(15)-1.0,start=RUN_START,flip=False)
report("mom_120d_skip5",close.shift(5)/close.shift(125)-1.0,start=RUN_START,flip=False)
report("kaufman_eff_20d",(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum()),start=RUN_START,flip=False)
report("skew_20d",(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3)),start=RUN_START,flip=False)
report("bb_width_20d",rets.rolling(20).std(),start=RUN_START,flip=True)
report("vol_z_20d",rets.rolling(20).std().rank(axis=1),start=RUN_START,flip=True)
ac1 = close.rolling(120).apply(lambda x: x.autocorr(1) if len(x.dropna())>=30 else np.nan, raw=False)
report("ac1_120d",ac1,start=RUN_START,flip=True)
report("rng_pos_20d",rets.rolling(20).apply(lambda x:(x>0).mean(),raw=True),start=RUN_START,flip=True)
report("kurt_20d",rets.rolling(20).kurt(),start=RUN_START,flip=True)