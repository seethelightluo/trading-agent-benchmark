"""miner_3 cycle 2033-10-13. Visible through 2033-10-12. No lookahead.
Revalidate effective library + explore fresh candidates.
Admission gates: abs daily paper IC>=0.0070, abs ICIR>=0.084 (10d fwd).
Cross-section >=8 names of the 15-instrument universe.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-10-12'
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
def report(name,fv,start=None,flip=False):
    f=fv.reindex(fwd10.index)
    a=compute_ic(f,fwd10,start=start,flip=flip)
    b=compute_ic(f,fwd5,start=start,flip=flip); c=compute_ic(f,fwd20,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}",flush=True)
    return a

RUN_START='2031-06-15'  # recent regime
print("=== REVALIDATE EFFECTIVE LIBRARY (recent regime) ===",flush=True)
report("mom_10d_skip5",close.shift(5)/close.shift(15)-1.0,start=RUN_START)
report("mom_120d_skip5",close.shift(5)/close.shift(125)-1.0,start=RUN_START)
report("kaufman_eff_20d",(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum()),start=RUN_START)
report("skew_20d",(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3)),start=RUN_START)
report("bb_width_20d",rets.rolling(20).std(),start=RUN_START,flip=True)
report("vol_z_20d",rets.rolling(20).std().rank(axis=1),start=RUN_START,flip=True)
ac1 = close.rolling(120).apply(lambda x: x.autocorr(1) if len(x.dropna())>=30 else np.nan, raw=False)
report("ac1_120d",ac1,start=RUN_START,flip=True)
report("rng_pos_20d",rets.rolling(20).apply(lambda x:(x>0).mean(),raw=True),start=RUN_START,flip=True)
report("kurt_20d",rets.rolling(20).kurt(),start=RUN_START,flip=True)


print("\n=== FRESH CANDIDATES (recent regime) ===", flush=True)
dd120=close/close.rolling(120).max()-1.0
report("retrace_120",dd120,start=RUN_START,flip=True)
dd60=close/close.rolling(60).max()-1.0
report("retrace_60",dd60,start=RUN_START,flip=True)
delta=close.diff()
up=delta.clip(lower=0); dn=(-delta).clip(lower=0)
rs=up.rolling(14).mean().div(dn.rolling(14).mean().replace(0,np.nan))
rsi=100-100/(1+rs)
report("rsi_14",rsi,start=RUN_START,flip=True)
m10=close.shift(5)/close.shift(15)-1.0
m60=close.shift(5)/close.shift(65)-1.0
report("mom_accel_10x60",m10-m60,start=RUN_START)
report("hlr_20",(high-low)/close.rolling(20).mean(),start=RUN_START,flip=True)
# long-term reversal
mom240=close.shift(5)/close.shift(245)-1.0
report("mom_240d_skip5",mom240,start=RUN_START,flip=True)
# volatility-scaled momentum
report("mom10_over_vol20",m10.div(rets.rolling(20).std().replace(0,np.nan)),start=RUN_START,flip=False)
# 52wk residual: price relative to 120d median
report("price_vs_med120",close/close.rolling(120).median()-1.0,start=RUN_START,flip=True)