"""miner_3 cycle 2033-07-07. Visible through 2033-07-06. No lookahead.
Explore interest-rate sensitivity: 60d rolling beta of each asset's returns to
absolute change in US10Y yield. Cross-asset rate-cyclical factor.
Gates: abs IC>=0.0070, abs ICIR>=0.084 (10d), cross-section >=8 names.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-07-06'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(assets,end):
    C={}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=end].sort_values('date').set_index('date')
        C[a]=df['close'].astype(float)
    return C
C=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(C).dropna()
rets=close.pct_change().dropna(); idx=close.index
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}",flush=True)

u10=close['US10Y']
du10=u10.diff()
cn10=close['CN10Y']; dcn10=cn10.diff()

def beta_win(x_rets, x_r, w):
    return x_rets.rolling(w).cov(x_r).div(x_r.rolling(w).var())
us10y_beta60=beta_win(rets,du10,60)
us10y_beta20=beta_win(rets,du10,20)
cn10y_beta60=beta_win(rets,dcn10,60)

def compute_ic2(f,fwd,min_dates=30,start=None,flip=False):
    if flip: f=-f
    f=f.reindex(fwd.index)
    ii=fwd.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[]; ok=0
    for d in ii:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0: ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)
def report(name,fv,start=None,flip=False):
    a=compute_ic2(fv,fwd10,start=start,flip=flip); b=compute_ic2(fv,fwd5,start=start,flip=flip); c=compute_ic2(fv,fwd20,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} | [5]{b['IC']:.3f}[20]{c['IC']:.3f}",flush=True)
    return a

print("\n===== US10Y BETA FACTORS =====",flush=True)
report("us10y_beta_60",us10y_beta60)
report("us10y_beta_20",us10y_beta20)
report("us10y_beta_60_flip",us10y_beta60,flip=True)
report("us10y_beta_20_flip",us10y_beta20,flip=True)
report("cn10y_beta_60",cn10y_beta60)
report("cn10y_beta_60_flip",cn10y_beta60,flip=True)

print("\n=== RECENT DRIFT (2030-12-01+) ===",flush=True)
REC='2030-12-01'
report("us10y_beta_60[r]",us10y_beta60,start=REC)
report("us10y_beta_60[r].flip",us10y_beta60,start=REC,flip=True)
report("us10y_beta_20[r]",us10y_beta20,start=REC)
report("us10y_beta_20[r].flip",us10y_beta20,start=REC,flip=True)
report("cn10y_beta_60[r].flip",cn10y_beta60,start=REC,flip=True)