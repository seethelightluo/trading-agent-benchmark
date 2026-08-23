import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    x=get_stock_daily_data(s,days=5000)
    if x is None or len(x)==0: x=get_index_daily_data(s,days=5000)
    if x is None or len(x)==0: return None
    return x[['date','close']].drop_duplicates('date').set_index('date')['close']
px={s:fetch(s) for s in U}; px={s:x for s,x in px.items() if x is not None}
C=pd.DataFrame(px).sort_index().ffill(); C=C.loc[C.index<=pd.Timestamp('2035-09-26')]
r20=C/C.shift(20)-1; r60=C/C.shift(60)-1
# Continuous market breadth regime: reversal is stronger when the cross-asset trend is broadly positive,
# and suppressed during broad selloffs where short-term reversal is less reliable.
breadth=(r60>0).mean(axis=1)
mult=(0.5+1.5*breadth).clip(0.5,2.0)
sig=-r20.mul(mult,axis=0)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20350927_breadth_weighted_reversal_signal.csv',index=False)
for h in [5,10,20,40]:
    fwd=C.shift(-h)/C-1; vals=[]; ns=[]
    for d in sig.index:
        ok=sig.loc[d].notna()&fwd.loc[d].notna()
        if ok.sum()>=8:
            z=sig.loc[d,ok].corr(fwd.loc[d,ok],method='spearman')
            if pd.notna(z): vals.append(z); ns.append(ok.sum())
    a=pd.Series(vals); print(f'h={h} dates={len(a)} avg_inst={np.mean(ns):.3f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.8f} hit={(a>0).mean():.4f}')
rank=sig.rank(axis=1,pct=True); print(f'coverage={sig.notna().sum().sum()/(len(sig)*len(U)):.6f} turnover={rank.diff().abs().mean().mean():.6f} instruments={len(U)} dates={len(sig)} end={C.index.max().date()}')
