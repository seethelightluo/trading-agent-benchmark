import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0: x=get_index_daily_data(s,5000)
 return None if x is None or len(x)==0 else x[['date','close']].drop_duplicates('date').set_index('date')['close']
p={s:load(s) for s in U}; p={s:x for s,x in p.items() if x is not None}
C=pd.DataFrame(p).sort_index().ffill().loc[:pd.Timestamp('2035-10-10')]
r=C.pct_change(); vol=r.rolling(60).std(); r20=C/C.shift(20)-1
# Interpretable regime-conditioned cross-sectional signal: volatility-scaled reversal
# in elevated-dispersion regimes, and zero-neutral signal in compressed regimes.
cs=r20.sub(r20.median(axis=1),axis=0); base=-cs/(vol+1e-8)
disp=r20.std(axis=1); threshold=disp.rolling(120,min_periods=60).median()
sig=base.where(disp>threshold,0.0).shift(1)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20351011_dispersion_gated_reversal_signal.csv',index=False)
for h in [5,10,20,40,60]:
 fw=C.shift(-h)/C-1; vals=[]; ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&fw.loc[d].notna()&(sig.loc[d]!=0)
  if ok.sum()>=8:
   q=sig.loc[d,ok].corr(fw.loc[d,ok],method='spearman')
   if pd.notna(q): vals.append(q); ns.append(ok.sum())
 a=pd.Series(vals); print(f'h={h} dates={len(a)} avg_inst={np.mean(ns):.3f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.8f} hit={(a>0).mean():.4f}')
print(f'coverage={sig.notna().sum().sum()/(len(sig)*len(U)):.6f} active_cross_sections={(sig.abs().sum(axis=1)>0).sum()} turnover={sig.rank(axis=1,pct=True).diff().abs().mean().mean():.6f} instruments={len(U)} dates={len(sig)} end={C.index.max().date()}')
