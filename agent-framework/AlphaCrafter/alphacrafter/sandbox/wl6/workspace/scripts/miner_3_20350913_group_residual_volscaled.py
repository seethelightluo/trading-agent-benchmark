import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; G={s:('eq' if s in U[:8] else 'cm' if s in U[8:11] else 'cr' if s in U[11:13] else 'rt') for s in U}
def f(s):
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0:x=get_index_daily_data(s,5000)
 return x.set_index('date')['close'] if x is not None else None
C=pd.DataFrame({s:f(s) for s in U}).sort_index().ffill(); C=C.loc[C.index<=pd.Timestamp('2035-09-12')]; r=np.log(C/C.shift(1)); ret=C/C.shift(20)-1
med=pd.DataFrame({g:ret[[s for s in U if G[s]==g]].median(axis=1) for g in set(G.values())}); resid=pd.DataFrame({s:ret[s]-med[G[s]] for s in U}); vol=r.rolling(30).std(); sig=-resid/vol
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20350913_group_residual_volscaled_signal.csv',index=False)
for h in [5,10,20,40]:
 y=C.shift(-h)/C-1; a=[];ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:a.append(sig.loc[d,ok].corr(y.loc[d,ok],method='spearman'));ns.append(ok.sum())
 a=pd.Series(a).dropna();print(f'h={h} dates={len(a)} avg_inst={np.mean(ns):.3f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.8f} hit={(a>0).mean():.4f}')
print('coverage',sig.notna().sum().sum()/(len(sig)*15),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean(),'dates',len(sig))
