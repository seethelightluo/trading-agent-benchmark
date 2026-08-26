import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r5=p.pct_change(5); disp=r5.std(axis=1); gate=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
f=(-r5.mul(gate,axis=0)).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=ds).dropna(); print(f'H{h} IC {q.mean():.8f} ICIR {q.mean()/q.std(ddof=1):.8f} hit {(q>0).mean():.4f} dates {len(q)} avgN {np.mean(ns):.2f}')
 if h==10:
  for n in [180,500,750]:
   z=q.iloc[-n:]; print(f'recent{n} H10 IC {z.mean():.8f} ICIR {z.mean()/z.std(ddof=1):.8f} hit {(z>0).mean():.4f} dates {len(z)}')
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns))
print('coverage %.6f active %.6f turnover %.6f'%(f.notna().mean().mean(),(gate>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340626_dispersion_gated_short_reversal_signal.csv',index=False)
print('artifact scripts/miner_2_20340626_dispersion_gated_short_reversal_signal.csv')
print('max_abs_library_correlation null')
