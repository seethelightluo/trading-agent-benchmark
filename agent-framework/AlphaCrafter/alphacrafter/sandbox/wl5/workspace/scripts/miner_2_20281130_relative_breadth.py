import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
# Relative-strength persistence: asset's 20d return relative to contemporaneous
# cross-asset median, weighted by consistency of its daily outperformance.
m20=p/p.shift(20)-1
csmed=m20.median(axis=1)
rel=m20.sub(csmed,axis=0)
peer=r.sub(r.median(axis=1),axis=0)
cons=(peer>0).rolling(30,min_periods=20).mean()-0.5
f=rel*(1+0.8*cons)
f=f.sub(f.mean(axis=1),axis=0)
fw={h:p.shift(-h)/p-1 for h in [5,10,20]}
allics={}
for h in [5,10,20]:
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); allics[h]=a
 print('H',h,'N',len(a),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
# 10d regimes and recent windows
h=10; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw[h].loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(vals,columns=['date','ic']).set_index('date').dropna()
for name,q in [('2020-24',a.loc[:'2024-12-31']),('2025-26',a.loc['2025-01-01':'2026-12-31']),('2027-28',a.loc['2027-01-01':]),('recent252',a.tail(252))]:
 print(name,len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12),(q.ic>0).mean()))
rank=f.rank(axis=1,pct=True)
print('rows',len(p),'dates',p.index.min(),p.index.max(),'assets',len(px),'coverage',f.notna().sum(axis=1).mean()/15,'turnover',(rank-rank.shift()).abs().mean(axis=1).mean())
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_2_20281130_relative_breadth_signal.csv')
