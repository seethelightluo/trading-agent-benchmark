import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
F={}; Y={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].loc[:cut]
 lr=np.log(d/d.shift(1)); net=lr.rolling(20,min_periods=16).sum().abs(); path=lr.abs().rolling(20,min_periods=16).sum()
 # directional efficiency: magnitude of net movement relative to total movement, signed by trend
 F[s]=np.sign(lr.rolling(20,min_periods=16).sum())*net/(path+1e-9)
 Y[s]={h:np.log(d.shift(-h)/d) for h in [1,5,10]}
f=pd.concat(F,axis=1); rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.DataFrame({'x':f.loc[dt], 'y':{s:Y[s][h].get(dt,np.nan) for s in U}}).dropna()
  if len(z)>=8: rows.append((dt,h,spearmanr(z.x,z.y).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=r[r.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1); print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),4))
rank=f.rank(axis=1,pct=True); print('coverage',round(f.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),4))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20261119_efficiency_signal.csv',index=False); print('artifact rows',len(out))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
 q=r[(r.h==1)&r.date.between(a,b)]; print('REG',a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
print('corr with 20d return',pd.concat(F,axis=1).corrwith(pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.pct_change(20) for s in U},axis=1),method='spearman').mean())
