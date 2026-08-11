import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change()
m=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cut').set_index('date')['close'].reindex(r.index).ffill().pct_change()
for win in [20,40,60]:
 out=[]
 for i in range(win,len(r)-10):
  x=r.iloc[i-win:i]; z=m.iloc[i-win:i]; var=z.var()
  if z.notna().sum()<win*.7 or var<1e-12: continue
  beta=x.apply(lambda q:q.cov(z)/var); f=-beta
  y=r.iloc[i+1]; q=pd.concat([f,y],axis=1).dropna()
  if len(q)>=8: out.append((r.index[i],spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
 s=pd.Series(dict(out)); print(win,'dates',len(s),'avgN',r.loc[s.index].notna().sum(axis=1).mean(),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=s[(s.index.year>=lo)&(s.index.year<=hi)]; print(lo, q.mean(),q.mean()/q.std(),len(q))
