import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-11-19')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in U}
D={a:d[d.index<=end] for a,d in D.items()}
# Volatility-normalized overnight gap reversal: use prior completed-day volatility only.
F={}; C={}
for a,d in D.items():
 gap=d.open/d.close.shift(1)-1
 vol=d.close.pct_change().rolling(20,min_periods=15).std()
 F[a]=(-gap/vol).replace([np.inf,-np.inf],np.nan)
 C[a]=d.close
fac=pd.concat(F,axis=1).sort_index(); cl=pd.concat(C,axis=1).reindex(fac.index)
for h in [1,5,10]:
 fwd=cl.pct_change(h).shift(-h); ics=[];ns=[];dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z));dates.append(dt)
 s=pd.Series(ics,index=dates); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   z=s[(s.index.year>=lo)&(s.index.year<=hi)]; print('REG',lo,hi,len(z),round(z.mean(),6))
print('coverage',round(fac.notna().mean().mean(),4),'dates',fac.index.nunique())
print('turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
fac.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}).to_csv('scripts/miner_2_20261119_volnorm_gap_signal.csv',index=False)
