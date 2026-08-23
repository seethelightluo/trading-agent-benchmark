import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-11-04'
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').query('date<=@END').set_index('date')
 r=x.close.pct_change()
 # trend persistence: signed fraction of positive daily returns, centered and scaled by realized vol; all inputs through t
 pos=(r>0).rolling(20,min_periods=15).mean()-0.5
 vol=r.rolling(20,min_periods=15).std()
 # interpretable persistence-adjusted trend, higher means persistent upward path
 sig=pos/(vol+1e-12)
 D[s]=pd.DataFrame({'sig':sig,'close':x.close})
P=pd.concat({s:D[s].close for s in U},axis=1).sort_index()
S=pd.concat({s:D[s].sig for s in U},axis=1).sort_index()
for h in [1,5,10]:
 fw=P.pct_change(h).shift(-h)
 vals=[]; dates=[]; ns=[]
 for dt in S.index:
  z=pd.concat([S.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.x,z.y).statistic);dates.append(dt);ns.append(len(z))
 q=pd.Series(vals,index=pd.to_datetime(dates))
 print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 for lab,sub in [('2020-22',q[:'2022']),('2023-24',q['2023':'2024']),('2025-26',q['2025':])]:
  print(lab,len(sub),round(sub.mean(),6),round(sub.mean()/sub.std(ddof=1),6))
rank=S.rank(axis=1,pct=True)
print('coverage',round(S.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean().mean(),4),'period',S.index.min().date(),S.index.max().date())
# signal artifact for reproducibility
S.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20261105_trend_persistence_signal.csv',index=False)
