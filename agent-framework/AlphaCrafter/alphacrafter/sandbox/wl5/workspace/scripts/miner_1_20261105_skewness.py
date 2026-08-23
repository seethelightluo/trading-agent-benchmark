import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-11-04'
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
 r=x.close.pct_change(fill_method=None)
 # positive skewness / asymmetry of recent returns, lag safe
 sig=r.rolling(30,min_periods=20).skew()
 D[s]=pd.DataFrame({'sig':sig,'r1':x.close.pct_change().shift(-1),'r5':x.close.pct_change(5).shift(-5),'r10':x.close.pct_change(10).shift(-10)})
all_dates=sorted(set().union(*[set(z.index) for z in D.values()]))
S=pd.DataFrame({s:D[s].sig for s in U}).reindex(all_dates)
for h in ['r1','r5','r10']:
 vals=[]; dates=[]; ns=[]
 for dt in all_dates:
  a=pd.Series({s:D[s].at[dt,'sig'] if dt in D[s].index else np.nan for s in U})
  b=pd.Series({s:D[s].at[dt,h] if dt in D[s].index else np.nan for s in U})
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt);ns.append(len(z))
 q=pd.Series(vals,index=pd.to_datetime(dates)); print(h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
 for lab,sub in [('2020-22',q[:'2022']),('2023-24',q['2023':'2024']),('2025-26',q['2025':])]: print(lab,len(sub),round(sub.mean(),6),round(sub.mean()/sub.std(ddof=1),6))
ranks=S.rank(axis=1,pct=True); print('coverage',round(S.notna().mean().mean(),4),'turnover',round(ranks.diff().abs().mean().mean(),4),'period',S.index.min().date(),S.index.max().date())
out=S.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20261105_skewness_signal.csv',index=False)
