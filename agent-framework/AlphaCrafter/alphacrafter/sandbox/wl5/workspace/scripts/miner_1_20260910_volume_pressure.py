import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 x=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 # volume pressure: range-normalized close-open, scaled by abnormal volume; lag-safe
 rng=(x.high-x.low).replace(0,np.nan)
 clv=((x.close-x.open)/rng).clip(-3,3)
 av=x.volume.rolling(20,min_periods=10).mean()
 abnormal=(x.volume/(av+1e-12)).clip(0,10)
 D[s]=pd.DataFrame({'sig':clv* np.sqrt(abnormal), 'r1':x.close.pct_change().shift(-1), 'r5':x.close.pct_change(5).shift(-5), 'date':x.index})
# cross-sectional daily IC, require >=8
for h in ['r1','r5']:
 vals=[]; dates=[]; ns=[]; sigrows=[]
 all_dates=sorted(set().union(*[set(z.index) for z in D.values()]))
 for dt in all_dates:
  a=pd.Series({s:D[s].at[dt,'sig'] if dt in D[s].index else np.nan for s in U})
  b=pd.Series({s:D[s].at[dt,h] if dt in D[s].index else np.nan for s in U})
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z)); sigrows.append(a)
 q=pd.Series(vals,index=pd.to_datetime(dates))
 print(h,'dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
 for lab,sub in [('2020-22',q[:'2022']),('2023-24',q['2023':'2024']),('2025-26',q['2025':])]:
  print(' ',lab,len(sub),sub.mean(),sub.mean()/sub.std(ddof=1) if len(sub)>1 else np.nan)
# coverage and rank turnover
S=pd.DataFrame({s:D[s].sig for s in U}); ranks=S.rank(axis=1,pct=True); print('coverage',S.notna().mean().mean(),'turnover',ranks.diff().abs().mean().mean())
