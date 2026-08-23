import os
import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-22'); base='../persistent/stock_data'
P={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').drop_duplicates('date'); P[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().loc[:END].ffill(); r=px.pct_change();
# Market-neutral 10-session momentum: asset return minus contemporaneous cross-sectional median, lagged one day.
med=r.median(axis=1); sig=(r.rolling(10,min_periods=8).sum().sub(med.rolling(10,min_periods=8).sum(),axis=0)).shift(1)
fwd=px.shift(-1)/px-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'rows',int(q.n.sum()),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*len(U))); print('IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
for name,mask in [('2020-22',q.index<'2023-01-01'),('2023-25',((q.index>='2023-01-01')&(q.index<'2026-01-01'))),('2026',((q.index>='2026-01-01')&(q.index<'2027-01-01'))),('2027',q.index>='2027-01-01'),('recent90',q.index>=END-pd.Timedelta(days=90))]:
 x=q.loc[mask,'ic']; print(name,len(x),('%.6f %.6f %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean())) if len(x)>2 else 'NA')
for h in [3,5,10]:
 fw=px.shift(-h)/px-1; v=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(v);print('h',h,'dates',len(x),'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std(ddof=1)))
sig.to_csv('scripts/miner_1_20270923_macro_neutral_momentum_signal.csv')
