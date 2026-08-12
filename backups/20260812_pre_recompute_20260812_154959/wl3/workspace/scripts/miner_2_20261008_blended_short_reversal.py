import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-10-07'; rows=[]; total=0
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); total+=len(x); r=x.close.pct_change()
 # blended short reversal: recent 1d and 3d losses, scaled by trailing volatility
 f=-(0.6*r+0.4*r.rolling(3,min_periods=3).sum())/(r.rolling(20,min_periods=15).std()+1e-12)
 rows.append(pd.DataFrame({'date':x.date,'symbol':s,'f':f,'y':x.close.shift(-1)/x.close-1}))
a=pd.concat(rows,ignore_index=True); valid=a.dropna(); vals=[]; ns=[]
for d,g in valid.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append((d,spearmanr(g.f,g.y).statistic)); ns.append(len(g))
z=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); q=z.ic
rank=valid.assign(rank=valid.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
print('candidate blended_short_reversal cutoff',cut,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(len(valid)/total,4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),6))
for name,lo,hi in [('early','2020','2022-12-31'),('mid','2023','2024-12-31'),('late','2025','2026-10-07')]:
 v=z.loc[lo:hi].ic; print('regime',name,len(v),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6))
for h in [3,5,10]:
 b=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); r=x.close.pct_change(); f=-(0.6*r+0.4*r.rolling(3,min_periods=3).sum())/(r.rolling(20,min_periods=15).std()+1e-12); b.append(pd.DataFrame({'date':x.date,'f':f,'y':x.close.shift(-h)/x.close-1}))
 aa=pd.concat(b).dropna(); vv=[]
 for d,g in aa.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vv.append(spearmanr(g.f,g.y).statistic)
 vv=np.array(vv); print('decay',h,len(vv),round(vv.mean(),6),round(vv.mean()/vv.std(ddof=1),6))
valid[['date','symbol','f']].rename(columns={'f':'signal'}).to_csv('scripts/miner_2_20261008_blended_short_reversal_signal.csv',index=False)
