import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-09-23'; rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); r=x.close.pct_change(); vol=r.rolling(20,min_periods=15).std(); f=r.rolling(10,min_periods=8).sum()/(vol*np.sqrt(10)+1e-12); y=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'date':x.date,'symbol':s,'f':f,'y':y}))
a=pd.concat(rows,ignore_index=True).dropna(); vals=[];ns=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append((d,spearmanr(g.f,g.y).statistic));ns.append(len(g))
z=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); q=z.ic
rank=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank'); print('candidate risk_scaled_momentum_10d cutoff',cut,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(len(a)/(sum(pd.read_csv('../persistent/stock_data/'+s+'.csv').query("date<=@cut").shape[0] for s in U)*1),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),6))
for name,lo,hi in [('early','2020','2022-12-31'),('mid','2023','2024-12-31'),('late','2025','2026-09-23')]:
 v=z.loc[lo:hi].ic; print('regime',name,len(v),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6))
for h in [3,5,10]:
 vals=[]
 for s,g in pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').assign(symbol=s) for s in U]).groupby('symbol'):
  r=g.close.pct_change(); f=r.rolling(10,min_periods=8).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(10)+1e-12); g=g.assign(f=f,y=g.close.shift(-h)/g.close-1); vals.append(g)
 aa=pd.concat(vals).dropna(); vv=[]
 for d,g in aa.groupby('date'):
  if len(g)>=8: vv.append(spearmanr(g.f,g.y).statistic)
 vv=np.array(vv);print('decay',h,len(vv),round(vv.mean(),6),round(vv.mean()/vv.std(ddof=1),6))
a[['date','symbol','f']].rename(columns={'f':'signal'}).to_csv('scripts/miner_3_20260924_risk_scaled_momentum_10d_signal.csv',index=False)
