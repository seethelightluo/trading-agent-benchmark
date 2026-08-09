import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').sort_values('date'); r=d.close.pct_change(); f=d.close.pct_change(20)/(r.abs().rolling(20,min_periods=15).sum()+1e-12).shift(1)
 for dt,a,b in zip(d.date,f,d.close.shift(-10)/d.close-1): rows.append((dt,s,a,b))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd10']); z=x.dropna(); q=[]; ns=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd10.nunique()>1:q.append(spearmanr(g.factor,g.fwd10).statistic);ns.append(len(g))
q=pd.Series(q); print('dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0),'coverage',len(z)/len(x))
for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
 w=z[(z.date.dt.year>=a)&(z.date.dt.year<=b)]; v=[]
 for _,g in w.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd10.nunique()>1:v.append(spearmanr(g.factor,g.fwd10).statistic)
 v=pd.Series(v);print(a,b,len(v),v.mean(),v.mean()/v.std(ddof=1))
f=x.pivot(index='date',columns='symbol',values='factor'); f.to_csv('scripts/miner_2_20261217_range_efficiency_signal.csv');print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
