import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d['signal']=-(d.close/d.close.shift(3)-1); d['fwd']=d.close.shift(-1)/d.close-1; fs[s]=d[['date','signal','fwd']]
D=sorted(set.intersection(*[set(x.date) for x in fs.values()])); rows=[]; sr=[]
for dt in D:
 v=[];y=[]
 for s,d in fs.items():
  q=d[d.date==dt]
  if len(q) and np.isfinite(q.signal.iloc[0]) and np.isfinite(q.fwd.iloc[0]): v+=[q.signal.iloc[0]];y+=[q.fwd.iloc[0]];sr.append({'date':dt,'symbol':s,'signal':q.signal.iloc[0]})
 if len(v)>=8:
  c=spearmanr(v,y).statistic
  if np.isfinite(c):rows.append((dt,c,len(v)))
r=pd.DataFrame(rows,columns=['date','ic','n']);print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/len(r)/15,'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
 x=r[(r.date.dt.year>=a)&(r.date.dt.year<=b)];print(a,b,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1))
z=pd.DataFrame(sr);p=z.pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True);print('turnover',p.diff().abs().mean().mean());out='../persistent/factor_signals_miner_1_20270226_reversal3.csv';z.to_csv(out,index=False);print(out)
