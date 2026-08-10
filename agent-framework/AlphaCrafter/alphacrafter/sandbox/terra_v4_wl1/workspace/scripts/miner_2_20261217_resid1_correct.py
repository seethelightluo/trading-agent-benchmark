import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];r=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r1':r,'vol20':r.rolling(20,min_periods=15).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True);p=x.pivot(index='date',columns='symbol',values='r1'); med=p.median(axis=1);x['factor']=-(x.r1-x.date.map(med))/x.vol20.replace(0,np.nan)
def calc(q):
 a=[];ns=[]
 for dt,g in q.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('UNIVERSE',len(U),'rows',len(x),'H1',calc(x))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]:print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
v=x.dropna(subset=['factor']);r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('coverage',len(v)/len(x),'turnover',r.diff().abs().mean(axis=1).mean());v[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_resid1_signal.csv',index=False)
