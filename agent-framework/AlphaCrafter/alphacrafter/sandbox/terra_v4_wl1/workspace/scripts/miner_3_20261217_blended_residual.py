import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'r5':d.close.pct_change(5),'r7':d.close.pct_change(7),'vol20':r.rolling(20,min_periods=10).std(),'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
for k in ['r3','r5','r7']:
 w=x.pivot(index='date',columns='symbol',values=k); med=w.median(axis=1); med[w.count(axis=1)<8]=np.nan
 x[k+'f']=-(x[k]-x.date.map(med))/x.vol20.replace(0,np.nan)
x['factor']=(x.r3f+x.r5f+x.r7f)/3
def calc(y):
 a=[]; ns=[]
 for dt,g in x.assign(y=y).groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.array(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('UNIVERSE',len(syms),'rows',len(x));
for h in ['y1','y5','y10']: print(h,calc(x[h]))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]:
 q=x[(x.date>=lo)&(x.date<=hi)];
 for h in ['y1']: 
  a=[];ns=[]
  for dt,g in q.groupby('date'):
   g=g.dropna(subset=['factor',h]);
   if len(g)>=8:a.append(spearmanr(g.factor,g[h]).statistic);ns.append(len(g))
  a=np.array(a);print(n,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v))
v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_blended_residual_signal.csv',index=False)
