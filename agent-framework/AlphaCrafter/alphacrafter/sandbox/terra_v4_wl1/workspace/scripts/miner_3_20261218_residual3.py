import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-18')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'vol20':r.rolling(20,min_periods=15).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True); p=x.pivot(index='date',columns='symbol',values='r3'); med=p.median(axis=1); x['factor']=-(x.r3-x.date.map(med))/x.vol20.replace(0,np.nan)
def calc(df):
 a=[]; ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.array(a); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
print('UNIVERSE',len(syms),'rows',len(x)); print('H1',calc(x))
for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026-12-18','25-26')]: print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 z=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];z.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 print('H',h,calc(x[['date','symbol','factor']].merge(pd.concat(z),on=['date','symbol'])))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v));v[['date','symbol','factor']].to_csv('scripts/miner_3_20261218_residual3_signal.csv',index=False)
