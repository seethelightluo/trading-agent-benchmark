import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 r=d.close.pct_change();
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r5':d.close.pct_change(5),'vol20':r.rolling(20,min_periods=10).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
wide=x.pivot(index='date',columns='symbol',values='r5'); med=wide.median(axis=1); med[wide.count(axis=1)<8]=np.nan
x['factor']=-(x.r5-x.date.map(med))/x.vol20.replace(0,np.nan)
def calc(df):
 a=[]; ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.array(a); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
print('UNIVERSE',len(syms),'rows',len(x))
q=calc(x); print('H1 dates avgN IC ICIR hit',q)
for lo,hi,name in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]:
 q=calc(x[(x.date>=lo)&(x.date<=hi)]); print(name,q)
for h in [5,10]:
 z=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END]
  z.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 q=calc(x[['date','symbol','factor']].merge(pd.concat(z),on=['date','symbol']));print('H',h,q)
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v))
v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_volscaled_residual5_signal.csv',index=False)
