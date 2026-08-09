import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 r=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r5':d.close.pct_change(5),'vol10':r.rolling(10,min_periods=5).std(),'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
wide=x.pivot(index='date',columns='symbol',values='r5'); med=wide.median(axis=1); med[wide.count(axis=1)<8]=np.nan
x['factor']=-(x.r5-x.date.map(med))/x.vol10.replace(0,np.nan)
def calc(df,col='y1'):
 a=[]; ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor',col])
  if len(g)>=8:a.append(spearmanr(g.factor,g[col]).statistic);ns.append(len(g))
 a=np.asarray(a); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
print('UNIVERSE',len(syms),'rows',len(x))
print('H1 dates avgN IC ICIR hit',calc(x))
for lo,hi,name in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]: print(name,calc(x[(x.date>=lo)&(x.date<=hi)]))
print('H5',calc(x,'y5'))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v))
v[['date','symbol','factor']].to_csv('scripts/miner_1_20261217_volscaled_residual5_vol10_signal.csv',index=False)
