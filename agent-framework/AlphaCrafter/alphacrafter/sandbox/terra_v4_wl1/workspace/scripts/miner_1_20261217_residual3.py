import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 d['r3']=d.close.pct_change(3); d['y1']=d.close.shift(-1)/d.close-1
 rows.append(d[['date','r3','y1']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); wide=x.pivot(index='date',columns='symbol',values='r3'); med=wide.median(axis=1); med[wide.count(axis=1)<8]=np.nan
x['factor']=-(x.r3-x.date.map(med)); obs=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append(spearmanr(g.factor,g.y1).statistic); ns.append(len(g))
a=np.asarray(obs); print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',round(len(v)/len(x),4),'turnover',round(ranks.diff().abs().mean(axis=1).mean(),4),'instruments',len(syms),'minN',min(ns),'maxN',max(ns))
v[['date','symbol','factor']].to_csv('scripts/miner_1_20261217_residual3_signal.csv',index=False)
print('artifact_rows',len(v))
