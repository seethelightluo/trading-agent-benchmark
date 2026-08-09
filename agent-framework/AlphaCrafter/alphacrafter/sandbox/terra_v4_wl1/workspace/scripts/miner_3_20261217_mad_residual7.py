import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['r']=d.close.pct_change(); d['r7']=d.close.pct_change(7).shift(1)
 # robust lagged scale: median absolute daily return around zero, avoids outlier sensitivity
 d['mad']=d.r.abs().rolling(20,min_periods=15).median().shift(1)
 d['raw']=-d.r7/(d.mad+1e-12)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','raw','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); x['factor']=x.raw-x.groupby('date').raw.transform('median')
for h in [1,5,10]:
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8:a.append(spearmanr(g.factor,g[f'y{h}']).statistic);ns.append(len(g))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']);print('coverage',round(len(v)/sum(len(z) for z in rows),4)); r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
o=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8:o.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(o,columns=['date','ic']);print(o.assign(reg=np.select([o.date.dt.year<=2022,o.date.dt.year<=2024],['2020-22','2023-24'],default='2025-26')).groupby('reg').ic.agg(['mean','count']).round(5).to_string());x[['date','symbol','factor']].dropna().to_csv('scripts/miner_3_20261217_mad_residual7_signal.csv',index=False)
