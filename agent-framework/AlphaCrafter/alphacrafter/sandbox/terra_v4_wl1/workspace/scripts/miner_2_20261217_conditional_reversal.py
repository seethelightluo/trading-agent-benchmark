import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['r']=d.close.pct_change(); rows.append(d[['date','close','r']].assign(symbol=s))
x=pd.concat(rows); w=x.pivot(index='date',columns='symbol',values='r'); disp=w.std(axis=1); disp[w.count(axis=1)<8]=np.nan
med=disp.rolling(60,min_periods=30).median().shift(1); high=(disp>med).astype(float)
x=x.join(high.rename('high'),on='date'); x['factor']= -x.groupby('symbol').close.pct_change(3).shift(1)*x.high
for h in [1,5]: x[f'y{h}']=x.groupby('symbol').close.shift(-h)/x.close-1
x[['date','symbol','factor']].dropna().to_csv('../persistent/index_data/miner_2_20261217_conditional_reversal_signal.csv',index=False)
for h in [1,5]:
 a=[];ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}']);
  if len(g)>=8:a.append(spearmanr(g.factor,g[f'y{h}']).statistic);ns.append(len(g))
 a=np.array(a);print('H',h,'dates',len(a),'N',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',x.factor.notna().mean(),'turnover',x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('regimes')
o=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1']);
 if len(g)>=8:o.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(o,columns=['date','ic']);print(o.assign(reg=o.date.dt.year.map(lambda y:'2020-22' if y<=2022 else ('2023-24' if y<=2024 else '2025-26'))).groupby('reg').ic.agg(['mean','count']))
