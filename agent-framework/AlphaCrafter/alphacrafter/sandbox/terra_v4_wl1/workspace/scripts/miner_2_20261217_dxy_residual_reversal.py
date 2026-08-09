import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date'); macro=macro[macro.date<=END][['date','close']].rename(columns={'close':'m'}); macro['mr']=macro.m.pct_change()
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END][['date','close']]; d['r']=d.close.pct_change(); d=d.merge(macro,on='date',how='left')
 vol=d.r.rolling(20,min_periods=15).std().shift(1); beta=d.r.rolling(20,min_periods=15).cov(d.mr).shift(1)/(d.mr.rolling(20,min_periods=15).var().shift(1)+1e-12)
 d['factor']=-(d.close.pct_change(3).shift(1)-beta*d.mr.shift(1)*3)/(vol*np.sqrt(3)+1e-12)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 a=[];ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8:a.append(spearmanr(g.factor,g[f'y{h}']).statistic);ns.append(len(g))
 a=np.array(a);print(h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',x.factor.notna().mean())
rr=x.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turn',rr.diff().abs().mean(axis=1).mean())
o=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8:o.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(o,columns=['date','ic']);print(o.assign(reg=pd.cut(o.date,[pd.Timestamp('2019-12-31'),pd.Timestamp('2022-12-31'),pd.Timestamp('2024-12-31'),END],labels=['20-22','23-24','25-26'])).groupby('reg',observed=True).ic.mean())
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_2_20261217_dxy_residual_reversal_signal.csv',index=False)
