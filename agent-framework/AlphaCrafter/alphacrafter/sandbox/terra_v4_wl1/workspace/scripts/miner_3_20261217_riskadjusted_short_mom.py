import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
files={os.path.basename(x).replace('.csv',''):x for x in glob.glob('../persistent/stock_data/*.csv')}
rows=[]
for s in syms:
 p=files.get(s)
 if not p: continue
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['r']=d.close.pct_change()
 # prior completed 5-session return normalized by prior 20-session realized volatility
 d['factor']=d.close.pct_change(5).shift(1)/(d.r.rolling(20).std().shift(1)+1e-12)
 for h in [1,5,10]: d[f'fwd{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','fwd1','fwd5','fwd10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'fwd{h}'])
  if len(g)>=8: obs.append(spearmanr(g.factor,g[f'fwd{h}']).statistic); ns.append(len(g))
 a=np.array(obs); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),5),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/sum(len(z) for z in rows),4))
ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().mean(axis=1).mean(),4))
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
# regime means
x['year']=x.date.dt.year
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','fwd1'])
 if len(g)>=8: obs.append((dt,spearmanr(g.factor,g.fwd1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(year=o.date.dt.year).groupby('year').ic.agg(['mean','count']).round(5).to_string())
