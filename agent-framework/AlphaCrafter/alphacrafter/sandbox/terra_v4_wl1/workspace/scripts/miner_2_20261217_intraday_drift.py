import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); intr=d.close/d.open-1
 d['factor']=intr.rolling(20,min_periods=15).mean().shift(1); d['fwd1']=d.close.shift(-1)/d.close-1; d['fwd5']=d.close.shift(-5)/d.close-1; d['fwd10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','fwd1','fwd5','fwd10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); x[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_intraday_drift_signal.csv',index=False)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'fwd{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'fwd{h}'].nunique()>1:
   z=spearmanr(g.factor,g[f'fwd{h}']).statistic
   if np.isfinite(z): obs.append(z); ns.append(len(g))
 a=np.asarray(obs); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']); r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',round(len(v)/len(x),4),'turnover',round(r.diff().abs().mean(axis=1).mean(),6),'period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
