import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); intr=d.close/d.open-1
 d['factor']=intr.rolling(20,min_periods=15).mean().shift(1); d['fwd1']=d.close.shift(-1)/d.close-1; rows.append(d[['date','factor','fwd1']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); x[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_intraday_drift_signal.csv',index=False)
def calc(q):
 o=[]; n=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd1.nunique()>1:
   v=spearmanr(g.factor,g.fwd1).statistic
   if np.isfinite(v):o.append(v);n.append(len(g))
 a=np.array(o); return len(a),float(np.mean(n)),float(a.mean()),float(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))),float((a>0).mean())
print('ALL',calc(x));
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-16')]: print(label,calc(x[(x.date>=lo)&(x.date<=hi)]))
v=x.dropna(subset=['factor']); r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('coverage',len(v)/len(x),'turnover',r.diff().abs().mean(axis=1).mean(),'period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
