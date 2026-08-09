import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 intr=d.close/d.open-1
 # standardized intraday drift: recent mean open-close return divided by its own dispersion
 d['factor']=(intr.rolling(20,min_periods=15).mean()/intr.rolling(20,min_periods=15).std()).shift(1)
 d['fwd1']=d.close.shift(-1)/d.close-1; d['fwd5']=d.close.shift(-5)/d.close-1; d['fwd10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','fwd1','fwd5','fwd10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); x[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_scaled_intraday_signal.csv',index=False)
def calc(q,col):
 o=[]; n=[]
 for dt,g in q.groupby('date'):
  g=g.dropna(subset=['factor',col])
  if len(g)>=8 and g.factor.nunique()>1 and g[col].nunique()>1:
   v=spearmanr(g.factor,g[col]).statistic
   if np.isfinite(v):o.append(v);n.append(len(g))
 a=np.array(o); return len(a),float(np.mean(n)),float(a.mean()),float(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))),float((a>0).mean())
for c in ['fwd1','fwd5','fwd10']: print(c,calc(x,c))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-16')]: print(label,calc(x[(x.date>=lo)&(x.date<=hi)],'fwd1'))
v=x.dropna(subset=['factor']); r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('coverage',len(v)/len(x),'turnover',r.diff().abs().mean(axis=1).mean(),'period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
