import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end]
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=10).std()
 d['factor']=-r.rolling(10,min_periods=10).sum()/(vol*np.sqrt(10)+1e-12)
 d['y1']=d.close.shift(-1)/d.close-1; d['y5']=d.close.shift(-5)/d.close-1
 rows.append(d[['date','factor','y1','y5']].assign(symbol=s))
x=pd.concat(rows); results={}
for col in ['y1','y5']:
 out=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',col])
  if len(g)>=8 and g.factor.nunique()>1 and g[col].nunique()>1: out.append(spearmanr(g.factor,g[col]).statistic)
 q=pd.Series(out); results[col]=(len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
print(results); print('avgN',x.dropna(subset=['factor','y1']).groupby('date').size().mean(),'coverage',x.factor.notna().mean())
r=x.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
x.to_csv('scripts/miner_2_20261217_volscaled_reversal10_signal.csv',index=False)
