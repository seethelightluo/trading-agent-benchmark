import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end]
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=10).std()
 # five-session reversal normalized by trailing realized risk
 d['factor']=-r.rolling(5,min_periods=5).sum()/(vol*np.sqrt(5)+1e-12)
 d['y1']=d.close.shift(-1)/d.close-1; d['y5']=d.close.shift(-5)/d.close-1; d['y10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); out=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8 and g.factor.nunique()>1 and g.y1.nunique()>1: out.append((dt,spearmanr(g.factor,g.y1).statistic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']); q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),2),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',len(x.dropna(subset=['factor']))/(len(x)))
for col in ['y5','y10']:
 z=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',col])
  if len(g)>=8 and g.factor.nunique()>1 and g[col].nunique()>1:z.append(spearmanr(g.factor,g[col]).statistic)
 z=pd.Series(z); print(col,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
print('regimes',a.assign(reg=pd.cut(a.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg').ic.mean().to_dict())
r=x.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
x.to_csv('scripts/miner_2_20261217_volscaled_reversal5_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
