import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end].copy()
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=10).std()
 # five-session contrarian return scaled by recent risk; completed t signal forecasts t+1
 d['factor']=-r.rolling(5,min_periods=5).sum()/(vol*np.sqrt(5)+1e-12); d['y']=d.close.shift(-1)/d.close-1
 rows.append(d[['date','factor','y']].assign(symbol=s))
x=pd.concat(rows); out=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.factor.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.factor,g.y).statistic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',len(x.dropna())/len(x))
print('regimes',a.assign(reg=pd.cut(a.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg',observed=True).ic.mean().to_dict())
r=x.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
x.to_csv('scripts/miner_1_20261217_volscaled_reversal5_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
