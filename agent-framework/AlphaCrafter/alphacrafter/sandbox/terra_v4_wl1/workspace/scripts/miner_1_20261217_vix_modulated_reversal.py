import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-12-17')
# Observation-only VIX modulation: all features use data through t; forecast close t to t+1.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date')
v['vix5']=v.close.pct_change(5); v['vix20']=v.vix5.rolling(60,min_periods=20).mean(); v['vixstd']=v.vix5.rolling(60,min_periods=20).std()
v['mod']=1+0.5*np.tanh((v.vix5-v.vix20)/(v.vixstd+1e-12))
v=v[['date','mod']]
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end]
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=10).std()
 d['factor']=(-r.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+1e-12)).mul(d.date.map(v.set_index('date')['mod']))
 d['y']=d.close.shift(-1)/d.close-1; rows.append(d[['date','factor','y']].assign(symbol=s))
x=pd.concat(rows); out=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.factor.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.factor,g.y).statistic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']); q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),2),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',x.dropna().shape[0]/x.shape[0])
print('regimes',a.assign(reg=pd.cut(a.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg',observed=True).ic.mean().to_dict())
r=x.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
for h in [5,10]:
 # standalone decay test using re-created forward close returns
 pass
x.to_csv('scripts/miner_1_20261217_vix_modulated_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
