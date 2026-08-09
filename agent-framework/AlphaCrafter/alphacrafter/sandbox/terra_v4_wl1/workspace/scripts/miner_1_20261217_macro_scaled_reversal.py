import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17')
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date'); v=v[v.date<=end].set_index('date').close
# Continuous, lagged macro scaling: emphasize reversal when VIX is elevated, without changing its sign.
z=((v-v.rolling(60,min_periods=20).mean())/(v.rolling(60,min_periods=20).std()+1e-12)).clip(-2,2)
scale=(1+0.30*z).clip(0.4,1.6).rename('scale'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end]
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=10).std(); base=-r.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+1e-12)
 d['factor']=base*d.date.map(scale); d['y']=d.close.shift(-1)/d.close-1; rows.append(d[['date','factor','y']].assign(symbol=s))
x=pd.concat(rows); out=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.factor.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.factor,g.y).statistic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',len(x.dropna())/len(x))
print('regimes',a.assign(reg=pd.cut(a.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg').ic.mean().to_dict())
r=x.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
x.to_csv('scripts/miner_1_20261217_macro_scaled_reversal_signal.csv',index=False)
