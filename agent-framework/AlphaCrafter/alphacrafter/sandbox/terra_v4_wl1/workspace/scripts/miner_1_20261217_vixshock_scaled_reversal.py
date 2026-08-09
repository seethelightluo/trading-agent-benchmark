import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17')
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end].copy()
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=10).std()
 d['base']=-r.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+1e-12); d['y']=d.close.shift(-1)/d.close-1
 rows.append(d[['date','base','y']].assign(symbol=s))
x=pd.concat(rows)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date'); v=v[v.date<=end]
# macro state known at t: lagged VIX trend, aligned by date; use close or last column
col='close' if 'close' in v else [c for c in v.columns if c not in ('date',)][0]
v['vixshock']=v[col].pct_change(5).shift(1); v['state']=1+0.5*v.vixshock.clip(lower=0).fillna(0)
x=x.merge(v[['date','state']],on='date',how='left'); x['factor']=x.base*x.state
out=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.factor.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.factor,g.y).statistic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',len(x.dropna())/len(x))
print('regimes',a.assign(reg=pd.cut(a.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg',observed=True).ic.mean().to_dict())
r=x.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
x.to_csv('scripts/miner_1_20261217_vixshock_scaled_reversal_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
