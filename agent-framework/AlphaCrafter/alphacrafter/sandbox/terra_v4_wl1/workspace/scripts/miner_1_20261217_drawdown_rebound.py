import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end].copy()
 # Drawdown-rebound: distance below trailing 60-session high, normalized by 20d volatility.
 # More negative drawdown means stronger rebound candidate; invert so high factor is larger.
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=10).std(); high=d.close.shift(1).rolling(60,min_periods=30).max()
 d['factor']=-(d.close/high-1)/(vol+1e-12)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); out=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8 and g.factor.nunique()>1 and g.y1.nunique()>1:
  z=[dt,len(g)]+[spearmanr(g.factor,g[f'y{h}']).statistic for h in [1,5,10]]; out.append(z)
a=pd.DataFrame(out,columns=['date','n','ic1','ic5','ic10'])
print('dates',len(a),'avgN',a.n.mean(),'coverage',x.factor.notna().mean())
for h in [1,5,10]:
 q=a[f'ic{h}']; print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('regimes', {h:a.assign(reg=pd.cut(a.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg',observed=True)[f'ic{h}'].mean().round(5).to_dict() for h in [1,5,10]})
r=x.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
x.to_csv('scripts/miner_1_20261217_drawdown_rebound_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
