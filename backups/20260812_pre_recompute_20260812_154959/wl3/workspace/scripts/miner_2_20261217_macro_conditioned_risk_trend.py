import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2026-12-16'
def load(p): return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().query('date<=@CUT')
D={s:load('../persistent/stock_data/'+s+'.csv') for s in U}
vix=load('../persistent/index_data/VIX.csv')['close'].reindex(pd.concat([x.close for x in D.values()],axis=1).index).ffill()
dxy=load('../persistent/index_data/DXY.csv')['close'].reindex(vix.index).ffill()
# Macro-conditioned risk-adjusted trend: medium trend favored, with stress-sensitive
# scaling: 20d return / 30d volatility, multiplied by 1+0.5 stress where stress is rising VIX and DXY.
stress=((vix.pct_change(5)>0)&(dxy.pct_change(5)>0)).astype(float)
def factor(x):
 r=x.close.pct_change(); vol=r.rolling(30,min_periods=20).std()*np.sqrt(20)
 return (r.rolling(20,min_periods=20).sum()/vol*(1+0.5*stress)).reindex(x.index)
rows=[]
for s,x in D.items():
 f=factor(x); y=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'date':x.index,'f':f.to_numpy(),'y':y.to_numpy(),'s':s}))
a=pd.concat(rows).dropna(); out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); alln=sum(len(x) for x in D.values()); valid=sum(factor(x).notna().sum() for x in D.values())
ranks=pd.concat([factor(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True)
print('candidate=macro_conditioned_risk_trend_20d cutoff',CUT,'dates',len(z),'avg_n',z.n.mean(),'coverage',valid/alln,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',ranks.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-16')]:
 q=z.loc[lo:hi].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [3,5,10]:
 b=pd.concat([pd.DataFrame({'date':x.index,'f':factor(x).to_numpy(),'y':(x.close.shift(-h)/x.close-1).to_numpy()}) for x in D.values()]).dropna(); q=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=pd.Series(q); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
# save signal artifact for provenance
pd.concat([factor(x).rename(s) for s,x in D.items()],axis=1).stack().rename('signal').rename_axis(['date','symbol']).to_csv('scripts/miner_2_20261217_macro_conditioned_risk_trend_signal.csv')
