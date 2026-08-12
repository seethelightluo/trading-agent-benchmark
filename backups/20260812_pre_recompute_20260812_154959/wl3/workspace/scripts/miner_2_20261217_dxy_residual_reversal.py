import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2026-12-16'
def ld(p): return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().query('date<=@CUT')
D={s:ld('../persistent/stock_data/'+s+'.csv') for s in U}; m=ld('../persistent/index_data/DXY.csv')['close'].pct_change()
def fac(x):
 r=x.close.pct_change(); beta=r.rolling(60,min_periods=30).cov(m)/m.rolling(60,min_periods=30).var(); e=r-beta*m
 return (-e.rolling(3,min_periods=3).sum()/(e.rolling(20,min_periods=15).std()*np.sqrt(3))).reindex(x.index)
rows=[]
for s,x in D.items(): rows.append(pd.DataFrame({'date':x.index,'f':fac(x).values,'y':x.close.shift(-1).div(x.close).sub(1).values,'s':s}))
a=pd.concat(rows).dropna(); out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); R=pd.concat([fac(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True)
print('candidate=dxy_residual_reversal_3d dates',len(z),'avg_n',z.n.mean(),'coverage',a.shape[0]/sum(len(x) for x in D.values()),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',R.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-16')]:
 q=z.loc[lo:hi].ic;print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [3,5,10]:
 b=pd.concat([pd.DataFrame({'date':x.index,'f':fac(x).values,'y':x.close.shift(-h).div(x.close).sub(1).values}) for x in D.values()]).dropna();q=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=pd.Series(q);print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
pd.concat([fac(x).rename(s) for s,x in D.items()],axis=1).stack().rename('signal').rename_axis(['date','symbol']).to_csv('scripts/miner_2_20261217_dxy_residual_reversal_signal.csv')
