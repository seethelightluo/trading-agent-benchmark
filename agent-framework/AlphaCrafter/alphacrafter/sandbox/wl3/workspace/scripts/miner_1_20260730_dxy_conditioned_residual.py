import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT='2026-07-15'
def load(p):
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d[d.date<=CUT].set_index('date').close
m=load('../persistent/index_data/DXY.csv').pct_change()
# One family: DXY-residual momentum gated/scaled by lagged DXY trend.
for lb in [20,30,40]:
 rows=[]
 for s in U:
  r=load('../persistent/stock_data/'+s+'.csv').pct_change(); z=pd.concat([r,m],axis=1,join='inner'); z.columns=['r','m']
  beta=z.r.rolling(40,min_periods=30).cov(z.m)/z.m.rolling(40,min_periods=30).var()
  e=(r-beta*m)
  base=e.rolling(lb,min_periods=max(10,lb//2)).sum()
  dxy_trend=m.rolling(20,min_periods=10).sum()
  # interaction: residual momentum is rewarded when DXY trend is positive, reversed when negative
  f=base*np.sign(dxy_trend)
  q=pd.concat([f,r.shift(-1)],axis=1); q.columns=['f','y']; q['symbol']=s; rows.append(q)
 a=pd.concat(rows).reset_index().dropna()
 obs=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1: obs.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 o=pd.DataFrame(obs,columns=['date','ic','n']).dropna(); ic=o.ic.mean(); ir=ic/o.ic.std()
 # rank turnover, mean daily fraction changing rank ordering
 ranks=a.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True)
 turn=(ranks.diff().abs().mean(axis=1)>0).mean() # update frequency proxy
 print('lb',lb,'dates',len(o),'avgN',o.n.mean(),'IC',ic,'ICIR',ir,'hit',(o.ic>0).mean(),'rank_turn_proxy',turn)
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  x=o[(o.date.dt.year>=lo)&(o.date.dt.year<=hi)].ic
  print(' regime',lo,hi,'n',len(x),'ic',x.mean(),'icir',x.mean()/x.std() if len(x)>1 else np.nan)
 print('decay5', 'not computed')
