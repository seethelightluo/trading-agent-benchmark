import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 intr=d.close/d.open-1; rng=(d.high-d.low)/d.close.replace(0,np.nan)
 # signed pressure: close-open move scaled by own range, reversed; residualize against cross-asset daily median later
 p=intr/rng.replace(0,np.nan)
 y=d.close.pct_change().shift(-1)
 rows.append(pd.DataFrame({'date':d.date,'s':s,'p':p,'y':y}))
a=pd.concat(rows)
# remove common cross-sectional pressure on each date, then 2-day trailing residual reversal
wide=a.pivot(index='date',columns='s',values='p')
res=wide.sub(wide.median(axis=1),axis=0)
f=-res.rolling(2,min_periods=2).sum()
long=f.stack().rename('f').reset_index(); long.columns=['date','s','f']
a=a.merge(long,on=['date','s'],how='left')
out=[]
for dt,g in a.dropna(subset=['f','y']).groupby('date'):
 if len(g)>=8: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
v=np.array([x[1] for x in out]); print('dates',len(v),'avg_n',np.mean([x[2] for x in out]),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',(v>0).mean(),'coverage',a.f.notna().mean())
for h in [1,5,10]:
 # forward compounded close returns from each asset
 z=[]
 for s,g in a.groupby('s'):
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); rr=d.close.pct_change(); yy=(1+rr).rolling(h).apply(np.prod,raw=True).shift(-h)
  z.append(pd.DataFrame({'date':d.date,'s':s,'y':yy}))
 yy=pd.concat(z); q=a[['date','s','f']].merge(yy,on=['date','s']).dropna()
 vv=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8: vv.append(spearmanr(g.f,g.y).statistic)
 vv=np.array(vv); print('h',h,'dates',len(vv),'IC',vv.mean(),'ICIR',vv.mean()/vv.std(ddof=1))
print('regimes',[(yr,len([x for x in out if x[0].year==yr]),np.mean([x[1] for x in out if x[0].year==yr])) for yr in range(2020,2027)])
wide2=f.rank(axis=1,pct=True); print('turnover',wide2.diff().abs().mean(axis=1).mean())
