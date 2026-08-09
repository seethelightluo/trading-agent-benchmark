import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2026-12-17'
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']
 px[s]=d[d.index<=end]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# relative medium-term trend: 60d cumulative return minus contemporaneous cross-sectional median
mom=p/p.shift(60)-1
f=mom.sub(mom.median(axis=1),axis=0)
fr=p.shift(-1)/p-1
ics=[]; dates=[]; turns=[]; cov=[]
prev=None
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt)
  cov.append(len(z)/15)
  rank=a.rank(pct=True)
  if prev is not None: turns.append(np.mean(abs(rank-prev)))
  prev=rank
x=np.array(ics); print('factor=relative_60d_momentum dates',len(x),'avgN',np.mean([15*c for c in cov]),'coverage',np.mean(cov),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'turnover',np.nanmean(turns))
for h in [1,5,10,20]:
 ff=p/p.shift(60)-1; ff=ff.sub(ff.median(axis=1),axis=0); fwd=p.shift(-h)/p-1; xx=[]
 for dt in ff.index:
  z=pd.concat([ff.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: xx.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'n',len(xx),'IC',np.nanmean(xx),'ICIR',np.nanmean(xx)/np.nanstd(xx,ddof=1))
# artifact
out=pd.DataFrame(f.stack(),columns=['signal']);out.index.names=['date','symbol'];out.reset_index().to_csv('scripts/miner_1_20261217_relative60_signal.csv',index=False)
