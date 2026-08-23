import pandas as pd, numpy as np
from scipy.stats import spearmanr
import os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-11-13')
xs={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 d=d.loc[:end]; r=np.log(d.close).diff()
 vol=r.rolling(20).std().shift(1); ret5=np.log(d.close/d.close.shift(5)).shift(1)
 xs[a]=(-ret5/vol).rename(a)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.loc[:end]
# expanding/rolling percentile using only prior values; 252-day rolling rank, lagged
vp=v.rolling(252,min_periods=126).apply(lambda z: (z[:-1] <= z[-1]).mean() if len(z)>1 else np.nan, raw=True).shift(1)
# actually rolling apply includes current then shift ensures no lookahead
idx=sorted(set().union(*[x.index for x in xs.values()]).intersection(v.index))
F=pd.concat(xs,axis=1).reindex(idx).mul(1+0.75*vp.reindex(idx),axis=0)
prices=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in assets],axis=1).loc[:end]
fr=np.log(prices.shift(-10)/prices)
ics=[]; turns=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  turns.append((F.loc[dt].rank()-F.shift(1).loc[dt].rank()).abs().mean()/len(assets))
a=np.array(ics); print({'dates':len(a),'avg_names':round(F.notna().sum(1)[F.notna().sum(1)>=8].mean(),2),'coverage':round(F.notna().mean().mean(),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4),'turnover':round(np.nanmean(turns),6)})
for n in [180,365,756]:
 b=a[-n:]; print(n,round(b.mean(),6),round(b.mean()/b.std(ddof=1),6),len(b))
# decay same signal horizons
for h in [5,20]:
 rr=np.log(prices.shift(-h)/prices); q=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'ic',round(np.mean(q),6),'icir',round(np.mean(q)/np.std(q,ddof=1),6),'dates',len(q))
