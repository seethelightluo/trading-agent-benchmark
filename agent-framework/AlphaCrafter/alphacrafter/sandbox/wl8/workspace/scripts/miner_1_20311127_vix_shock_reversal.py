import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2031-11-27')
px=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in assets],axis=1).sort_index().loc[:end]
r=np.log(px).diff(); vol=r.rolling(20).std().shift(1); ret5=np.log(px/px.shift(5)).shift(1)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.sort_index().loc[:end]
shock=v.pct_change().abs(); sp=shock.rolling(252,min_periods=126).apply(lambda z:(z[:-1]<=z[-1]).mean(),raw=True).shift(1)
F=(-ret5/vol).mul(1+0.75*sp.reindex(px.index).ffill(),axis=0); fr=np.log(px.shift(-10)/px); ics=[]; names=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); names.append(len(z))
a=np.array(ics); print({'dates':len(a),'avg_names':round(np.mean(names),2),'coverage':round(F.notna().mean().mean(),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4)})
for n in [180,365,756]:
 b=a[-n:]; print(n,round(b.mean(),6),round(b.mean()/b.std(ddof=1),6),len(b))
for h in [5,20]:
 rr=np.log(px.shift(-h)/px); q=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print('horizon',h,'ic',round(q.mean(),6),'icir',round(q.mean()/q.std(ddof=1),6),'dates',len(q))
