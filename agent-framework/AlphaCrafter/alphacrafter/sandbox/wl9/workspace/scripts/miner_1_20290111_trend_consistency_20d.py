import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.sort_values('date').set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
F=(R.gt(0).rolling(20,min_periods=20).sum()/20 - .5)*2
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; vals=[]; dates=[]; ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); dates.append(dt); ns.append(len(a))
 z=pd.Series(vals,index=pd.to_datetime(dates)).dropna()
 def stats(x): return (round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4),len(x))
 print('H',h,'FULL',stats(z),'RECENT252',stats(z.tail(252)),'ONLINE',stats(z[z.index>='2026-07-16']))
print('coverage',round(F.notna().mean().mean(),4),'dates',len(F),'mean_n',round(np.mean(ns),2),'last',F.dropna().index[-1])
r=F.rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),4))
