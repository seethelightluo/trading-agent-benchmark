import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index; R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U})
# Directional persistence scaled by recent risk: signed breadth of up/down sessions divided by 20d realized volatility.
breadth=R.gt(0).rolling(12,min_periods=9).mean()-R.lt(0).rolling(12,min_periods=9).mean()
vol=R.rolling(20,min_periods=12).std()
F=(breadth/vol).shift(1)
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s].close.shift(-h).div(D[s].close).sub(1).reindex(dates) for s in U});q=[];ns=[];ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.array(q); print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   x=[q[i] for i,d in enumerate(ds) if d.year==yr];print('regime',yr,'dates',len(x),'IC',round(np.mean(x),6) if x else None,'ICIR',round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else None)
  for k in [252,504]:
   x=q[-k:];print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for n,x in {'rev5':-R.rolling(5).sum(),'mom20':R.rolling(20).sum()}.items():
 z=pd.concat([F.stack(),x.stack()],axis=1).dropna();print('corr',n,round(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),4))
