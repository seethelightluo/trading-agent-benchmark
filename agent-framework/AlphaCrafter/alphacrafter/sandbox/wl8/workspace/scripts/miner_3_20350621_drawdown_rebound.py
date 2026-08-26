import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None:
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); frames[s]=d.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(frames).sort_index().ffill()
# Rebound score: assets furthest below their lagged 60-session peak, scaled by recent volatility.
# Every input is shifted one completed session before forward return.
peak=px.rolling(60,min_periods=40).max(); vol=px.pct_change().rolling(20,min_periods=15).std()
sig=(-(px/peak-1)/vol).shift(1)
fwd=px.shift(-10)/px-1
ics=[]; ns=[]; turns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
 x=sig.loc[dt].dropna().rank(pct=True); p=sig.shift(1).loc[dt].reindex(x.index).dropna().rank(pct=True)
 if len(p): turns.append(np.abs(x.reindex(p.index)-p).mean())
ic=pd.Series(dict(ics)).dropna()
print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR_daily_paper',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(turns))
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; oo=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: oo.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(oo),len(oo))
for n in [365,750,1260]:
 q=ic.tail(n); print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20350621_drawdown_rebound_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_3_20350621_drawdown_rebound_ic.csv',index=False)
