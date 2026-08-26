import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None:
  d=d[['date','close']]; d.date=pd.to_datetime(d.date); fs[s]=d.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(fs).sort_index().ffill()
# One-day-lagged volatility-normalized 20-session reversal; designed to compare recent shock size across assets.
vol=px.pct_change().rolling(60,min_periods=40).std()*np.sqrt(252)
sig=(-px.pct_change(20))/(vol+1e-8); sig=sig.shift(1)
ics=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
ic=pd.Series(dict(ics)).dropna(); ranks=sig.rank(pct=True)
print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR_daily_paper',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',ranks.diff().abs().mean(axis=1).mean())
for h in [1,5,20]:
 oo=[]; yy=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: oo.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(oo),len(oo))
for n in [365,750,1260]:
 q=ic.tail(n); print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20350816_short_reversal_volnorm_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_2_20350816_short_reversal_volnorm_ic.csv',index=False)
